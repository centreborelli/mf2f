val = 800 #images will be cropped [:val, :val] in order to reduce the memory.

import argparse
import iio
import numpy as np
import os
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

from scipy.ndimage.morphology import binary_dilation

from model.models import FastDVDnet

from functions import *

cuda = torch.cuda.is_available()
if cuda:
    device = torch.device('cuda')
    print("CUDA is available")

#interp = 'bilinear'
interp = 'bicubic'
class Loss(nn.Module):
    fff = 0
    def __init__(self):
        super(Loss, self).__init__()
        self.criterion = nn.L1Loss(reduce=True)

    def cubic_interpolation(self, A, B, C, D, x):
        a,b,c,d = A.size()
        x = x.view(a,1,c,d).repeat(1,3,1,1)
        return B + 0.5*x*(C - A + x*(2.*A - 5.*B + 4.*C - D + x*(3.*(B - C) + D - A)))

    def bicubic_interpolation(self, im, grid):
        # Assume B == 1
        B, C, H, W = im.size()

        x0 = torch.floor(grid[0, 0, :, :] - 1).long()
        y0 = torch.floor(grid[0, 1, :, :] - 1).long()
        x1 = x0 + 1
        y1 = y0 + 1
        x2 = x0 + 2
        y2 = y0 + 2
        x3 = x0 + 3
        y3 = y0 + 3

        outsideX = torch.max(torch.gt(-x0, 0), torch.gt(x3, W-1)).repeat((B,C,1,1))
        outsideY = torch.max(torch.gt(-y0, 0), torch.gt(y3, H-1)).repeat((B,C,1,1))

        mask = torch.Tensor(np.ones((B, C, H, W))).cuda()
        mask[outsideX] = 0
        mask[outsideY] = 0

        x0 = x0.clamp(0, W-1)
        y0 = y0.clamp(0, H-1)
        x1 = x1.clamp(0, W-1)
        y1 = y1.clamp(0, H-1)
        x2 = x2.clamp(0, W-1)
        y2 = y2.clamp(0, H-1)
        x3 = x3.clamp(0, W-1)
        y3 = y3.clamp(0, H-1)

        A = self.cubic_interpolation(im[:, :, y0, x0], im[:, :, y1, x0], im[:, :, y2, x0],
                                     im[:, :, y3, x0], grid[:, 1, :, :] - torch.floor(grid[:, 1, :, :]))
        B = self.cubic_interpolation(im[:, :, y0, x1], im[:, :, y1, x1], im[:, :, y2, x1],
                                     im[:, :, y3, x1], grid[:, 1, :, :] - torch.floor(grid[:, 1, :, :]))
        C = self.cubic_interpolation(im[:, :, y0, x2], im[:, :, y1, x2], im[:, :, y2, x2],
                                     im[:, :, y3, x2], grid[:, 1, :, :] - torch.floor(grid[:, 1, :, :]))
        D = self.cubic_interpolation(im[:, :, y0, x3], im[:, :, y1, x3], im[:, :, y2, x3],
                                     im[:, :, y3, x3], grid[:, 1, :, :] - torch.floor(grid[:, 1, :, :]))
        return self.cubic_interpolation(A, B, C, D, grid[:, 0, :, :] - torch.floor(grid[:, 0, :, :])), mask

    #  Function heavily inspired by the code for PWCnet by JinWei GU and Zhile Ren available at https://github.com/NVlabs/PWC-Net
    # Warps the image 'x' using the optical flow defined by 'flo'
    def warp(self, x, flo):
        """
        warp an image/tensor (im2) back to im1, according to the optical flow
        x: [B, C, H, W] (im2)
        flo: [B, 2, H, W] flow
        """
        B, C, H, W = x.size()
        # mesh grid
        xx = torch.arange(0, W).view(1, -1).repeat(H, 1)
        yy = torch.arange(0, H).view(-1, 1).repeat(1, W)
        xx = xx.view(1, 1, H, W).repeat(B, 1, 1, 1)
        yy = yy.view(1, 1, H, W).repeat(B, 1, 1, 1)
        grid = torch.cat((xx, yy), 1).float()

        grid = grid.cuda()
        vgrid = Variable(grid) + flo.cuda()

        if interp == 'bicubic':
            output, mask = self.bicubic_interpolation(x, vgrid)
        else:  # if interp == 'bilinear':
            # scale grid to [-1,1]
            vgrid[:, 0, :, :] = 2.0*vgrid[:, 0, :, :]/max(W-1, 1)-1.0
            vgrid[:, 1, :, :] = 2.0*vgrid[:, 1, :, :]/max(H-1, 1)-1.0
            vgrid = vgrid.permute(0, 2, 3, 1)
            output = nn.functional.grid_sample(x, vgrid)
            # Define a first mask containing pixels that wasn't properly interpolated
            mask = torch.autograd.Variable(torch.ones(x.size()))
            mask = nn.functional.grid_sample(mask.cuda(), vgrid)
            mask[mask < 0.9999] = 0
            mask[mask > 0] = 1

        return output, mask

    # Computes the occlusion map based on the optical flow
    def occlusion_mask(self, warped, of, old_mask):
        """
        Computes an occlusion mask based on the optical flow
        warped: [B, C, H, W] warped frame (only used for size)
        of: [B, 2, H, W] flow
        old_mask: [B, C, H, W] first estimate of the mask
        """
        B,C,H,W = warped.size() # Suppose B==1
        a = np.zeros((1,1,H,W))
        b = np.zeros((1,1,H,W))

        a[:, :, :-1, :] = (of[0, 0, 1:, :] - of[0, 0, :-1, :])
        b[:, :, :, :-1] = (of[0, 1, :, 1:] - of[0, 1, :, :-1])
        mask = np.abs(a + b) > 0.75

        #mask = old_mask[0,0,:,:].detach().cpu().numpy()
        #mask = 1 - mask
        #mask = np.expand_dims(mask,0)
        #mask = np.expand_dims(mask,0)

        if interp == 'bicubic':
            # Slighlty dilates the occlusion map to remove pixels estimated with wrong values
            # bicubic interpolation uses a 4x4 kernel
            boule = np.ones((4, 4))
            mask[0, 0, :, :] = binary_dilation(mask[0, 0, :, :], boule)

            # Remove the boundaries (values extrapolated on the boundaries)
            mask[:, :, 1, :] = 1
            mask[:, :, mask.shape[2]-2, :] = 1
            mask[:, :, :, 1] = 1
            mask[:, :, :, mask.shape[3]-2] = 1
            mask[:, :, 0, :] = 1
            mask[:, :, mask.shape[2]-1, :] = 1
            mask[:, :, :, 0] = 1
            mask[:, :, :, mask.shape[3]-1] = 1
        else:
            # Slighlty dilates the occlusion map to remove pixels estimated with wrong values
            # bilinear interpolation uses a 2x2 kernel
            boule = np.ones((2, 2))
            mask[0, 0, :, :] = binary_dilation(mask[0, 0, :, :], boule)

            # Remove the boundaries (values extrapolated on the boundaries)
            mask[:, :, 0, :] = 1
            mask[:, :, mask.shape[2]-1, :] = 1
            mask[:, :, :, 0] = 1
            mask[:, :, :, mask.shape[3]-1] = 1

        # Invert the mask because we want a mask of good pixels
        mask = torch.Tensor(1-mask).cuda()
        mask = mask.view(1,1,H,W).repeat(1,C,1,1)
        mask = old_mask * mask
        return mask
        #return torch.ones(warped.size()).cuda()

    def forward(self, input1, input2, prev_frame1, flow1, mask1_0, exclusive_mask1, fastdvdnet):
        prev1 = prev_frame1.unsqueeze(0)
        # Warp input on target
        warped1, mask1 = self.warp(input1, flow1)
        # Compute the occlusion mask
        mask1 = self.occlusion_mask(warped1, flow1, mask1)
        Mask1 = mask1_0*mask1

        warp1, m1 = self.warp(input2, flow1)
        m1 = self.occlusion_mask(warp1, flow1, m1)
        wfastdvdnet, m11 = self.warp(fastdvdnet, flow1)
        m11 = self.occlusion_mask(wfastdvdnet, flow1, m11)
        m1 = m1 * m11
        
        self.loss = self.criterion(exclusive_mask1*Mask1*warped1, exclusive_mask1*Mask1*prev1) + self.criterion( (m1*exclusive_mask1*(1.-Mask1)*warp1), (m1*exclusive_mask1*(1.-Mask1)*wfastdvdnet) )
        
        return self.loss


def MF2F(**args):
    """
    Main function
    args: Parameters
    """

    ################
    # LOAD THE MODEL
    ################
    if args['network'] == "model/model.pth":
        print("Loading model a pre-trained FastDVDnet \n")
        
        model = FastDVDnet( num_input_frames=5)
   
         #Load saved weights
        state_temp_dict = torch.load(args['network'])

        if cuda:
            device = torch.device("cuda")
            device_ids = [0]
            model = nn.DataParallel(model, device_ids = device_ids).cuda()

        model.load_state_dict(state_temp_dict)
    
    else:
        model_fn = args['network']
        model_fn = os.path.join(os.path.abspath(os.path.dirname(__file__)), model_fn)

        model = torch.load(model_fn)[0]
        model.cuda()

        device=torch.device("cuda")

    #################
    # DEFINE THE LOSS
    #################

    # The loss needs to be changed when used with different networks
    lr = args['lr']
    weight_decay = 0.00001
    
    criterion = Loss() 
    criterion.cuda()
    optimizer = optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=weight_decay, amsgrad=False)

    #####   Useful thinks   #####

    list_PSNR_training = []
    list_PSNR_eval = []
    
    #Initialisation

    frame = iio.read(args['input'] % (args['first']))

    H, W, C = frame.shape
    H = H if H < val else val
    W = W if W < val else val
    noise_std = Variable(torch.FloatTensor(np.array(args['noise_level']/255.)))
    noise_map = noise_std.expand((1, 1, H, W))
    
    # Write the psnr per frame in this file
    output_path = os.path.dirname(args['output']) +"/"
    path_psnr          = output_path + "PSNR.txt"
    path_ssim          = output_path + "SSIM.txt"
    path_psnr_tot      = output_path + "PSNR_total.txt"
    path_ssim_tot      = output_path + "SSIM_total.txt"
    path_psnr_training = output_path + "PSNR_training.txt"
    path_ssim_training = output_path + "SSIM_training.txt"
    plot_psnr          = open(path_psnr, 'w')
    plot_ssim          = open(path_ssim, 'w')
    plot_psnr_tot      = open(path_psnr_tot, 'w')
    plot_ssim_tot      = open(path_ssim_tot, 'w')
    plot_psnr_training = open(path_psnr_training, 'w')
    plot_ssim_training = open(path_ssim_training, 'w')

    optimizer.zero_grad()

    ###########
    # MAIN LOOP
    ###########
    for training in range(args['nb_trainings']):
        i = np.random.randint(args['first']+4, args['last']-3) #index of center frame (ut use for the training)

        ut_moins_4 = reads_image(args['input'] % (i-4), H, W)
        ut_moins_3 = reads_image(args['input'] % (i-3), H, W)
        ut_moins_2 = reads_image(args['input'] % (i-2), H, W)
        ut_moins_1 = reads_image(args['input'] % (i-1), H, W)
        ut         = reads_image(args['input'] % (i),   H, W)
        ut_plus_1  = reads_image(args['input'] % (i+1), H, W)
        ut_plus_2  = reads_image(args['input'] % (i+2), H, W)
        ut_plus_3  = reads_image(args['input'] % (i+3), H, W)
        ut_plus_4  = reads_image(args['input'] % (i+4), H, W)

        #Creation of the stack
                
        inframes = [ut_moins_4, ut_moins_2, ut, ut_plus_2, ut_plus_4]
        stack = torch.stack(inframes, dim=0).contiguous().view((1, 5*C, H, W)).cuda()
        
        flow                 = gives_flow(args['flow'] % (i-1), H, W)
        mask, exclusive_mask = gives_masks(args['mask_collision']%(i-1), args['mask_warping_res']%(i-1), H, W)

        inframes_ori = [ut_moins_2, ut_moins_1, ut, ut_plus_1, ut_plus_2]
        stack_ori = torch.stack(inframes_ori, dim=0).contiguous().view((1, 5*C, H, W)).cuda()
        fastdvdnet = iio.read(args['teacher_outputs']%i)[:val, :val] / 255.
        fastdvdnet = fastdvdnet.transpose(2,0,1)
        fastdvdnet = np.expand_dims(fastdvdnet, 0)
        fastdvdnet = torch.tensor(fastdvdnet).cuda()

        model.eval()

        out_train1 = temp_denoise(model, stack    , noise_map)
        out_train2 = temp_denoise(model, stack_ori, noise_map)
        loss = criterion(out_train1, out_train2, ut_moins_1, flow, mask, exclusive_mask, fastdvdnet) 
        loss.backward()

        ## Do the backward and step everyloss.backward()
        if training%args['nb_trainings_before_step'] == 0 and training>=1:
            optimizer.step()
            del loss
            optimizer.zero_grad()
            color = '\033[1;31;40m'
        else:
            color = '\033[0;37;40m' 
        
        #Compute and save the denoising
        model.eval()
        with torch.no_grad():
            #denoise with the training stack : 
            outimg = temp_denoise(model, stack, noise_map)
            outimg = tensor_to_image(outimg)

            #denoise with the natural stack
            inframes = [ut_moins_2, ut_moins_1, ut, ut_plus_1, ut_plus_2]
            stack = torch.stack(inframes, dim=0).contiguous().view((1, 5*C, H, W)).cuda()
            outimg2 = temp_denoise(model, stack, noise_map)
            outimg2 = tensor_to_image(outimg2)

        
        # Load frame to compute the PSNR
        ref_frame = iio.read(args['ref'] % (i))[:val, :val, :] 

        # Compute the PSNR according to the reference frame
        quant_our_stack = psnr(ref_frame.astype(outimg.dtype)/255, outimg)
        quant_Tassano_stack = psnr(ref_frame.astype(outimg2.dtype)/255., outimg2)
        if quant_Tassano_stack > quant_our_stack:
            value = 1
        else:
            value = 0

        ssim_our     = ssim(outimg*255 , ref_frame)
        ssim_Tassano = ssim(outimg2*255, ref_frame)
        
        plot_psnr_tot.write(str(quant_Tassano_stack)+'\n')
        plot_ssim_tot.write(str(ssim_Tassano)+'\n')

        print(color + "Paires = {:03d}-{:03d}, PSNR our stack = {:5.3f}, PSNR Tassano's stack = {:5.3f}, SSIM our {:4.3f}, SSIM Tassano's = {:4.3f}   {:1d}    {:4d}/{:04d}".format(i-1, i, quant_our_stack, quant_Tassano_stack, ssim_our, ssim_Tassano, value, training, args['nb_trainings']))
       

    ## Save the offline fine-tuned network    
    torch.save([model, optimizer], output_path + "final_mf2f.pth")

    print("")
    print("")
    print("     Process the entire video with the final offline fine-tuned network")

    for i in range(args['first']+4, args['last']-3):

        ut_moins_4 = reads_image(args['input'] % (i-4), H, W)
        ut_moins_3 = reads_image(args['input'] % (i-3), H, W)
        ut_moins_2 = reads_image(args['input'] % (i-2), H, W)
        ut_moins_1 = reads_image(args['input'] % (i-1), H, W)
        ut         = reads_image(args['input'] % (i),   H, W)
        ut_plus_1  = reads_image(args['input'] % (i+1), H, W)
        ut_plus_2  = reads_image(args['input'] % (i+2), H, W)
        ut_plus_3  = reads_image(args['input'] % (i+3), H, W)
        ut_plus_4  = reads_image(args['input'] % (i+4), H, W)

        #Creation of the stack
                
        inframes = [ut_moins_4, ut_moins_2, ut, ut_plus_2, ut_plus_4]
        stack = torch.stack(inframes, dim=0).contiguous().view((1, 5*C, H, W)).cuda()
        
        #Compute and save the denoising
        model.eval()
        with torch.no_grad():
            #denoise with the training stack : 
            outimg = temp_denoise(model, stack, noise_map)
            outimg = tensor_to_image(outimg)

            #denoise with the natural stack
            inframes = [ut_moins_2, ut_moins_1, ut, ut_plus_1, ut_plus_2]
            stack = torch.stack(inframes, dim=0).contiguous().view((1, 5*C, H, W)).cuda()
            outimg2 = temp_denoise(model, stack, noise_map)
            outimg2 = tensor_to_image(outimg2)
        
        # Load frame to compute the PSNR
        ref_frame = iio.read(args['ref'] % (i))[:val, :val, :] 

        # Compute the PSNR according to the reference frame
        quant_our_stack = psnr(ref_frame.astype(outimg.dtype)/255, outimg)
        quant_Tassano_stack = psnr(ref_frame.astype(outimg2.dtype)/255., outimg2)
        if quant_Tassano_stack > quant_our_stack:
            value = 1
        else:
            value = 0
        
        ssim_our     = ssim(outimg*255 , ref_frame)
        ssim_Tassano = ssim(outimg2*255, ref_frame)

        print("Itération = {:03d}, PSNR our stack = {:5.3f}, PSNR Tassano's stack = {:5.3f}, SSIM our {:4.3f}, SSIM Tassano's = {:4.3f}   {:1d}".format(i, quant_our_stack, quant_Tassano_stack, ssim_our, ssim_Tassano, value))
        
        # Store the results
        iio.write(output_path + "training_{:03d}".format(i), 255*outimg) 
        iio.write(args['output']%(i), 255*outimg2) 

        list_PSNR_training.append(quant_our_stack)
        list_PSNR_eval.append(quant_Tassano_stack)
        plot_psnr.write(str(quant_Tassano_stack)+'\n')
        plot_ssim.write(str(ssim_Tassano)+'\n')
        plot_psnr_training.write(str(quant_our_stack)+'\n')
        plot_ssim_training.write(str(ssim_our)+'\n')
        
    tab_PSNR_training = np.array(list_PSNR_training)
    tab_PSNR_eval     = np.array(list_PSNR_eval)
    print("Average PSNR: training stack = {:5.3f}, natural stack = {:5.3f}".format(np.mean(tab_PSNR_training), np.mean(tab_PSNR_eval)))
    plot_psnr.close()
    plot_ssim.close()
    plot_psnr_tot.close()
    plot_ssim_tot.close()
    plot_psnr_training.close()
    plot_ssim_training.close()

if __name__ == "__main__":
    # Parse arguments
    parser = argparse.ArgumentParser(description="MF2F offline with teacher")
    parser.add_argument("--input"                   , type=str  , default=""               , help='path to input frames (C type)') 
    parser.add_argument("--ref"                     , type=str  , default=""               , help='path to reference frames (C type), against which the psnr is going to be computed')
    parser.add_argument("--flow"                    , type=str  , default=""               , help='path to optical flow (C type)')
    parser.add_argument("--mask_collision"          , type=str  , default=""               , help='path to collision mask (C type)')
    parser.add_argument("--mask_warping_res"        , type=str  , default=""               , help='path to warping res mask (C type)')
    parser.add_argument("--teacher_outputs"         , type=str  , default=""               , help='path to teacher outputs (have to be previously precomputed) (C type)')
    parser.add_argument("--output"                  , type=str  , default="./%03d.png"     , help='path to output image (C type)')
    parser.add_argument("--first"                   , type=int  , default=1                , help='index first frame')
    parser.add_argument("--last"                    , type=int  , default=300              , help='index last frame')
    parser.add_argument("--nb_trainings"            , type=int  , default=4002             , help='number of trainings')
    parser.add_argument("--nb_trainings_before_step", type=int  , default=20               , help='number of trainings before each step')
    parser.add_argument("--network"                 , type=str  , default="model/model.pth", help='path to the network')
    parser.add_argument("--noise_level"             , type=int  , default=25               , help='sigma standard deviation of the noise')
    parser.add_argument("--lr"                      , type=float, default=0.00001          , help='learning rate')

    argspar = parser.parse_args()

    print("\n### MF2F offline with teacher ###")
    print("> Parameters:")
    for p, v in zip(argspar.__dict__.keys(), argspar.__dict__.values()):
        print('\t{}: {}'.format(p, v))
    print('\n')

    MF2F(**vars(argspar))
