val = 800 #images will be cropped [:val, :val] in order to reduce the memory.

import argparse
import iio
import numpy as np
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import torch.nn.functional as F

from scipy.ndimage.morphology import binary_dilation
from skimage.metrics import structural_similarity

from functions import *
from model.models_8sigmas import FastDVDnet

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
        self.criterion = nn.L1Loss(reduction = 'mean')

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

    def forward(self, input1, prev_frame1, flow1, mask1_0, exclusive_mask1, input2, prev_frame2, flow2, mask2_0, exclusive_mask2,i):
        prev1 = prev_frame1.unsqueeze(0)
        prev2 = prev_frame2.unsqueeze(0)
        # Warp input on target
        warped1, mask1 = self.warp(input1, flow1)
        warped2, mask2 = self.warp(input2, flow2)
        # Compute the occlusion mask
        mask1 = self.occlusion_mask(warped1, flow1, mask1)
        mask2 = self.occlusion_mask(warped2, flow2, mask2)
        Mask1 = mask1_0*mask1
        Mask2 = mask2_0*mask2
        
        self.loss = self.criterion(exclusive_mask1*Mask1*warped1, exclusive_mask1*Mask1*prev1) + self.criterion(exclusive_mask2*Mask2*warped2, exclusive_mask2*Mask2*prev2) 
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
        print("Loading model a pre-trained gaussian FastDVDnet \n")
        
        model = FastDVDnet(num_input_frames=5)
   
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
    
    ##Freeze all the parameters 
    for param in model.parameters():
        param.requires_grad = False
    
    ut_moins_3 = iio.read(args['input'] % (args['first']))
    H, W, C = ut_moins_3.shape
    H = H if H < val else val
    W = W if W < val else val

    sigma11 = create_parameter(args['noise_level'] / 255) 
    sigma12 = create_parameter(args['noise_level'] / 255)
    sigma13 = create_parameter(args['noise_level'] / 255)
    sigma14 = create_parameter(args['noise_level'] / 255)
    sigma15 = create_parameter(args['noise_level'] / 255)
    sigma16 = create_parameter(args['noise_level'] / 255)
    sigma17 = create_parameter(args['noise_level'] / 255)
    sigma18 = create_parameter(args['noise_level'] / 255)
    sigma21 = create_parameter(args['noise_level'] / 255)
    sigma22 = create_parameter(args['noise_level'] / 255)
    sigma23 = create_parameter(args['noise_level'] / 255)
    sigma24 = create_parameter(args['noise_level'] / 255)
    sigma25 = create_parameter(args['noise_level'] / 255)
    sigma26 = create_parameter(args['noise_level'] / 255)
    sigma27 = create_parameter(args['noise_level'] / 255)
    sigma28 = create_parameter(args['noise_level'] / 255)
    sigma31 = create_parameter(args['noise_level'] / 255)
    sigma32 = create_parameter(args['noise_level'] / 255)
    sigma33 = create_parameter(args['noise_level'] / 255)
    sigma34 = create_parameter(args['noise_level'] / 255)
    sigma35 = create_parameter(args['noise_level'] / 255)
    sigma36 = create_parameter(args['noise_level'] / 255)
    sigma37 = create_parameter(args['noise_level'] / 255)
    sigma38 = create_parameter(args['noise_level'] / 255)

    #################
    # DEFINE THE LOSS
    #################

    # The loss needs to be changed when used with different networks
    lr = args['lr']
    
    weight_decay = 0.00001
    
    criterion = Loss() 
    criterion.cuda()
    optimizer = optim.Adam([{'params': [sigma11]}, {'params':[sigma12]}, {'params':[sigma13]}, {'params':[sigma14]}, {'params':[sigma15]}, {'params':[sigma16]}, {'params':[sigma17]}, {'params':[sigma18]}, {'params': [sigma21]}, {'params':[sigma22]}, {'params':[sigma23]}, {'params':[sigma24]}, {'params':[sigma25]}, {'params':[sigma26]}, {'params':[sigma27]}, {'params':[sigma28]}, {'params': [sigma31]}, {'params':[sigma32]}, {'params':[sigma33]}, {'params':[sigma34]}, {'params':[sigma35]}, {'params':[sigma36]}, {'params':[sigma37]}, {'params':[sigma38]}], lr=lr, betas=(0.2, 0.2), eps=1e-08, weight_decay=weight_decay, amsgrad=False)


    #####   Useful thinks   #####

    list_PSNR_training = []
    list_PSNR_eval     = []
    
    #Initialisation

    frame = iio.read(args['input'] % (args['first']))

    H, W, C = frame.shape
    H = H if H < val else val
    W = W if W < val else val
    
    # Write the psnr per frame in this file
    output_path = os.path.dirname(args['output']) +"/"

    path_psnr          = output_path + "PSNR.txt"
    path_ssim          = output_path + "SSIM.txt"
    path_training      = output_path + "PSNR_training.txt"
    path_ssim_training = output_path + "SSIM_training.txt"

    plot_psnr          = open(path_psnr, 'w')
    plot_ssim          = open(path_ssim, 'w')
    plot_psnr_training = open(path_training, 'w')
    plot_ssim_training = open(path_ssim_training, 'w')

    ###########
    # MAIN LOOP
    ###########
    for i in range(args['first']+4, args['last']-3):

        ut_moins_4 = reads_image(args['input'] % (i-4), H, W)
        ut_moins_3 = reads_image(args['input'] % (i-3), H, W)
        ut_moins_2 = reads_image(args['input'] % (i-2), H, W)
        ut_moins_1 = reads_image(args['input'] % (i-1), H, W)
        ut         = reads_image(args['input'] % (i  ), H, W)
        ut_plus_1  = reads_image(args['input'] % (i+1), H, W) 
        ut_plus_2  = reads_image(args['input'] % (i+2), H, W)
        ut_plus_3  = reads_image(args['input'] % (i+3), H, W)
        ut_plus_4  = reads_image(args['input'] % (i+4), H, W)

        #Creation of the stack
        if i%2==(args['first']%2):
                
            inframes = [ut_moins_4, ut_moins_2, ut, ut_plus_2, ut_plus_4]
            stack1 = torch.stack(inframes, dim=0).contiguous().view((1, 5*C, H, W)).cuda()
            stack1.requires_grad = False
            stack = stack1
            
            flow1 = gives_flow(args['flow'] % (i-1), H, W)
            mask1, exclusive_mask1 = gives_masks(args['mask_collision']%(i-1), args['mask_warping_res']%(i-1), H, W)

        else:
            inframes = [ut_moins_4, ut_moins_2, ut, ut_plus_2, ut_plus_4]
            stack2 = torch.stack(inframes, dim=0).contiguous().view((1, 5*C, H, W)).cuda()
            stack2.requires_grad = False
            stack = stack2

            flow2 = gives_flow(args['flow'] % (i-1), H, W)
            mask2, exclusive_mask2 = gives_masks(args['mask_collision']%(i-1), args['mask_warping_res']%(i-1), H, W)

            model.eval()
            optimizer.zero_grad()

            for it in range(args['iter']):
                ##Define noise_map depending on luminosity
                u1, u2, u3, u4, u5, u6, u7, u8 = find_brightness(ut_moins_1)
                noise_map_moins_1 = build_variance_map(u1, u2, u3, u4, u5, u6, u7, u8, sigma11, sigma12, sigma13, sigma14, sigma15, sigma16, sigma17, sigma18)
       
                u1, u2, u3, u4, u5, u6, u7, u8 = find_brightness(ut)
                noise_map = build_variance_map(u1, u2, u3, u4, u5, u6, u7, u8, sigma21, sigma22, sigma23, sigma24, sigma25, sigma26, sigma27, sigma28)
        
                u1, u2, u3, u4, u5, u6, u7, u8 = find_brightness(ut_plus_1)
                noise_map_plus_1 = build_variance_map(u1, u2, u3, u4, u5, u6, u7, u8, sigma31, sigma32, sigma33, sigma34, sigma35, sigma36, sigma37, sigma38)
        
                optimizer.zero_grad()   
                out_train1 = temp_denoise_8_sigmas(model, stack1, noise_map_moins_1, noise_map, noise_map_plus_1)
                out_train2 = temp_denoise_8_sigmas(model, stack2, noise_map_moins_1, noise_map, noise_map_plus_1)
                loss = criterion(out_train1, ut_moins_2, flow1, mask1, exclusive_mask1, out_train2, ut_moins_1, flow2, mask2, exclusive_mask2, i) 
                loss.backward()
                optimizer.step()
                del loss

        ##Define noise_map depending on luminosity
        u1, u2, u3, u4, u5, u6, u7, u8 = find_brightness(ut_moins_2)
        noise_map_moins_1 = build_variance_map(u1, u2, u3, u4, u5, u6, u7, u8, sigma11, sigma12, sigma13, sigma14, sigma15, sigma16, sigma17, sigma18)
        
        u1, u2, u3, u4, u5, u6, u7, u8 = find_brightness(ut)
        noise_map = build_variance_map(u1, u2, u3, u4, u5, u6, u7, u8, sigma21, sigma22, sigma23, sigma24, sigma25, sigma26, sigma27, sigma28)
        
        u1, u2, u3, u4, u5, u6, u7, u8 = find_brightness(ut_plus_2)
        noise_map_plus_1 = build_variance_map(u1, u2, u3, u4, u5, u6, u7, u8, sigma31, sigma32, sigma33, sigma34, sigma35, sigma36, sigma37, sigma38)

        #Compute and save the denoising
        model.eval()
        with torch.no_grad():
            #denoise with training stack : 
            outimg = temp_denoise_8_sigmas(model, stack, noise_map_moins_1, noise_map, noise_map_plus_1)
            outimg = torch.clamp(outimg, 0, 1)  
            outimg = np.array(outimg.cpu())
            outimg = np.squeeze(outimg)
            outimg = outimg.transpose(1, 2, 0)

            #denoise with the natural stack

            inframes = [ut_moins_2, ut_moins_1, ut, ut_plus_1, ut_plus_2]
            stack = torch.stack(inframes, dim=0).contiguous().view((1, 5*C, H, W)).cuda()
            outimg2 = temp_denoise_8_sigmas(model, stack, noise_map_moins_1, noise_map, noise_map_plus_1)
            outimg2= torch.clamp(outimg2, 0, 1) 
            outimg2= np.array(outimg2.cpu())
            outimg2= np.squeeze(outimg2)
            outimg2= outimg2.transpose(1, 2, 0)

            
        #store the results
        iio.write(output_path + "training{:03d}.png".format(i), (outimg*255))
        iio.write(args['output']%i, (outimg2*255))

        # Load frame to compute the PSNR
        ref_frame = iio.read(args['ref'] % (i))[:val, :val, :] 

        # Compute the PSNR according to the reference frame
        quant_training_stack = psnr(ref_frame.astype(outimg.dtype)/255, outimg)
        quant_eval_stack = psnr(ref_frame.astype(outimg2.dtype)/255., outimg2)
        if quant_eval_stack > quant_training_stack:
            value = 1
        else:
            value = 0
        
        ssim_training = ssim(outimg*255, ref_frame)
        ssim_eval = ssim(outimg2*255, ref_frame)
        print("Iteration = {:2d}, PSNR training stack = {:5.3f}, PSNR eval stack = {:5.3f}, SSIM training stack {:4.3f}, SSIM eval stack = {:4.3f} {:1d}".format(i, quant_training_stack, quant_eval_stack, ssim_training, ssim_eval, value))

        list_PSNR_training.append(quant_training_stack)
        list_PSNR_eval.append(quant_eval_stack)
        plot_psnr.write(str(quant_eval_stack)+'\n')
        plot_ssim.write(str(ssim_eval)+'\n')
        plot_psnr_training.write(str(quant_training_stack)+'\n')
        plot_ssim_training.write(str(ssim_training)+'\n')
        
    tab_PSNR_training = np.array(list_PSNR_training)
    tab_PSNR_eval     = np.array(list_PSNR_eval)
    print("Average PSNR: training stack = {:5.3f}, eval stack = {:5.3f}".format(np.mean(tab_PSNR_training), np.mean(tab_PSNR_eval)))

    torch.save([model, optimizer], output_path + args['output_network_after_training'])
    plot_psnr.close()
    plot_ssim.close()
    plot_psnr_training.close()
    plot_ssim_training.close()


if __name__ == "__main__":
    # Parse arguments
    parser = argparse.ArgumentParser(description="MF2F teacher 8sigmas")
    parser.add_argument("--input"                        , type=str  , default=""               , help='path to input frames (C type)')
    parser.add_argument("--ref"                          , type=str  , default=""               , help='path to reference frames (C type), against which the psnr is going to be computed')
    parser.add_argument("--flow"                         , type=str  , default=""               , help='path to optical flow (C type)')
    parser.add_argument("--mask_collision"               , type=str  , default=""               , help='path to collision masks (C type)')
    parser.add_argument("--mask_warping_res"             , type=str  , default=""               , help='path to masks based on warping residues (C type)')
    parser.add_argument("--output"                       , type=str  , default="./%03d.png"     , help='path to output image (C type)')
    parser.add_argument("--output_network_after_training", type=str  , default="final_mf2f.pth" , help='path to output network')
    parser.add_argument("--first"                        , type=int  , default=1                , help='index first frame')
    parser.add_argument("--last"                         , type=int  , default=30               , help='index last frame')
    parser.add_argument("--iter"                         , type=int  , default=20               , help='number of time the learning is done on a given frame')
    parser.add_argument("--network"                      , type=str  , default="model/model.pth", help='path to the network')
    parser.add_argument("--noise_level"                  , type=int  , default=25               , help='sigma standard deviation of the noise')
    parser.add_argument("--lr"                           , type=float, default=0.008            , help='learning rate')

    argspar = parser.parse_args()

    print("\n### MF2F teacher 8 sigmas ###")
    print("> Parameters:")
    for p, v in zip(argspar.__dict__.keys(), argspar.__dict__.values()):
        print('\t{}: {}'.format(p, v))
    print('\n')

    MF2F(**vars(argspar))
