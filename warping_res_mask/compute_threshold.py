# Modules
import argparse
import iio
import numpy as np
import os
from scipy import stats

##Functions
    
diff   = lambda x : x[1:] - x[:-1]
length = lambda x : np.max(np.shape(x))
    
##Script

def compute_mask(**args):
    boundaries_removal = 20 #number of pixel to remove at the frame borders (so that the warping errors don't affect the statistics)
    taus_size = 201
    #length_taus = np.max(np.shape(taus))
    length_taus = taus_size

    taus = np.linspace(0, 20, taus_size)
    other_taus = taus[:-1]

    for f in range(args['first'], args['last']):
        #compuute the warping errors map + the mask of invalid pixels and then store them
        noisy1 = args['input']%f + ' '
        noisy2 = args['input']%(f+1) + ' '
        flow = args['flow']%(f) + ' '
        if args['dummy'][-1]=='/':
            out_path = args['dummy'][:-1] 
        else:
            out_path = args['dummy'] 
        
        os.system('bash compute_warping_error.sh ' + noisy1 + noisy2 + flow + out_path)

        k                   = iio.read(out_path+'/warping_error.tiff') # k is in fact the warping_errors
        os.system('cp ' + out_path+'/WERR.tiff ' + os.path.dirname(args['output'])+"/warping_error{:03d}.png".format(f)) 
        mask_invalid_pixels = iio.read(out_path+'/mask_invalid_pixels.png')

        #k = warping_errors[boundaries_removal:-boundaries_removal, boundaries_removal:-boundaries_removal]
        
       #slow_version = True
       #npx = np.zeros(taus_size)
       #if slow_version:
       #    for i in range(0, length_taus):
       #        npx[i] = np.sum( k<taus[i])
       #        
       #    dd = diff(npx)
       #else: #efficient version
       #    sk = np.sort(k)
       #    nk = length_taus
       #    ik1 = 0
       #    for i in range(0, length_taus):
       #        ik0 = ik1
       #        print(sk[ik1+1])
       #        while ik1<nk and sk[ik1+1] < taus[i]:
       #            ik1 = ik1 + 1
       #        dd[i] = ik1 - ik0

        valid_k = k * (mask_invalid_pixels==0)
        kde = stats.gaussian_kde(valid_k.flatten(), args['bandwidth'])
        dd = kde(taus)        
        #chaine = 'np.array(['
        #for valeur in dd:
        #    chaine = chaine + str(valeur) + ', '
        #print(chaine[:-2] + "])")
                
        #Automatic threshold
        imax = np.int(np.argmax(dd))
        mod = taus[imax]
        #threshold = args['factor'] * np.median(dd[imax:])
        #tau = np.max(other_taus[dd > threshold])

        med = np.median(k.flatten())
        scale_iqr = mod - np.percentile(k.flatten(),10)
        tau = mod + args['factor']*scale_iqr
        print("frame {:03d}, mod = {:3.2f}, scale_iqr = {:3.2f}, tau = {:4.2f}".format(f, mod, scale_iqr, tau))

        mtau = 1.*(k < tau) #warping errors flagged with value 1
        mtau[mask_invalid_pixels>0] = 2 #out-of-frame pixels with value 2
        output=np.zeros((mtau.shape[0]+2, mtau.shape[1]+2,1))
        output[1:-1, 1:-1] = mtau.astype(np.uint8)
        iio.write(args['output']%f, output)
        
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute mask based on warping error")
    parser.add_argument("--dummy", type=str, default="/mnt/adisk/dewil/dummy", help="dummy path to cmpute overwritten warping errors")
    parser.add_argument("--warping", type=str, default="", help="path to warping errors (previously computed using the batch script")
    parser.add_argument("--output", type=str, default="", help="path to output mask")
    parser.add_argument("--input", type=str, default="", help="path to noisy input")
    parser.add_argument("--flow", type=str, default="", help="path to optical flow")
    parser.add_argument("--first", type=int, default=1, help="index first frame")
    parser.add_argument("--last", type=int, default=15, help="index last frame")
    parser.add_argument("--factor", type=int, default=5, help="factor for tau")
    parser.add_argument("--bandwidth", type=float, default=0.5, help="bandwidth used for the kernel density estimation")
   
    argspar = parser.parse_args()
    print("Be CAREFUL: DON'T LAUNCH SIMULTANEOUSLY SEVERAL COMPUTATIONS.")
    print("LAUCH ONE AFTER AN OTHER")
    compute_mask(**vars(argspar))
    
            
        
    
