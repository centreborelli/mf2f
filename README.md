Multi Frame2Frame
=========================================================================
DEPENDANCIES
------------
TO DO 
- python modules
- iio github
- Enric's imscript github

USAGE OF THE END TO END CODE
----------------------------

Run the end-to-end script using the command:</br>
```bash run.sh "input_sequence_path" "ref_sequence_path" first last "output_folder_path"```

The five arguments are mandatory and represent:
* `input_sequence_path` : path to the noisy input, using the C standard (e.g. "noisy_video/%03d.tiff")
* `ref_sequence_path` : path to the clean frames, using the C standard (e.g. "clean_video/%03d.png"). Put the same as for the noisy input if you do not have them
* `first`number of the first frame
* `last` number of the last frame 
* `output_folder_path` path where the code can create a folder containing the results of the methods (e.g. "results_MF2F")

The codes read and store color images in png, jpeg and tiff using iio

OUTPUT FOLDER
-------------

The MF2F codes need mask and optical flow results which are computed offline previously to the MF2F codes.
The output folder will be created at the location you put in the `output_folder_path`
It will contain eight subfolders:
* `flow` : This subfolder will contain the forward flows between two contiguous frames
* `mask_collision` : This subfolder will contain the collision mask for every frame
* `mask_warping_res` : This subfolder will contain an occlusion mask based on the warping residual (as well as those residuals) computed for each frame
* `results_8sigmas` : This subfolder will contain the results of the teacher network (i.e. the results of FastDVDnet fine-tuned in an online way and only its variance map parameter. We use a spatial variance map based on the value of the illumination)
* `results_online_no_teacher` : This subfolder will contain the results of the online MF2F method, without the use of the teacher network
* `results_online_with_teacher` : This subfolder will contain the results of the online MF2F method, using the teacher network
* `results_offline_no_teacher` : This subfolder will contain the results of the offline MF2F method, without the use of the teacher network
* `results_online_with_teacher` : This subfolder will contain the results of the offline MF2F method, using the teacher network


RESULTS FOLDER CONTAIN
----------------------

Each result folder (results_8sigmas, results_online_no_teacher, results_online_with_teacher, results_offline_no_teacher, results_offline_with_teacher) contains the result frames obtained by evaluating the fine-tuned network on the training stack (file `training_%03d.png`) and on the natural stack (file `%03d.png`).
In the online methods, these are the results of the so far fine-tuned network. In the offline methods, theses are the results of the final fine-tuned network. 
In addition, the PSNR and SSIM values are stored in file respectively called PSNR.txt and SSIM.txt for the evaluation with the natural stack and PSNR_training.txt, SSIM_training.txt for the evaluation with the training stack. The offline results subfolders also contain files PSNR_tot.txt and SSIM_tot.txt which are the PSNR and SSIM value evaluated on the natural stack for every loss computations for the whole offline fine-tuning.
