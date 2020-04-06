#!/bin/bash

input=$1
ref=$2
first=$3
last=$4
data=$5

format=03

noise_level=25


##make data folder and sub-folders
if [ ! -d $data ];then
    mkdir $data
fi
if [ ! -d $data/flow ];then
    mkdir $data/flow
fi
if [ ! -d $data/mask_collition ];then
    mkdir $data/mask_collition
fi
if [ ! -d $data/mask_warping_res ];then
    mkdir $data/mask_warping_res
fi
if [ ! -d $data/results_8sigmas ];then
    mkdir $data/results_8sigmas
fi
if [ ! -d $data/results_online_no_teacher ];then
    mkdir $data/results_online_no_teacher
fi
if [ ! -d $data/results_online_with_teacher ];then
    mkdir $data/results_online_with_teacher
fi
if [ ! -d $data/results_offline_no_teacher ];then
    mkdir $data/results_offline_no_teacher
fi
if [ ! -d $data/results_offline_with_teacher ];then
    mkdir $data/results_offline_with_teacher
fi



##useful paths
flow=$data/flow/%${format}d.flo
mask_collition=$data/mask_collition/%${format}d.png
mask_warping_res=$data/mask_warping_res/%${format}d.png
eight_sigmas=$data/results_8sigmas/%${format}d.tiff
results_online_no_teach=$data/results_online_no_teacher/%${format}d.png
results_online_with_teach=$data/results_online_with_teacher/%${format}d.png
results_offline_no_teach=$data/results_offline_no_teacher/%${format}d.png
results_offline_with_teach=$data/results_offline_with_teacher/%${format}d.png




##compute flow
cd tvl1flow
bash tvl1flow.sh $input $first $(($last-1)) $flow
cd ..
echo flows computed

##compute collition mask
cd collition_mask
bash compute_mask.sh $flow $first $(($last-1)) $mask_collition
cd ..
echo collition masks computed

##compute warping_error_mask
cd warping_res_mask
python3.6 compute_threshold.py --dummy $data --output $mask_warping_res --input $input --flow $flow --first $first --last $last
rm $data/{downs.tiff,downs_warp.tiff,dwo.png,mask_invalid_pixels.png,warping_error.tiff,warp.tiff,WERR.tiff}
cd ..
echo warping res masks computed



## compute the results with FastDVDnet-8sigmas
python3.6 video_f2f_8sigmas.py --input $input --ref $ref --flow $flow --mask_collition $mask_collition --mask_warping_res $mask_warping_res --output $eight_sigmas --first $first --last $last --noise_level $noise_level 
echo FastDVDnet-8sigmas computed

##compute the results with the method online no teacher
python3.6 video_f2f_online_no_teacher.py --input $input --ref $ref --flow $flow --mask_collition $mask_collition --mask_warping_res $mask_warping_res --output $results_online_no_teach --first $first --last $last --noise_level $noise_level
echo online no teacher computed


##compute the results with the method online with teacher
python3.6 video_f2f_online_with_teacher.py --input $input --ref $ref --flow $flow --mask_collition $mask_collition --mask_warping_res $mask_warping_res --output $results_online_with_teach --first $first --last $last --teacher_outputs $eight_sigmas --noise_level $noise_level
echo online with teacher computed


##compute the results with the method offline no teacher
python3.6 video_f2f_offline_no_teacher.py --input $input --ref $ref --flow $flow --mask_collition $mask_collition --mask_warping_res $mask_warping_res --output $results_offline_no_teach --first $first --last $last --noise_level $noise_level
echo offline no teacher computed


##compute the results with the method offline with teacher
python3.6 video_f2f_offline_with_teacher.py --input $input --ref $ref --flow $flow --mask_collition $mask_collition --mask_warping_res $mask_warping_res --output $results_offline_with_teach --first $first --last $last --teacher_outputs $eight_sigmas --noise_level $noise_level
