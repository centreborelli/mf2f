#! /bin/bash

noisy0=$1   #frame t
noisy1=$2   #frame t+1
flow=$3   #OF frame t to frame t+1 (warping t+1 to t)
output=$4

# warp frame N+1 to N using optical flow
WFRAME1=$(printf $output/warp.tiff)
WARP3="/home/dewil/occlu-mask/build/bin/warp-bicubic"
$WARP3 -f $flow -i $noisy1 -o $WFRAME1

WOUT=$(printf $output/mask_invalid_pixels.png)
plambda $WFRAME1 "x[0] isnan 255 *" -o $WOUT

# downsample by a factor of 2
DFRAME0=$(printf $output/downs.tiff)
DWFRAME1=$(printf $output/downs_warp.tiff)
downsa v 2 $noisy0 $DFRAME0
downsa v 2 $WFRAME1 $DWFRAME1

DWOUT=$(printf $output/dwo.png)
downsa v 2 $WOUT | plambda - "x 0.1 > 255 *" -o $DWOUT

# blur downscaled frames
blur -s g 2 $DFRAME0  $DFRAME0 
blur -s g 2 $DWFRAME1 $DWFRAME1

blur -s g 2 $DWOUT | plambda - "x 0.1 > 255 *" -o $DWOUT

# warping error
WERR=$(printf $output/WERR.tiff)
plambda $DWFRAME1 $DFRAME0 \
		  "x[0] y[0] - fabs x[1] y[1] - fabs x[2] y[2] - fabs + + 3 /" \
		  -o $WERR

# blur warping error
blur -s g 2 $WERR $WERR

blur -s g 2 $DWOUT | plambda - "x 0.1 > 255 *" -o $DWOUT

# upsample back
UWERR=$(printf $output/warping_error.tiff)
upsa 2 2 $WERR $UWERR
upsa 2 2 $DWOUT | plambda - "x 0.1 > 255 *" -o $WOUT



