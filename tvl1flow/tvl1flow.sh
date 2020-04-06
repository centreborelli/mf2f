#!/bin/bash
# Computes tvl1 optical flow for a (noisy) sequence. 

I=${1:-""}
F=${2:-1}
L=${3:-1}
O=${4:-""}

for i in `seq $F $L`;
do
    echo ./tvl1flow `printf $I $((i))` \
        `printf $I $((i+1))` \
        `printf $O $((i))` \
        1 0.25 0.15 0.3 100 0 0.5 5 0.01 0; 
done | parallel --bar
