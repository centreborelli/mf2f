#!/bin/bash/

I=${1:-""}
F=${2:-1}
L=${3-1}
O=${4:-""}

for j in `seq $F $L`;
do
    #echo $j;
    echo build/bin/occlu-mask -i `printf $I $((j))`  -t 1.6 -o `printf $O $((j))`;
done | parallel --bar 
