#!/bin/bash

markers=(
    H3K9Ac
    CD45
    HLA-Class-1
    Ki67
    CD20
    Vimentin
)

len=${#markers[@]}

for (( i=0; i<len; i++ )); do
    for (( j=$i+1; j<len; j++ )); do
        python /home/ubuntu/candace/scripts/Care_Model_candace.py epoch50_${markers[$i]}_${markers[$j]} 50 ${markers[$i]}.tif ${markers[$j]}.tif 2>&1 | tee /home/ubuntu/candace/logs/epoch50_${markers[$i]}_${markers[$j]}.out
    done
done

