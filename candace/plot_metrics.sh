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
        python /home/ubuntu/candace/scripts/plot_model_metrics.py epoch50_${markers[$i]}_${markers[$j]}
    done
done

