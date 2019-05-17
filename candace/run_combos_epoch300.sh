#!/bin/bash

markers=(
    CD45
    Ki67
    CD20
    Vimentin
)

len=${#markers[@]}

for (( i=0; i<len; i++ )); do
    for (( j=$i+1; j<len; j++ )); do
        python /home/ubuntu/candace/scripts/Care_Model_candace.py epoch300_${markers[$i]}_${markers[$j]} 300 ${markers[$i]}.tif ${markers[$j]}.tif 2>&1 | tee /home/ubuntu/candace/logs/epoch300_${markers[$i]}_${markers[$j]}.out
    done
done

python /home/ubuntu/candace/scripts/Care_Model_candace.py epoch300_ki67_cd45_cd20 300 Ki67.tif CD45.tif CD20.tif 2>&1 | tee /home/ubuntu/candace/logs/epoch300_ki67_cd45_cd20.out

python /home/ubuntu/candace/scripts/Care_Model_candace.py epoch300_ki67_cd45_vimentin 300 Ki67.tif CD45.tif Vimentin.tif 2>&1 | tee /home/ubuntu/candace/logs/epoch300_ki67_cd45_vimentin.out

python /home/ubuntu/candace/scripts/Care_Model_candace.py epoch300_ki67_cd20_vimentin 300 Ki67.tif CD20.tif Vimentin.tif 2>&1 | tee /home/ubuntu/candace/logs/epoch300_ki67_cd20_vimentin.out

python /home/ubuntu/candace/scripts/Care_Model_candace.py epoch300_cd45_cd20_vimentin 300 CD45.tif CD20.tif Vimentin.tif 2>&1 | tee /home/ubuntu/candace/logs/epoch300_cd45_cd20_vimentin.out

python /home/ubuntu/candace/scripts/Care_Model_candace.py epoch300_ki67_cd45_cd20_vimentin 300 Ki67.tif CD45.tif CD20.tif Vimentin.tif 2>&1 | tee /home/ubuntu/candace/logs/epoch300_ki67_cd45_cd20_vimentin.out

