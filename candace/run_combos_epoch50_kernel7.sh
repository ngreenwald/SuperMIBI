#!/bin/bash
# Before running, make sure Care_Model_candace.py is changed so that kernel is of size 7

markers=(
    CD45
    Ki67
    CD20
    Vimentin
)

len=${#markers[@]}

for (( i=0; i<len; i++ )); do
    for (( j=$i+1; j<len; j++ )); do
        python /home/ubuntu/candace/scripts/Care_Model_candace.py kernel7_epoch50_${markers[$i]}_${markers[$j]} 50 ${markers[$i]}.tif ${markers[$j]}.tif 2>&1 | tee /home/ubuntu/candace/logs/kernel7_epoch50_${markers[$i]}_${markers[$j]}.out
    done
done

python /home/ubuntu/candace/scripts/Care_Model_candace.py kernel7_epoch50_ki67_cd45_cd20 50 Ki67.tif CD45.tif CD20.tif 2>&1 | tee /home/ubuntu/candace/logs/kernel7_epoch50_ki67_cd45_cd20.out

python /home/ubuntu/candace/scripts/Care_Model_candace.py kernel7_epoch50_ki67_cd45_vimentin 50 Ki67.tif CD45.tif Vimentin.tif 2>&1 | tee /home/ubuntu/candace/logs/kernel7_epoch50_ki67_cd45_vimentin.out

python /home/ubuntu/candace/scripts/Care_Model_candace.py kernel7_epoch50_ki67_cd20_vimentin 50 Ki67.tif CD20.tif Vimentin.tif 2>&1 | tee /home/ubuntu/candace/logs/kernel7_epoch50_ki67_cd20_vimentin.out

python /home/ubuntu/candace/scripts/Care_Model_candace.py kernel7_epoch50_cd45_cd20_vimentin 50 CD45.tif CD20.tif Vimentin.tif 2>&1 | tee /home/ubuntu/candace/logs/kernel7_epoch50_cd45_cd20_vimentin.out

python /home/ubuntu/candace/scripts/Care_Model_candace.py kernel7_epoch50_ki67_cd45_cd20_vimentin 50 Ki67.tif CD45.tif CD20.tif Vimentin.tif 2>&1 | tee /home/ubuntu/candace/logs/kernel7_epoch50_ki67_cd45_cd20_vimentin.out

