#!/bin/bash

for d in $(find /home/ubuntu/candace/models/* -type d -printf "%f\n"); do
    python /home/ubuntu/candace/scripts/plot_multiple_1samp.py $d
done

