#!/bin/bash

exp_values=(2 3 4)

man_values=(1 2 3)

# Loop through all combinations
for v in "${exp_values[@]}"; do
    for c in "${man_values[@]}"; do
        echo "Running with man_w=$c, exp_w=$v"
        CUDA_VISIBLE_DEVICES=1 python bert_sst2.py --man_w $c --exp_w $v
        echo "Completed man_w=$c, exp_w=$v"
        echo "---"
    done
done

echo "All runs completed!"
