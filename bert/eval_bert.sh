#!/bin/bash

exp_values=(2 3 4)

man_values=(1 2 3)

# Loop through all combinations
for v in "${exp_values[@]}"; do
    for c in "${man_values[@]}"; do
        echo "Running with man_w=$c, exp_w=$v"
        q_config=$(printf '{"quant":"MXFPQuantizer","man_w":%d,"exp_w":%d}' "$c" "$v")
        CUDA_VISIBLE_DEVICES=1 python bert_sst2.py \
                    --config "k_quantizer=$q_config" \
                    --config "q_quantizer=$q_config" \
                    --config "s_quantizer=$q_config" \
                    --config "v_quantizer=$q_config" \
                    --config "p_quantizer=$q_config" \
                    --config "o_quantizer=$q_config"
        echo "Completed man_w=$c, exp_w=$v"
        echo "---"
    done
done

echo "All runs completed!"
