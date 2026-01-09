#!/bin/bash

exp_values=(3)

man_values=(4)

# Loop through all combinations
for v in "${exp_values[@]}"; do
    for c in "${man_values[@]}"; do
        echo "Running with man_w=$c, exp_w=$v"
        q_config=$(printf '{"quant":"MXFPQuantizer","man_w":%d,"exp_w":%d}' "$c" "$v")
        CUDA_VISIBLE_DEVICES=1 python llama_ppl.py \
                    --model_id "meta-llama/Llama-3.2-1B" \
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
