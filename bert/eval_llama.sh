#!/bin/bash

man_values=(2)
exp_values=(6)

sum_types=(
    # '"quant"'
    # '"kahan"'
    # '"2sum"'
    # '"fast2sum"'
    # '"neumaier"'
    '"klein"'
)


# Loop through all combinations
for v in "${exp_values[@]}"; do
    for c in "${man_values[@]}"; do
        for sum_type in "${sum_types[@]}"; do
            echo "Running with q_man_w=$c, v_man_w=$v, sum=$sum_type"
            k_config=$(printf '{"quant":"MXFPQuantizer","man_w":%d,"exp_w":4}' "$c")
            s_config=$(printf '{"quant":"MXFPQuantizer","man_w":%d,"exp_w":4}' "$c")
            v_config=$(printf '{"quant":"MXFPQuantizer","man_w":%d,"exp_w":4}' "$v")
            CUDA_VISIBLE_DEVICES=1 python llama_ppl.py \
                        --model_id "meta-llama/Llama-3.2-1B" \
                        --config "k_quantizer=$k_config" \
                        --config "s_quantizer=$s_config" \
                        --config "v_quantizer=$v_config" \
                        --config "use_kulisch=false" \
                        --config "sum_type=$sum_type"
            echo "----------------------------------------------------------"
        done
    done
done

echo "All runs completed!"
