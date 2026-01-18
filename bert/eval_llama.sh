#!/bin/bash

v_values=(2)
c_values=(32)

sum_types=(
    # '"KULISCH"'
    '"QUANT"'
    '"KAHAN"'
    # '"TWOSUM"'
    # '"FASTTWOSUM"'
    # '"NEUMAIER"'
    # '"KLEIN"'
)


# Loop through all combinations
for v in "${v_values[@]}"; do
    for sum_type in "${sum_types[@]}"; do
        for c in "${c_values[@]}"; do
            echo "Running with q_man_w=$c, v_man_w=$v, sum=$sum_type"
            k_config=$(printf '{"quant":"MXFPQuantizer","man_w":4,"exp_w":3,"group_size":%d}' "$c")
            s_config=$(printf '{"quant":"MXFPQuantizer","man_w":2,"exp_w":3,"group_size":%d}' "$c")
            v_config=$(printf '{"quant":"MXFPQuantizer","man_w":4,"exp_w":3,"group_size":%d}' "$c")
            # k_config=$(printf '{"quant":"MXINTQuantizer","bit_w":8,"group_size":%d}' "$c")
            # s_config=$(printf '{"quant":"MXINTQuantizer","bit_w":8,"group_size":%d}' "$c")
            # v_config=$(printf '{"quant":"MXINTQuantizer","bit_w":8,"group_size":%d}' "$c")
            CUDA_VISIBLE_DEVICES=1 python llama_ppl.py \
                        --model_id "meta-llama/Llama-3.2-1B" \
                        --config "k_quantizer=$k_config" \
                        --config "s_quantizer=$s_config" \
                        --config "v_quantizer=$v_config" \
                        --config "sum_type_attn_s=$sum_type" \
                        --config "sum_type_smax=$sum_type" \
                        --config "sum_type_attn_o=$sum_type"
            echo "----------------------------------------------------------"
        done
    done
done

echo "All runs completed!"
