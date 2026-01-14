import re
import os

files = [
    ("KULISCH", "src/attention/synth_output/attention_fp_S_q_8_S_kv_4_d_kq_4_d_v_4_k_2_scale_width_8_M1_E_4_M1_M_3_M2_E_4_M2_M_3_M3_E_4_M3_M_3_ACCUM_METHOD_NEUMAIER_KULISCH_KLEIN_DSP_yes_yes_yes_time_20260113_0150"),
    ("KLEIN", "src/attention/synth_output/attention_fp_S_q_8_S_kv_4_d_kq_4_d_v_4_k_2_scale_width_8_M1_E_4_M1_M_3_M2_E_4_M2_M_3_M3_E_4_M3_M_3_ACCUM_METHOD_NEUMAIER_KLEIN_KLEIN_DSP_yes_yes_yes_time_20260113_0202"),
    ("TWOSUM", "src/attention/synth_output/attention_fp_S_q_8_S_kv_4_d_kq_4_d_v_4_k_2_scale_width_8_M1_E_4_M1_M_3_M2_E_4_M2_M_3_M3_E_4_M3_M_3_ACCUM_METHOD_NEUMAIER_TWOSUM_KLEIN_DSP_yes_yes_yes_time_20260113_0207"),
    ("FASTTWOSUM", "src/attention/synth_output/attention_fp_S_q_8_S_kv_4_d_kq_4_d_v_4_k_2_scale_width_8_M1_E_4_M1_M_3_M2_E_4_M2_M_3_M3_E_4_M3_M_3_ACCUM_METHOD_NEUMAIER_FASTTWOSUM_KLEIN_DSP_yes_yes_yes_time_20260113_0209"),
    ("NEUMAIER", "src/attention/synth_output/attention_fp_S_q_8_S_kv_4_d_kq_4_d_v_4_k_2_scale_width_8_M1_E_4_M1_M_3_M2_E_4_M2_M_3_M3_E_4_M3_M_3_ACCUM_METHOD_NEUMAIER_NEUMAIER_KLEIN_DSP_yes_yes_yes_time_20260113_0212")
]

print("| Middle Accum (Softmax) | LUTs | DSPs | WNS (ns) |")
print("| :--- | :--- | :--- | :--- |")

for name, base_path in files:
    lut = "-"
    dsp = "-"
    wns = "-"
    
    util_file = base_path + "_util.rpt"
    timing_file = base_path + "_timing.rpt"
    
    if os.path.exists(util_file):
        with open(util_file, 'r') as f:
            content = f.read()
            lut_match = re.search(r'\|\s*CLB LUTs\*?\s*\|\s*(\d+)\s*\|', content)
            if lut_match:
                lut = lut_match.group(1)
            else:
                 lut_match = re.search(r'\|\s*Slice LUTs\*?\s*\|\s*(\d+)\s*\|', content)
                 if lut_match: lut = lut_match.group(1)
            
            dsp_match = re.search(r'\|\s*DSPs\s*\|\s*(\d+)\s*\|', content)
            if dsp_match:
                dsp = dsp_match.group(1)
                
    if os.path.exists(timing_file):
        with open(timing_file, 'r') as f:
            content = f.read()
            wns_match = re.search(r'WNS\(ns\).*?\n\s*(-?\d+\.\d+)', content, re.DOTALL)
            if wns_match:
                wns = wns_match.group(1)

    print(f"| {name} | {lut} | {dsp} | {wns} |")
