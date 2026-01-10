set part        xcu250-figd2104-2L-e
set top         attention_fp
set outputDir   ./src/attention/synth_output
file mkdir $outputDir

# Set parameters based on command line arguments or defaults
set S_q           [expr {[llength $argv] > 0 ? [lindex $argv 0] : 4}]
set S_kv          [expr {[llength $argv] > 1 ? [lindex $argv 1] : 4}]
set d_kq          [expr {[llength $argv] > 2 ? [lindex $argv 2] : 8}]
set d_v           [expr {[llength $argv] > 3 ? [lindex $argv 3] : 8}]
set k             [expr {[llength $argv] > 4 ? [lindex $argv 4] : 2}]
set scale_width   [expr {[llength $argv] > 5 ? [lindex $argv 5] : 8}]

# Mixed Precision Config (New)
# Default to 0 (Integer Mode) if not provided
set m1_exp        [expr {[llength $argv] > 6 ? [lindex $argv 6] : 0}] 
set m1_man        [expr {[llength $argv] > 7 ? [lindex $argv 7] : 8}]
set m2_exp        [expr {[llength $argv] > 8 ? [lindex $argv 8] : 0}]
set m2_man        [expr {[llength $argv] > 9 ? [lindex $argv 9] : 8}]
set m3_exp        [expr {[llength $argv] > 10 ? [lindex $argv 10] : 0}]
set m3_man        [expr {[llength $argv] > 11 ? [lindex $argv 11] : 8}]
# Accumulation method parameters (Defaults to "Kulisch")
set accum_method1 [expr {[llength $argv] > 12 ? [lindex $argv 12] : "KULISCH"}]
set accum_method2 [expr {[llength $argv] > 13 ? [lindex $argv 13] : "KULISCH"}]
set accum_method3 [expr {[llength $argv] > 14 ? [lindex $argv 14] : "KULISCH"}]
# DSP Control Params (Defaults to "yes")
set m1_dsp        [expr {[llength $argv] > 15 ? [lindex $argv 15] : "yes"}]
set m2_dsp        [expr {[llength $argv] > 16 ? [lindex $argv 16] : "yes"}]
set sm_dsp        [expr {[llength $argv] > 17 ? [lindex $argv 17] : "yes"}]

set generics "S_q=$S_q S_kv=$S_kv d_kq=$d_kq d_v=$d_v k=$k scale_width=$scale_width M1_EXP_WIDTH=$m1_exp M1_MAN_WIDTH=$m1_man M2_EXP_WIDTH=$m2_exp M2_MAN_WIDTH=$m2_man M3_EXP_WIDTH=$m3_exp M3_MAN_WIDTH=$m3_man ACCUM_METHOD1=$accum_method1 ACCUM_METHOD2=$accum_method2 ACCUM_METHOD3=$accum_method3 M1_USE_DSP=\"$m1_dsp\" M2_USE_DSP=\"$m2_dsp\" SOFTMAX_USE_DSP=\"$sm_dsp\""

# Set the number of threads for Vivado
set_param general.maxThreads 12

# Generate timestamp
set timestamp [clock format [clock seconds] -format "%Y%m%d_%H%M"]

# Build common prefix
set prefix "${outputDir}/${top}_S_q_${S_q}_S_kv_${S_kv}_d_kq_${d_kq}_d_v_${d_v}_k_${k}_scale_width_${scale_width}_M1_E_${m1_exp}_M1_M_${m1_man}_M2_E_${m2_exp}_M2_M_${m2_man}_M3_E_${m3_exp}_M3_M_${m3_man}_ACCUM_METHOD_${accum_method1}_${accum_method2}_${accum_method3}_DSP_${m1_dsp}_${m2_dsp}_${sm_dsp}_time_${timestamp}"

# Read sources
read_verilog    [glob ./src/attention/attention_fp.sv]
read_verilog    [glob ./src/attention/matmul_fp.sv]
read_verilog    [glob ./src/dot/dot_general_fp.sv]
read_verilog    [glob ./src/dot/dot_fp.sv]
read_verilog    [glob ./src/util/arith/mul_fp.sv]
read_verilog    [glob ./src/util/arith/vec_mul_fp.sv]
read_verilog    [glob ./src/util/arith/vec_sum_int.sv]
read_verilog    [glob ./src/util/arith/add_nrm.sv]
read_verilog    [glob ./src/attention/mxoperators/*.sv]
read_verilog    [glob ./src/attention/mxoperators/lib/*.sv]
read_xdc        [ glob ./src/*.xdc ]

# Synthesis
# Suppress "Port unconnected" warnings (benign for parameterized IP and combinatorial wrappers)
set_msg_config -id {Synth 8-7129} -suppress
set t1 [clock milliseconds]
synth_design -top $top -part $part -flatten rebuilt -retiming -generic $generics -include_dirs {./src/attention}
set t2 [clock milliseconds]
puts "Time for synth_design: [expr {($t2 - $t1) / 1000.0}] seconds"

# Checkpoint after synthesis
write_checkpoint -force ${prefix}_post_synth.dcp
set t3 [clock milliseconds]
puts "Time for write_checkpoint: [expr {($t3 - $t2) / 1000.0}] seconds"

# Reports
report_utilization      -file ${prefix}_util.rpt
set t4 [clock milliseconds]
puts "Time for report_utilization: [expr {($t4 - $t3) / 1000.0}] seconds"
report_timing_summary   -datasheet -file ${prefix}_timing.rpt
set t5 [clock milliseconds]
puts "Time for report_timing_summary: [expr {($t5 - $t4) / 1000.0}] seconds"
report_power            -file ${prefix}_power.rpt
set t6 [clock milliseconds]
puts "Time for report_power: [expr {($t6 - $t5) / 1000.0}] seconds"

# opt_design
# place_design
# phys_opt_design
# write_checkpoint -force $outputDir/post_place
# report_timing_summary -file $outputDir/post_place_timing_summary.rpt

# route_design
# write_checkpoint -force $outputDir/post_route
# report_timing_summary -file $outputDir/post_route_timing_summary.rpt
# report_timing -sort_by group -max_paths 100 -path_type summary -file $outputDir/post_route_timing.rpt
# report_clock_utilization -file $outputDir/clock_util.rpt
# report_utilization -file $outputDir/post_route_util.rpt
# report_power -file $outputDir/post_route_power.rpt
# report_drc -file $outputDir/post_imp_drc.rpt
# write_verilog -force $outputDir/top_impl_netlist.v
# write_xdc -no_fixed_only -force $outputDir/top_impl.xdc

# write_bitstream -force $outputDir/$top.bit