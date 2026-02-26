# Using Alveo V80 part directly
set part        xcv80-lsva4737-2MHP-e-S
set top         mxint_softmax
set outputDir   ./src/attention/synth_output_softmax
file mkdir $outputDir

# Set parameters based on command line arguments or defaults
set S_q           [expr {[llength $argv] > 0 ? [lindex $argv 0] : 4}]
set S_kv          [expr {[llength $argv] > 1 ? [lindex $argv 1] : 4}]
set d_kq          [expr {[llength $argv] > 2 ? [lindex $argv 2] : 8}]
set d_v           [expr {[llength $argv] > 3 ? [lindex $argv 3] : 8}]
set k1             [expr {[llength $argv] > 4 ? [lindex $argv 4] : 2}]
set k2             [expr {[llength $argv] > 5 ? [lindex $argv 5] : 2}]
set k3             [expr {[llength $argv] > 6 ? [lindex $argv 6] : 2}]
set scale_width   [expr {[llength $argv] > 7 ? [lindex $argv 7] : 8}]

# Mixed Precision Config
set m1_exp        [expr {[llength $argv] > 8 ? [lindex $argv 8] : 0}] 
set m1_man        [expr {[llength $argv] > 9 ? [lindex $argv 9] : 8}]
set m2_exp        [expr {[llength $argv] > 10 ? [lindex $argv 10] : 0}]
set m2_man        [expr {[llength $argv] > 11 ? [lindex $argv 11] : 8}]
set m3_exp        [expr {[llength $argv] > 12 ? [lindex $argv 12] : 0}]
set m3_man        [expr {[llength $argv] > 13 ? [lindex $argv 13] : 8}]

# Accumulation method parameters
set accum_method1 [expr {[llength $argv] > 14 ? [lindex $argv 14] : "KULISCH"}]
set accum_method2 [expr {[llength $argv] > 15 ? [lindex $argv 15] : "KULISCH"}]
set accum_method3 [expr {[llength $argv] > 16 ? [lindex $argv 16] : "KULISCH"}]
# DSP Control Params
set m1_dsp        [expr {[llength $argv] > 17 ? [lindex $argv 17] : "yes"}]
set m2_dsp        [expr {[llength $argv] > 18 ? [lindex $argv 18] : "yes"}]
set m3_dsp        [expr {[llength $argv] > 19 ? [lindex $argv 19] : "yes"}]
# Design Name suffix
set prefix_name   [expr {[llength $argv] > 20 ? [lindex $argv 20] : "softmax"}]

# Derived Params for Softmax
# M1 outputs (QKt) are input to Softmax.
# M2 outputs (Softmax result) are output of Softmax.
# Softmax uses M1 format for input and M2 format for output.
# We also need to calculate BWs.

set BW_1 [expr {1 + $m1_exp + $m1_man}]
set BW_2 [expr {1 + $m2_exp + $m2_man}]
set BW_3 [expr {1 + $m3_exp + $m3_man}]

set generics "DATA_IN_0_PRECISION_0=$BW_2 DATA_IN_0_PRECISION_1=$scale_width DATA_IN_0_DIM=$BW_3 DATA_IN_0_PARALLELISM=$k1 DATA_OUT_0_PRECISION_0=$BW_3 DATA_OUT_0_PRECISION_1=$scale_width DATA_OUT_0_DIM=$BW_3 DATA_OUT_0_PARALLELISM=$k2 USE_DSP=\"$m2_dsp\" ACCUM_METHOD=$accum_method2"

# Set the number of threads for Vivado
set_param general.maxThreads 12

# Generate timestamp
set timestamp [clock format [clock seconds] -format "%Y%m%d_%H%M"]

# Build common prefix (Use prefix_name instead of top)
set prefix "${outputDir}/${prefix_name}_S_q_${S_q}_S_kv_${S_kv}_d_kq_${d_kq}_d_v_${d_v}_k1_${k1}_k2_${k2}_k3_${k3}_scale_width_${scale_width}_M1_E_${m1_exp}_M1_M_${m1_man}_M2_E_${m2_exp}_M2_M_${m2_man}_M3_E_${m3_exp}_M3_M_${m3_man}_ACCUM_METHOD_${accum_method1}_${accum_method2}_${accum_method3}_DSP_${m1_dsp}_${m2_dsp}_${m3_dsp}_time_${timestamp}"


# Read sources - Include everything for dependency resolution
read_verilog    [glob ./src/attention/attention_fp.sv]
read_verilog    [glob ./src/attention/matmul_fp.sv]
read_verilog    [glob ./src/attention/mxoperators/mxint_softmax.sv]
read_verilog    [glob ./src/dot/dot_general_fp.sv]
read_verilog    [glob ./src/dot/dot_fp.sv]
read_verilog    [glob ./src/util/arith/mul_fp.sv]
read_verilog    [glob ./src/util/arith/vec_mul_fp.sv]
read_verilog    [glob ./src/util/arith/vec_sum_int.sv]
read_verilog    [glob ./src/util/arith/add_nrm.sv]
read_verilog    [glob ./src/util/accum/kahan/*.sv]
read_verilog    [glob ./src/util/accum/twosum/*.sv]
read_verilog    [glob ./src/util/accum/fasttwosum/*.sv]
read_verilog    [glob ./src/util/accum/neumaier/*.sv]
read_verilog    [glob ./src/util/accum/klein/*.sv]
read_verilog    [glob ./src/attention/mxoperators/*.sv]
read_verilog    [glob ./src/mase/src/mase_components/linear_layers/mxint_operators/rtl/*.sv]
read_verilog    [glob ./src/mase/src/mase_components/memory/rtl/*.sv]
read_verilog    [glob ./src/mase/src/mase_components/common/rtl/*.sv]
read_verilog    [glob ./src/mase/src/mase_components/cast/rtl/*.sv]
read_verilog    [glob ./src/mase/src/mase_components/scalar_operators/fixed/rtl/*.sv]
read_xdc        [ glob ./src/*.xdc ]

# Synthesis
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
# report_timing_summary   -datasheet -file ${prefix}_timing.rpt
# set t5 [clock milliseconds]
# puts "Time for report_timing_summary: [expr {($t5 - $t4) / 1000.0}] seconds"
# report_power            -file ${prefix}_power.rpt
# set t6 [clock milliseconds]
# puts "Time for report_power: [expr {($t6 - $t5) / 1000.0}] seconds"
