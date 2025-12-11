set part        xcu250-figd2104-2L-e
set top         attention_int
set outputDir   ./src/attention/attention_int/synth_output
file mkdir $outputDir

# Set parameters based on command line arguments or defaults
set S_q         [expr {[llength $argv] > 0 ? [lindex $argv 0] : 4}]
set S_kv        [expr {[llength $argv] > 1 ? [lindex $argv 1] : 4}]
set d_kq        [expr {[llength $argv] > 2 ? [lindex $argv 2] : 8}]
set d_v         [expr {[llength $argv] > 3 ? [lindex $argv 3] : 8}]
set k           [expr {[llength $argv] > 4 ? [lindex $argv 4] : 2}]
set bit_width   [expr {[llength $argv] > 5 ? [lindex $argv 5] : 8}]
set out_width   [expr {[llength $argv] > 6 ? [lindex $argv 6] : 8}]
set scale_width [expr {[llength $argv] > 7 ? [lindex $argv 7] : 8}]
set generics "S_q=$S_q S_kv=$S_kv d_kq=$d_kq d_v=$d_v k=$k bit_width=$bit_width out_width=$out_width scale_width=$scale_width"

# Set the number of threads for Vivado
set_param general.maxThreads 12

# Generate timestamp
set timestamp [clock format [clock seconds] -format "%Y%m%d_%H%M"]

# Build common prefix
set prefix "${outputDir}/${top}_S_q_${S_q}_S_kv_${S_kv}_d_kq_${d_kq}_d_v_${d_v}_k_${k}_bit_width_${bit_width}_out_width_${out_width}_scale_width_${scale_width}_time_${timestamp}"

# Read sources
read_verilog    [glob ./src/attention/attention_int/*.sv]
read_xdc        [ glob ./src/*.xdc ]

# Synthesis
set t1 [clock milliseconds]
synth_design -top $top -part $part -flatten rebuilt -retiming -generic $generics
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