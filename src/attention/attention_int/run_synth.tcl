set part        xcu250-figd2104-2L-e
set top         attention_int
set outputDir   ./src/attention/attention_int/synth_output
file mkdir $outputDir

# Set parameters based on command line arguments or defaults
# set EXP_WIDTH_I  [expr {[llength $argv] > 0 ? [lindex $argv 0] : 5}]
# set MANT_WIDTH_I [expr {[llength $argv] > 1 ? [lindex $argv 1] : 2}]
# set ELEMS_COUNT  [expr {[llength $argv] > 2 ? [lindex $argv 2] : 32}]
# set generics "EXP_WIDTH_I=$EXP_WIDTH_I MANT_WIDTH_I=$MANT_WIDTH_I ELEMS_COUNT=$ELEMS_COUNT"

# Set the number of threads for Vivado
set_param general.maxThreads 12

# Generate timestamp
set timestamp [clock format [clock seconds] -format "%Y%m%d_%H%M"]

# Build common prefix
# set prefix "${outputDir}/${top}_E${EXP_WIDTH_I}_M${MANT_WIDTH_I}_N${ELEMS_COUNT}_${timestamp}"
set prefix "${outputDir}/${top}_${timestamp}"

# Read sources
read_verilog    [glob ./src/attention/attention_int/*.sv]
read_xdc        [ glob ./src/*.xdc ]

# Synthesis
set t1 [clock milliseconds]
# synth_design -top $top -part $part -flatten rebuilt -retiming -generic $generics
synth_design -top $top -part $part -flatten rebuilt -retiming
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