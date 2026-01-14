# show_v80_details.tcl
# Run this script in Vivado 2024.2 to see V80 details

puts "================================================================"
puts "                Xilinx Alveo V80 / Versal Premium Details       "
puts "================================================================"

# 1. Search for V80 Parts
# puts "\n--- V80 Specific Parts (xcv80*) ---"
# set parts [get_parts -quiet -filter {NAME =~ xcv80*}]
# if {[llength $parts] > 0} {
#     foreach p $parts {
#         puts "Part: $p"
#         puts "  Family:      [get_property FAMILY $p]"
#         puts "  Package:     [get_property PACKAGE $p]"
#         puts "  Speed Grade: [get_property SPEED $p]"
#     }
# } else {
#     puts "No specific 'xcv80' parts found."
# }

# 2. Search for Versal Premium Parts
# puts "\n--- Versal Premium Parts (xcvp1502/1802) ---"
# set vp_parts [get_parts -quiet -filter {NAME =~ xcvp1502* || NAME =~ xcvp1802*}]
# if {[llength $vp_parts] > 0} {
#     # limit output if too many
#     set count 0
#     foreach p $vp_parts {
#         if {$count < 5} {
#             puts "Part: $p"
#         }
#         incr count
#     }
#     if {$count > 5} { puts "... and [expr $count - 5] more." }
# } else {
#     puts "No Versal Premium parts found."
# }

# 3. Search for Boards
puts "\n--- V80 Boards ---"
set boards [get_boards -quiet]
set found_board 0
foreach b $boards {
    set disp_name [get_property DISPLAY_NAME $b]
    if {[string match -nocase "*V80*" $b] || [string match -nocase "*V80*" $disp_name]} {
        set found_board 1
        puts "Board: $disp_name"
        puts "  ID:        $b"
        puts "  Part Name: [get_property PART0_NAME $b]"
        puts "  Vendor:    [get_property VENDOR $b]"
        puts "  File:      [get_property FILE_NAME $b]"
    }
}

if {!$found_board} {
    puts "No boards matching '*V80*' found."
}
puts "\n================================================================"
