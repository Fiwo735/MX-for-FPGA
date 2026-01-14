# get_v80_resources.tcl
puts "----------------------------------------------------------------"
puts "DEBUGGING ALVEO V80 AVAILABILITY"

# 1. Broad Part Search for 'V80'
puts "\n[1] Searching for parts with 'V80' in NAME or DESCRIPTION..."
set parts [get_parts -quiet -filter {NAME =~ *V80* || DESCRIPTION =~ *V80*}]
if {[llength $parts] > 0} {
    foreach part $parts {
        puts "  Found Part: $part ([get_property FAMILY $part])"
    }
} else {
    puts "  > No parts found with '*V80*' pattern."
}

# 2. Search for Versal Premium Parts (XCVP)
puts "\n[2] Searching for Versal Premium (xcvp*) parts (potential V80 matches)..."
set vp_parts [get_parts -quiet -filter {NAME =~ xcvp*1802* || NAME =~ xcvp*1502* || NAME =~ xcvp*1202*}]
# Note: V80 is often associated with the VP1802 or similar high-end Versal Premium. Use a broad search if unsure.
if {[llength $vp_parts] > 0} {
    puts "  Found [llength $vp_parts] Versal Premium candidate parts. Listing first 5:"
    set count 0
    foreach part $vp_parts {
        puts "  Candidate: $part ([get_property DESCRIPTION $part])"
        incr count
        if {$count >= 5} { break }
    }
} else {
    puts "  > No 'xcvp*' Versal Premium parts found. You might be missing the Versal Premium device support."
}

# 3. Board Search
puts "\n[3] Searching for Board 'V80'..."
set boards [get_boards -quiet -filter {NAME =~ *V80* || DISPLAY_NAME =~ *V80*}]
if {[llength $boards] > 0} {
    foreach board $boards {
        puts "  Found Board: [get_property DISPLAY_NAME $board]"
        puts "    ID: $board"
        
        # Dump properties to find the part
        set p_part [get_property PART_NAME $board]
        set p_part0 [get_property PART0_NAME $board]
        set p_file [get_property FILE_NAME $board]
        set p_vendor [get_property VENDOR $board]

        puts "    Property PART_NAME: '$p_part'"
        puts "    Property PART0_NAME: '$p_part0'"
        puts "    Property FILE_NAME: '$p_file'"
        puts "    Property VENDOR: '$p_vendor'"
    }
} else {
    puts "  > No V80 board found."
}
puts "----------------------------------------------------------------"
