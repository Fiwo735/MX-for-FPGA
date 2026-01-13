# check_v80.tcl
puts "@@@VERSION:[version -short]"

set vp_found 0
set part_found 0
set board_found 0
set debug_vp ""
set debug_part ""

# Check parts
if { [catch { set parts [get_parts -quiet] } err] } {
    # silently ignore error
} else {
    foreach p $parts {
        if {[string match "xcvp*1802*" $p] || [string match "xcvp*1502*" $p]} {
            set vp_found 1
            if {$debug_vp == ""} { set debug_vp $p }
        }
        if {[string match "*V80*" $p] || [string match "*v80*" $p]} {
            set part_found 1
            if {$debug_part == ""} { set debug_part $p }
        }
    }
}

# Check boards
if { [catch { set boards [get_boards -quiet] } err] } {
    # ignore
} else {
    foreach b $boards {
        if {[string match "*V80*" $b] || [string match "*v80*" $b]} {
            set board_found 1
        }
    }
}

# Construct strings dynamically to avoid grepping source code
set tag "###RESULT"
if {$board_found} { puts "$tag:V80_BOARD_FOUND" } else { puts "$tag:NO_BOARD" }
if {$part_found} { puts "$tag:V80_PART_FOUND:[string range $debug_part 0 20]" } else { puts "$tag:NO_V80_PART" }
if {$vp_found} { puts "$tag:VERSAL_PREMIUM_FOUND:[string range $debug_vp 0 20]" } else { puts "$tag:NO_VERSAL_PREMIUM" }
