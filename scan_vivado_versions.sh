#!/bin/bash
SEARCH_DIR="/mnt/applications/Xilinx"
TCL_SCRIPT="/home/os717/MX-for-FPGA/check_v80.tcl"

echo "Scanning for Vivado installations in $SEARCH_DIR..."
echo "---------------------------------------------------"
printf "%-50s | %-10s | %-10s | %-10s | %-10s\n" "Vivado Path" "Version" "Board" "Part (Match)" "Versal Prem."
echo "------------------------------------------------------------------------------------------------------------------"
process_vivado() {
    local vivado_bin=$1
    output=$("$vivado_bin" -mode batch -source "$TCL_SCRIPT" -nolog -nojournal 2>/dev/null)
    
    version=$(echo "$output" | grep "@@@VERSION" | cut -d':' -f2 | tr -d '\r')
    if [[ -z "$version" ]]; then version="Unknown"; fi
    
    board="NO"
    part="NO"
    vp="NO"
    match_info=""
    
    if echo "$output" | grep -q "###RESULT:V80_BOARD_FOUND"; then board="YES"; fi
    if echo "$output" | grep -q "###RESULT:V80_PART_FOUND"; then 
        part="YES"
        match_info=$(echo "$output" | grep "###RESULT:V80_PART_FOUND" | cut -d':' -f3 | tr -d '\r')
    fi
    if echo "$output" | grep -q "###RESULT:VERSAL_PREMIUM_FOUND"; then vp="YES"; fi
    
    if [[ "$board" == "YES" && "$vp" == "YES" ]]; then
       printf "\033[0;32m%-50s | %-10s | %-10s | %-15s | %-10s\033[0m\n" "$vivado_bin" "$version" "$board" "$match_info" "$vp"
    else
       printf "%-50s | %-10s | %-10s | %-15s | %-10s\n" "$vivado_bin" "$version" "$board" "$match_info" "$vp"
    fi
}

# explicit search for common locations to avoid slow 'find'

# Pattern 1: /mnt/applications/Xilinx/Vivado/<Version>/bin/vivado
for v_dir in "$SEARCH_DIR"/Vivado/*; do
    vivado_bin="$v_dir/bin/vivado"
    if [[ -x "$vivado_bin" ]]; then
       process_vivado "$vivado_bin"
    fi
done

# Pattern 2: /mnt/applications/Xilinx/<Version>/Vivado/bin/vivado
for v_dir in "$SEARCH_DIR"/*; do
    # Check direct bin
    vivado_bin="$v_dir/Vivado/bin/vivado"
    if [[ -x "$vivado_bin" ]]; then
       process_vivado "$vivado_bin"
    fi
    
    # Check nested version folder (e.g. 24.2/Vivado/2024.2/bin/vivado)
    # We glob for the version folder inside Vivado
    for inner_ver in "$v_dir/Vivado"/*; do
        vivado_bin="$inner_ver/bin/vivado"
        if [[ -x "$vivado_bin" ]]; then
           process_vivado "$vivado_bin"
        fi
    done
done
