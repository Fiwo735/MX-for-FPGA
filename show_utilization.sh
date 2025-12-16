#!/bin/bash

# Find the latest utilization report in the attention output directory
REPORT=$(ls -t src/attention/attention_int/synth_output/*_util.rpt 2>/dev/null | head -n 1)

if [ -z "$REPORT" ]; then
    echo "Error: No utilization report found in src/attention/attention_int/synth_output/"
    exit 1
fi

echo "================================================================================"
echo "Report: $REPORT"
echo "================================================================================"

# Extract specific sections
# CLB Logic (LUTs, FFs)
echo "--- CLB LOGIC ---"
grep -A 15 "| CLB LUTs" "$REPORT" | head -n 12
echo ""

# Block RAM
echo "--- BRAM ---"
grep -A 5 "| Block RAM Tile" "$REPORT" | head -n 5
echo ""

# DSPs
echo "--- DSPs ---"
grep -A 5 "| DSPs" "$REPORT" | head -n 5
echo ""

# Primitives Summary
# echo "--- PRIMITIVES ---"
# grep -A 20 "8. Primitives" "$REPORT"

# Extract values for the summary table
LUTS=$(grep "| CLB LUTs\*" "$REPORT" | awk -F '|' '{print $3}' | tr -d ' ')
LUT6=$(grep "| LUT6" "$REPORT" | awk -F '|' '{print $3}' | tr -d ' ')
LUT5=$(grep "| LUT5" "$REPORT" | awk -F '|' '{print $3}' | tr -d ' ')
LUT4=$(grep "| LUT4" "$REPORT" | awk -F '|' '{print $3}' | tr -d ' ')
LUT3=$(grep "| LUT3" "$REPORT" | awk -F '|' '{print $3}' | tr -d ' ')
LUT2=$(grep "| LUT2" "$REPORT" | awk -F '|' '{print $3}' | tr -d ' ')
LUT1=$(grep "| LUT1" "$REPORT" | awk -F '|' '{print $3}' | tr -d ' ')
CARRY8=$(grep "| CARRY8" "$REPORT" | head -n 1 | awk -F '|' '{print $3}' | tr -d ' ')
DSPS=$(grep "| DSPs" "$REPORT" | head -n 1 | awk -F '|' '{print $3}' | tr -d ' ')
REGS=$(grep "| CLB Registers" "$REPORT" | head -n 1 | awk -F '|' '{print $3}' | tr -d ' ')
BRAMS=$(grep "| Block RAM Tile" "$REPORT" | head -n 1 | awk -F '|' '{print $3}' | tr -d ' ')

# Calculate sum of primitive LUTs (Total Logic Cells used as LUTs)
# Use 0 if empty
L6=${LUT6:-0}; L5=${LUT5:-0}; L4=${LUT4:-0}; L3=${LUT3:-0}; L2=${LUT2:-0}; L1=${LUT1:-0}
TOTAL_PRIMITIVE_LUTS=$((L6 + L5 + L4 + L3 + L2 + L1))

echo ""
echo "================================================================================"
echo "Final Resource Optimization Overview"
echo "================================================================================"
printf "%-20s | %-10s\n" "Resource" "Count"
echo "---------------------+-----------"
printf "%-20s | %s\n" "CLB LUTs (Phys)" "$LUTS"
printf "%-20s | %s\n" "  LUT6" "$LUT6"
printf "%-20s | %s\n" "  LUT5" "$LUT5"
printf "%-20s | %s\n" "  LUT4" "$LUT4"
printf "%-20s | %s\n" "  LUT3" "$LUT3"
printf "%-20s | %s\n" "  LUT2" "$LUT2"
printf "%-20s | %s\n" "  LUT1" "$LUT1"
echo "---------------------+-----------"
printf "%-20s | %s\n" "Total LUTs (Sum)" "$TOTAL_PRIMITIVE_LUTS"
echo "---------------------+-----------"
printf "%-20s | %s\n" "CARRY8" "$CARRY8"
printf "%-20s | %s\n" "DSPs" "$DSPS"
printf "%-20s | %s\n" "Registers" "$REGS"
printf "%-20s | %s\n" "BRAMs" "$BRAMS"
echo "---------------------+-----------"

