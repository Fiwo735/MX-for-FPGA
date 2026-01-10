#!/bin/bash

# FP8 DSP Sweep Benchmark
# Varies M1_USE_DSP, SOFTMAX_USE_DSP, M2_USE_DSP
# "yes" vs "logic" (Force No DSP)
# FP8 Configuration: E=5, M=2

LOG_DIR="./src/attention/synth_logs"
mkdir -p $LOG_DIR
SUMMARY_FILE="$LOG_DIR/FP8_DSP_SWEEP.csv"

# Header
echo "Config,M1_DSP,SM_DSP,M2_DSP,LUTs,DSPs" > $SUMMARY_FILE

# Common Params for FP8
SQ=4
SKV=4
DKQ=8
DV=8
K=2
OUT=32
BIT=8
SCALE=8
E=5
M=2

run_test() {
    M1_OPT=$1
    SM_OPT=$2
    M2_OPT=$3
    
    # Construct a short name for logging
    # e.g. Y_N_Y
    SHORT_NAME="${M1_OPT:0:1}_${SM_OPT:0:1}_${M2_OPT:0:1}"
    if [ "$M1_OPT" == "logic" ]; then S1="L"; else S1="Y"; fi
    if [ "$SM_OPT" == "logic" ]; then S2="L"; else S2="Y"; fi
    if [ "$M2_OPT" == "logic" ]; then S3="L"; else S3="Y"; fi
    
    CONFIG_NAME="FP8_${S1}${S2}${S3}"
    
    echo "Running $CONFIG_NAME: M1=$M1_OPT, SM=$SM_OPT, M2=$M2_OPT..."
    
    vivado -mode batch -source ./src/attention/run_synth_fp.tcl \
           -tclargs $SQ $SKV $DKQ $DV $K \
           $BIT $OUT $SCALE \
           $E $M $E $M $E $M \
           "$M1_OPT" "$M2_OPT" "$SM_OPT" > "$LOG_DIR/log_${CONFIG_NAME}.txt" 2>&1

    # Extract Results
    LATEST_RPT=$(ls -t src/attention/synth_output_int/*_util.rpt 2>/dev/null | head -n 1)
    
    if [ -f "$LATEST_RPT" ]; then
        LUTS=$(grep "| CLB LUTs" "$LATEST_RPT" | awk -F '|' '{print $3}' | tr -d ' ')
        DSPS=$(grep "| DSPs"      "$LATEST_RPT" | awk -F '|' '{print $3}' | tr -d ' ')
        
        if [ -z "$LUTS" ]; then LUTS="ERR"; fi
        if [ -z "$DSPS" ]; then DSPS="ERR"; fi
        
        echo "$CONFIG_NAME,$M1_OPT,$SM_OPT,$M2_OPT,$LUTS,$DSPS" >> $SUMMARY_FILE
        echo "  -> Done. LUTs: $LUTS, DSPs: $DSPS"
    else
        echo "$CONFIG_NAME,$M1_OPT,$SM_OPT,$M2_OPT,FAIL,FAIL" >> $SUMMARY_FILE
        echo "  -> Failed."
    fi
}

# Iterate all 8 combinations
OPTIONS=("yes" "logic")

for m1 in "${OPTIONS[@]}"; do
  for sm in "${OPTIONS[@]}"; do
    for m2 in "${OPTIONS[@]}"; do
      run_test "$m1" "$sm" "$m2"
    done
  done
done

echo "FP8 Sweep completed. Results in $SUMMARY_FILE"
