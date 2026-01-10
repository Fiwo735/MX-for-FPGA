#!/bin/bash

# Batch benchmark for No-DSP configurations
# Usage: ./run_batch.sh

LOG_DIR="./src/attention/synth_logs"
mkdir -p $LOG_DIR
SUMMARY_FILE="$LOG_DIR/SUMMARY_RESULTS.csv"

# DO NOT OVERWRITE HEADER if appending
echo "Config_Name,BitWidth,M1_Exp,M1_Man,M2_Exp,M2_Man,M3_Exp,M3_Man,LUTs,DSPs" > $SUMMARY_FILE

# Common Params
SQ=4
SKV=4
DKQ=8
DV=8
K=2
OUT=32

run_synth() {
    BIT=$1
    SCALE=$2
    E1=$3
    M1=$4
    E2=$3
    M2=$4
    E3=$3
    M3=$4
    NAME=$5
    DSP_ARG=$6
    
    echo "Running $NAME: DSP=$DSP_ARG..."
    
    vivado -mode batch -source ./src/attention/run_synth_fp.tcl \
           -tclargs $SQ $SKV $DKQ $DV $K \
           $BIT $OUT $SCALE \
           $E1 $M1 $E2 $M2 $E3 $M3 \
           "$DSP_ARG" "$DSP_ARG" "$DSP_ARG" > "$LOG_DIR/log_${NAME}.txt" 2>&1
           
    # Extract Results
    LATEST_RPT=$(ls -t src/attention/synth_output_int/*_util.rpt 2>/dev/null | head -n 1)
    
    if [ -f "$LATEST_RPT" ]; then
        LUTS=$(grep "| CLB LUTs" "$LATEST_RPT" | awk -F '|' '{print $3}' | tr -d ' ')
        DSPS=$(grep "| DSPs"      "$LATEST_RPT" | awk -F '|' '{print $3}' | tr -d ' ')
        
        if [ -z "$LUTS" ]; then LUTS="ERR"; fi
        if [ -z "$DSPS" ]; then DSPS="ERR"; fi
        
        echo "$NAME,$BIT,$E1,$M1,$E2,$M2,$E3,$M3,$LUTS,$DSPS" >> $SUMMARY_FILE
        echo "  -> Done. LUTs: $LUTS, DSPs: $DSPS"
    else
        echo "$NAME,$BIT,$E1,$M1,$E2,$M2,$E3,$M3,FAIL,FAIL" >> $SUMMARY_FILE
        echo "  -> Failed to generate report. Check log_${NAME}.txt"
    fi
}

# 1. Integer Suite (Exp=0)
# for W in {4..7}; do
#     run_synth $W $W 0 $W "INT_${W}"
# done
# for W in {8..10}; do
#     run_synth $W $W 0 $W "INT_${W}"
# done
# Run All DSPs (Expected: ~256)
# Run All DSPs (Expected: ~256)
# run_synth 8 8 0 8 "INT8_YES" "yes"

# Hybrid Test Function (M1, SM, M2)
run_hybrid() {
    BIT=$1
    SCALE=$2
    E=$3
    M=$4
    NAME=$5
    DSP_M1=$6
    DSP_SM=$7
    DSP_M2=$8
    
    echo "Running Hybrid $NAME: M1=$DSP_M1, SM=$DSP_SM, M2=$DSP_M2..."
    
    vivado -mode batch -source ./src/attention/run_synth_fp.tcl \
           -tclargs $SQ $SKV $DKQ $DV $K \
           $BIT $OUT $SCALE \
           $E $M $E $M $E $M \
           "$DSP_M1" "$DSP_M2" "$DSP_SM" > "$LOG_DIR/log_${NAME}.txt" 2>&1
           
    # Extract Results (Similar Logic)
    LATEST_RPT=$(ls -t src/attention/synth_output_int/*_util.rpt 2>/dev/null | head -n 1)
    if [ -f "$LATEST_RPT" ]; then
        LUTS=$(grep "| CLB LUTs" "$LATEST_RPT" | awk -F '|' '{print $3}' | tr -d ' ')
        DSPS=$(grep "| DSPs"      "$LATEST_RPT" | awk -F '|' '{print $3}' | tr -d ' ')
        echo "$NAME,$BIT,$E,$M,$E,$M,$E,$M,$LUTS,$DSPS" >> $SUMMARY_FILE
        echo "  -> Done. LUTs: $LUTS, DSPs: $DSPS"
    else
        echo "$NAME,FAIL" >> $SUMMARY_FILE
    fi
}

# COMPARISON: Auto Mode

# 1. FP_INT8_AUTO (Should match Baseline ~41k)
# E=0, M=7 (Total 8), DSP="auto"
run_synth 8 8 0 7 "FP_INT8_AUTO" "auto"

# 2. FP_FP8_AUTO (Should be ~43k)
# E=5, M=2 (Total 8), DSP="auto"
run_synth 8 8 5 2 "FP_FP8_AUTO" "auto"
# 2. FP Suite
# run_synth 4 4 2 1 "FP_4"
# run_synth 5 5 3 1 "FP_5"
# run_synth 6 6 3 2 "FP_6"
# run_synth 7 7 4 2 "FP_7"
# run_synth 8 8 5 2 "FP_8"
# run_synth 9 9 5 3 "FP_9"
# run_synth 10 10 6 3 "FP_10"

# 3. Random Mixed Configurations
# run_mixed 8 8 5 2 2 1 5 2 "MIX_1_HighLowHigh"
# run_mixed 4 4 2 1 5 2 2 1 "MIX_2_LowHighLow"
# run_mixed 6 6 3 2 4 3 3 2 "MIX_3_MedMix"
# run_mixed 4 4 2 1 2 1 2 1 "MIX_4_MinSpec"
# run_mixed 10 10 6 3 6 3 6 3 "MIX_5_MaxSpec"
# run_mixed 5 5 2 2 3 1 2 2 "MIX_6_OddWidths"
# run_mixed 7 7 3 3 3 3 3 3 "MIX_7_Balanced7"
# run_mixed 8 8 6 1 6 1 6 1 "MIX_8_ExpHeavy"
# run_mixed 8 8 2 5 2 5 2 5 "MIX_9_ManHeavy"
# run_mixed 8 8 5 2 5 2 5 2 "MIX_10_PaperRef"

echo "Batch completed. Results in $SUMMARY_FILE"
