import glob
import re
import os

def parse_util(filepath):
    luts = 0
    dsps = 0
    with open(filepath, 'r') as f:
        content = f.read()
        # Updated Regex based on file content observation
        # | CLB LUTs*                  | 1376 | ...
        lut_pattern = re.compile(r"\|\s*CLB LUTs\*\s*\|\s*(\d+)\s*\|")
        # | DSPs      |    0 | ...
        dsp_pattern = re.compile(r"\|\s*DSPs\s*\|\s*(\d+)\s*\|")
        
        matches = lut_pattern.findall(content)
        if matches:
            luts = int(matches[0])
            
        dsp_matches = dsp_pattern.findall(content)
        if dsp_matches:
            dsps = int(dsp_matches[0])
            
    return luts, dsps

def parse_timing(filepath):
    wns = "N/A"
    try:
        with open(filepath, 'r') as f:
            lines = f.readlines()
            # Look for WNS(ns) header then next line
            # Header: "    WNS(ns)      TNS(ns) ..." (spaces vary)
            # Next line: "      3.989        0.000 ..."
            capture_next = False
            for line in lines:
                if "WNS(ns)" in line and "TNS(ns)" in line:
                    capture_next = True
                    continue
                if capture_next:
                    if line.strip() == "": continue # skip empty lines
                    parts = line.split()
                    if len(parts) > 0:
                        wns = parts[0]
                    break
    except FileNotFoundError:
        pass
    return wns

print("| Middle Accumulator | LUTs | DSPs | WNS (ns) |")
print("| :--- | :--- | :--- | :--- |")

files = glob.glob("./src/attention/synth_output/*20260113*_util.rpt")
results = []

for util_file in files:
    # Extract Accumulator Method from filename
    # Pattern: ..._ACCUM_METHOD_NEUMAIER_[MIDDLE]_KLEIN_...
    match = re.search(r"ACCUM_METHOD_NEUMAIER_([A-Za-z]+)_KLEIN", util_file)
    if match:
        middle_accum = match.group(1)
        luts, dsps = parse_util(util_file)
        
        timing_file = util_file.replace("_util.rpt", "_timing.rpt")
        wns = parse_timing(timing_file)
        
        results.append((middle_accum, luts, dsps, wns))

results.sort(key=lambda x: x[0])

for r in results:
    print(f"| {r[0]} | {r[1]} | {r[2]} | {r[3]} |")
