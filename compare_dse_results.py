import glob
import re
import os

def parse_rpt(filepath):
    metrics = {'LUT': '-', 'DSP': '-', 'WNS': '-'}
    if not os.path.exists(filepath):
        return metrics
        
    # Utilization
    if filepath.endswith('_util.rpt'):
        try:
            with open(filepath, 'r') as f:
                content = f.read()
                # Try generic CLB LUTs (handles asterisk)
                lut_match = re.search(r'\|\s*CLB LUTs\*?\s*\|\s*(\d+)\s*\|', content)
                if lut_match:
                    metrics['LUT'] = int(lut_match.group(1))
                else:
                    # Try Slice LUTs fallback
                    lut_match = re.search(r'\|\s*Slice LUTs\*?\s*\|\s*(\d+)\s*\|', content)
                    if lut_match:
                        metrics['LUT'] = int(lut_match.group(1))

                dsp_match = re.search(r'\|\s*DSPs\s*\|\s*(\d+)\s*\|', content)
                if dsp_match:
                    metrics['DSP'] = int(dsp_match.group(1))
        except: pass

    # Timing
    if filepath.endswith('_timing.rpt'):
        try:
            with open(filepath, 'r') as f:
                content = f.read()
                # Look for WNS under "Design Timing Summary" table
                # Usually lines like:
                # WNS(ns) TNS(ns) ...
                # 1.234 0.000 ...
                # We search for the first float after WNS header
                wns_match = re.search(r'WNS\(ns\).*?\n\s*(-?\d+\.\d+)', content, re.DOTALL)
                if wns_match:
                    metrics['WNS'] = float(wns_match.group(1))
        except: pass
            
    return metrics

def get_params(filename):
    tag = "PREVIOUS" if "PREVIOUS_CODE" in filename else "NEW"
    
    # Extract M1_E, M1_M
    match = re.search(r'M1_E_(\d+)_M1_M_(\d+)', filename)
    if not match: return None
    
    m1_e = int(match.group(1))
    m1_m = int(match.group(2))
    width = 1 + m1_e + m1_m
    dtype = "INT" if m1_e == 0 else "FP"
    
    return {'tag': tag, 'width': width, 'dtype': dtype, 'config': f"{dtype}{width} (E{m1_e}M{m1_m})"}

def main():
    base_dir = "src/attention/synth_output/"
    util_files = glob.glob(os.path.join(base_dir, "*_util.rpt"))
    
    # Store: data[(dtype, width, config)] = { 'NEW': {metrics}, 'PREVIOUS': {metrics} }
    data = {}
    
    for u_file in util_files:
        p = get_params(u_file)
        if not p: continue
        
        key = (p['dtype'], p['width'], p['config'])
        if key not in data:
            data[key] = {'NEW': {'LUT': '-', 'DSP': '-', 'WNS': '-'}, 
                         'PREVIOUS': {'LUT': '-', 'DSP': '-', 'WNS': '-'}}
        
        tag = p['tag']
        metrics = parse_rpt(u_file)
        data[key][tag]['LUT'] = metrics['LUT']
        data[key][tag]['DSP'] = metrics['DSP']
        
        # Timing
        t_file = u_file.replace('_util.rpt', '_timing.rpt')
        if os.path.exists(t_file):
            t_metrics = parse_rpt(t_file)
            data[key][tag]['WNS'] = t_metrics['WNS']

    # Sort Keys
    sorted_keys = sorted(data.keys(), key=lambda x: (0 if x[0]=='INT' else 1, x[1]))

    # Print Comparison Table
    print("### DSE Comparison (New vs Old)")
    print("| Config | New LUT | Old LUT | New DSP | Old DSP | New WNS | Old WNS |")
    print("|---|---|---|---|---|---|---|")
    
    for key in sorted_keys:
        d = data[key]
        n = d['NEW']
        p = d['PREVIOUS']
        print(f"| {key[2]} | {n['LUT']} | {p['LUT']} | {n['DSP']} | {p['DSP']} | {n['WNS']} | {p['WNS']} |")

    # Group by Width for Side-by-Side
    # width_map[width] = {'INT': metrics, 'FP': metrics}
    width_map = {}
    for key in sorted_keys:
        dtype = key[0]
        width = key[1]
        n = data[key]['NEW']
        
        # Only care if we have data
        if n['LUT'] == '-' and n['DSP'] == '-':
            continue
            
        if width not in width_map:
            width_map[width] = {'INT': {'LUT': '-', 'DSP': '-', 'WNS': '-'}, 'FP': {'LUT': '-', 'DSP': '-', 'WNS': '-'}}
        
        width_map[width][dtype] = n

    # Print Side-by-Side Table
    print("\n### Side-by-Side Comparison: INT vs FP (New Codebase)")
    print("| Width | INT LUT | INT DSP | INT WNS | FP LUT | FP DSP | FP WNS | Delta LUT (FP-INT) |")
    print("|---|---|---|---|---|---|---|---|")
    
    for w in sorted(width_map.keys()):
        i = width_map[w]['INT']
        f = width_map[w]['FP']
        
        # Calculate Delta LUT if possible
        delta = "-"
        if i['LUT'] != '-' and f['LUT'] != '-':
            delta = f"{f['LUT'] - i['LUT']:+}"
            
        print(f"| {w} | {i['LUT']} | {i['DSP']} | {i['WNS']} | {f['LUT']} | {f['DSP']} | {f['WNS']} | {delta} |")

if __name__ == "__main__":
    main()
