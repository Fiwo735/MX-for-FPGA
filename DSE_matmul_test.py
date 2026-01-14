import os
import glob
import re
import subprocess
import time
import copy
import itertools
import numpy as np
from enum import Enum
from datetime import datetime
from argparse import ArgumentParser

class MXFPBits:
  def __init__(self, exp_bits, mant_bits):
    self.exp_bits = exp_bits
    self.mant_bits = mant_bits
    
  def __repr__(self):
    return f"E{self.exp_bits}M{self.mant_bits}"
  
class AccumMethod(Enum):
  Kulisch = "KULISCH"
  Kahan = "KAHAN"
  Neumaier = "NEUMAIER"
  Klein = "KLEIN"
  TwoSum = "TWOSUM"
  FastTwoSum = "FASTTWOSUM"

class DesignConfig:
  def __init__(self, name, S_q=-1, S_kv=-1, d_kq=-1, d_v=-1, k=-1, scale_width=-1, M1_E=-1, M1_M=-1, M2_E=-1, M2_M=-1, M3_E=-1, M3_M=-1, accum_method1=AccumMethod.Kulisch, accum_method2=AccumMethod.Kulisch, accum_method3=AccumMethod.Kulisch, m1_dsp="yes", m2_dsp="yes", m3_dsp="yes"):
    self.name = name
    
    self.S_q = S_q # x_rows
    self.S_kv = S_kv # y_cols
    self.d_kq = d_kq # vec_elem_count
    self.d_v = d_v # Unused in simple matmul synth
    
    self.k = k
    self.scale_width = scale_width
    
    self.M1_bits = MXFPBits(M1_E, M1_M)
    self.M2_bits = MXFPBits(M2_E, M2_M)
    self.M3_bits = MXFPBits(M3_E, M3_M)
    self.accum_method1 = accum_method1
    self.accum_method2 = accum_method2
    self.accum_method3 = accum_method3

    self.m1_dsp = m1_dsp
    self.m2_dsp = m2_dsp
    self.m3_dsp = m3_dsp
    
  def get_bert_flags(self):
    return ""

  def __repr__(self):
    return (
      f"{self.name}_S_q_{self.S_q}_S_kv_{self.S_kv}_d_kq_{self.d_kq}_d_v_{self.d_v}_k_{self.k}_"
      f"scale_width_{self.scale_width}_M1_E_{self.M1_bits.exp_bits}_M1_M_{self.M1_bits.mant_bits}_"
      f"M2_E_{self.M2_bits.exp_bits}_M2_M_{self.M2_bits.mant_bits}_M3_E_{self.M3_bits.exp_bits}_M3_M_{self.M3_bits.mant_bits}_"
      f"ACCUM_METHOD_{self.accum_method1.value}_{self.accum_method2.value}_{self.accum_method3.value}_"
      f"DSP_{self.m1_dsp}_{self.m2_dsp}_{self.m3_dsp}"
    )
    
  def __str__(self):
    s = f"Design: {self.name}\n"
    s += f"  Rows: {self.S_q}\n"
    s += f"  Cols: {self.S_kv}\n"
    s += f"  Inner Dim: {self.d_kq}\n"
    s += f"  k: {self.k}\n"
    s += f"  Precision: {self.M1_bits}\n" # Input Precision
    s += f"  Accumulation method: {self.accum_method1.value}\n"
    return s
    
  def get_vivado_tclargs(self):
    return f"{self.S_q} {self.S_kv} {self.d_kq} {self.d_v} {self.k} {self.scale_width} {self.M1_bits.exp_bits} {self.M1_bits.mant_bits} {self.M2_bits.exp_bits} {self.M2_bits.mant_bits} {self.M3_bits.exp_bits} {self.M3_bits.mant_bits} {self.accum_method1.value} {self.accum_method2.value} {self.accum_method3.value} {self.m1_dsp} {self.m2_dsp} {self.m3_dsp} {self.name}"
  
  @staticmethod
  def get_filename_regex():
    return r"([^/]+_S_q_\d+_S_kv_\d+_d_kq_\d+_d_v_\d+_k_\d+_scale_width_\d+_M1_E_\d+_M1_M_\d+_M2_E_\d+_M2_M_\d+_M3_E_\d+_M3_M_\d+_ACCUM_METHOD_[A-Z]+_[A-Z]+_[A-Z]+_DSP_[a-zA-Z]+_[a-zA-Z]+_[a-zA-Z]+)_time_(\d+_\d+)"
  
  @staticmethod
  def get_design_regex():
    return r"([^/]+)_S_q_(\d+)_S_kv_(\d+)_d_kq_(\d+)_d_v_(\d+)_k_(\d+)_scale_width_(\d+)_M1_E_(\d+)_M1_M_(\d+)_M2_E_(\d+)_M2_M_(\d+)_M3_E_(\d+)_M3_M_(\d+)_ACCUM_METHOD_([A-Z]+)_([A-Z]+)_([A-Z]+)_DSP_([a-zA-Z]+)_([a-zA-Z]+)_([a-zA-Z]+)"
  
  @classmethod
  def from_str(cls, design_str):
    details = re.search(
      cls.get_design_regex(),
      design_str
    ) 
    if not details:
      raise ValueError(f"Design string {design_str} does not match expected pattern.")
    
    name = details.group(1)
    S_q = int(details.group(2))
    S_kv = int(details.group(3))
    d_kq = int(details.group(4))
    d_v = int(details.group(5))
    k = int(details.group(6))
    scale_width = int(details.group(7))
    M1_E = int(details.group(8))
    M1_M = int(details.group(9))
    M2_E = int(details.group(10))
    M2_M = int(details.group(11))
    M3_E = int(details.group(12))
    M3_M = int(details.group(13))
    accum_method1 = AccumMethod(details.group(14))
    accum_method2 = AccumMethod(details.group(15))
    accum_method3 = AccumMethod(details.group(16))
    m1_dsp = details.group(17)
    m2_dsp = details.group(18)
    m3_dsp = details.group(19)
    
    return cls(name=name, S_q=S_q, S_kv=S_kv, d_kq=d_kq, d_v=d_v, k=k, scale_width=scale_width, M1_E=M1_E, M1_M=M1_M, M2_E=M2_E, M2_M=M2_M, M3_E=M3_E, M3_M=M3_M, accum_method1=accum_method1, accum_method2=accum_method2, accum_method3=accum_method3, m1_dsp=m1_dsp, m2_dsp=m2_dsp, m3_dsp=m3_dsp)

class SynthesisResult:
  def __init__(self, design_config, power, timing, utilisation, accuracy=0.0):
    self.design_config = design_config
    self.power = power
    self.timing = timing
    self.utilisation = utilisation
    self.accuracy = accuracy
    
  def __str__(self):
    s = f"{self.design_config!s}\n"
    s += f"Power: {self.power['total']:.2f} W (Dynamic {self.power['dynamic']:.2f} W, Static {self.power['static']:.2f} W)\n"
    
    s += f"Max freq: {self.timing['max_freq']:.2f} MHz"
    if not self.timing['no_violation']:
      s += " (TIMING VIOLATION)"
    s += "\n"
    
    s += "Resource utilisation:\n"
    for key, value in self.utilisation.items():
      s += f"\t{key}: {value:,} ({(value / SynthesisHandler.get_available_fpga_resources(key)) * 100:.2f}%)\n"
      
    return s

class SynthesisHandler:
  def __init__(self, designs_to_synthesise=None, hdl_dir="./src/attention/", clock_period_ns=5):
    self.results = []
    self.designs_to_synthesise = designs_to_synthesise
    self.hdl_dir = hdl_dir
    self.clock_period_ns = clock_period_ns
    
    # Updated output dir for Matmul
    self.synth_output_dir = os.path.join(self.hdl_dir, "synth_output_matmul")
    if not os.path.exists(self.synth_output_dir):
        os.makedirs(self.synth_output_dir)
    
    self._time_format = "%Y%m%d_%H%M"
    
  @staticmethod
  def get_available_fpga_resources(key=None):
    # Device: xcv80
    AVAILABLE_FPGA_RESOURCES = {
      "LUTs": 1728000, 
      "FFs": 3456000,
      "CARRY8": 216000,
      "Muxes": 864000+432000+216000,
      "BRAMs": 2688,
      "DSPs": 12288,
    }
    
    return AVAILABLE_FPGA_RESOURCES if key is None else AVAILABLE_FPGA_RESOURCES.get(key, 1)
    
  def check_if_result_exist(self, design, suffix):
    return bool(glob.glob(os.path.join(self.synth_output_dir, f"{design!r}_time_*{suffix}")))
  
  def check_if_results_exist(self, design, suffixes):
    return all(self.check_if_result_exist(design, suffix) for suffix in suffixes)
  
  def check_if_design_is_invalid(self, design):
    return False 
    
  def run_synthesis(self, dry_run=False, verbose=False):
    if not self.designs_to_synthesise:
      print("No designs to synthesise specified.")
      return
    
    if verbose:
      print(f"Starting synthesis for {len(self.designs_to_synthesise)} designs...")
    
    for design in self.designs_to_synthesise:
      if self.check_if_results_exist(design, ["_power.rpt", "_timing.rpt", "_util.rpt"]):
        if verbose:
          print(f"Skipping synthesis for {design!r} as results already exist.")
        continue
      
      # Point to new script
      run_synth_path = os.path.join(self.hdl_dir, "run_synth_matmul.tcl")
      synthesis_cmd = f"/mnt/applications/Xilinx/24.2/Vivado/2024.2/bin/vivado -mode batch -source {run_synth_path} -tclargs {design.get_vivado_tclargs()}"
      
      if verbose:
        print(f"Running synthesis command: {synthesis_cmd}")
      
      if dry_run:
        continue
      
      try:
          start_time = time.perf_counter()
          completed_process = subprocess.run(synthesis_cmd, shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
      except subprocess.CalledProcessError as e:
          print(f"Synthesis failed for {design} with return code: {e.returncode}")
          
      end_time = time.perf_counter()
      if verbose:
        print(f"Synthesis for {design!r} completed in {end_time - start_time:.2f} seconds.")
          
    if verbose:
      print("Synthesis completed for all designs.")

  def _read_power_report(self, file_path):
    with open(file_path, 'r') as file:
      text = file.read()
      
    dynamic_match = re.search(r"Dynamic \(W\)\s*\|\s*([\d.]+)", text)
    static_match = re.search(r"Device Static \(W\)\s*\|\s*([\d.]+)", text)

    dynamic_power = float(dynamic_match.group(1)) if dynamic_match else 0.0
    static_power = float(static_match.group(1)) if static_match else 0.0

    return dynamic_power, static_power

  def _read_timing_report(self, file_path):
    with open(file_path, 'r') as file:
      text = file.read()
      
    timing_match = re.search(r"\n\s*([-\d\.]+)\s+([-\d\.]+)\s+\d+\s+\d+\s+([-\d\.]+)\s+([-\d\.]+)\s+\d+\s+\d+", text)
    
    wns = float(timing_match.group(1)) if timing_match else None
    
    no_timing_violation = wns is not None and wns >= 0
    if no_timing_violation:
      max_freq = 1e3 / (self.clock_period_ns - wns)
    elif wns is not None:
      max_freq = 1e3 / (self.clock_period_ns - wns) 
    else:
      max_freq = 0

    return no_timing_violation, max_freq
    
  def _read_utilisation_report(self, file_path):
    with open(file_path, "r") as file:
        text = file.read()

    results = {}
    patterns = {
        "LUTs": r"\|\s*CLB LUTs\*?\s*\|\s*(\d+)",
        "FFs": r"\|\s*CLB Registers\s*\|\s*(\d+)",
        "CARRY8": r"\|\s*CARRY8\s*\|\s*(\d+)",
        "BRAMs": r"\|\s*Block RAM Tile\s*\|\s*(\d+)",
        "DSPs": r"\|\s*DSPs\s*\|\s*(\d+)"
    }

    for key, pattern in patterns.items():
        match = re.search(pattern, text)
        results[key] = int(match.group(1)) if match else 0
    
    # Fill missing keys like Muxes with 0 for now to avoid errors
    if "Muxes" not in results: results["Muxes"] = 0

    return results
  
  def _process_results(self, design_str, date_time):
    file_path = os.path.join(self.synth_output_dir, f"{design_str}_time_{date_time.strftime(self._time_format)}")
    design = DesignConfig.from_str(design_str)
    
    power_report_path = f"{file_path}_power.rpt"
    timing_report_path = f"{file_path}_timing.rpt"
    utilisation_report_path = f"{file_path}_util.rpt"
    
    try:
      dynamic_power, static_power = self._read_power_report(power_report_path)
      no_timing_violation, max_freq = self._read_timing_report(timing_report_path)
      utilisation = self._read_utilisation_report(utilisation_report_path)
    except FileNotFoundError as e:
      print(f"Error processing {file_path}: {e}")
      return
    
    result = SynthesisResult(
      design_config=design,
      power={
          "dynamic": dynamic_power,
          "static": static_power,
          "total": dynamic_power + static_power
      },
      timing={
          "no_violation": no_timing_violation,
          "max_freq": max_freq
      },
      utilisation=utilisation
    )
    
    self.results.append(result)
      
  def _find_results(self, directory):
    pattern = re.compile(DesignConfig.get_filename_regex())
    matches = {}

    for file_path in glob.glob(os.path.join(directory, "*.rpt")):
      filename = os.path.basename(file_path)
      m = pattern.match(filename)
      if not m: continue
      
      matched_str = m.group(1)
      result_date_time = datetime.strptime(m.group(2), self._time_format)
      
      if matched_str not in matches:
        matches[matched_str] = result_date_time
      elif result_date_time > matches[matched_str]:
        matches[matched_str] = result_date_time

    return matches
  
  def find_and_process_results(self):  
    matches = self._find_results(self.synth_output_dir)
    for design_str, date_time in matches.items():
      self._process_results(design_str, date_time)
  
  def find_pareto_optimal(self):
      # Simple find best frequency for now, or least area
      if not self.results: return None
      # Return result with max frequency
      return max(self.results, key=lambda r: r.timing['max_freq'])

  def __str__(self):
    spacer = "="*60 + "\n"
    return (
      f"\t\t\t{len(self.results)} Synthesis Results:\n" +
      spacer + ("\n" + spacer).join([f"{result!s}" for result in self.results]) + spacer
    )
    
if __name__ == "__main__":
  parser = ArgumentParser(description='Run DSE for MatMul module assignment')
  parser.add_argument('--dry', action='store_true', help='Dry run')
  parser.add_argument('--verbose', action='store_true', help='Verbose output')
  args = parser.parse_args()
  
  # Setup Logging
  timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
  log_filename = f"DSE_run_matmul_{timestamp}.log"
  
  class Tee(object):
    def __init__(self, filename):
      self.terminal = sys.stdout
      self.log = open(filename, "a")
    def write(self, message):
      self.terminal.write(message)
      self.log.write(message)
    def flush(self):
      self.terminal.flush()
      self.log.flush()

  import sys
  original_stdout = sys.stdout
  original_stderr = sys.stderr
  sys.stdout = Tee(log_filename)
  sys.stderr = sys.stdout # Redirect stderr to same log

  print(f"Starting DSE for MatMul at {timestamp}")
  print(f"Logging to: {log_filename}\n")
  
  # Define MatMul Configurations
  # MatMul typically uses Accum Method 1
  designs_to_synthesise = [
    DesignConfig("matmul", 
                 S_q=32, S_kv=32, d_kq=32, d_v=32,
                 k=4, 
                 scale_width=8, 
                 M1_E=4, M1_M=3, 
                 M2_E=4, M2_M=3, 
                 M3_E=4, M3_M=3,
                 accum_method1=AccumMethod.Kulisch,
                 m1_dsp="yes")
  ]

  try:
      synthesis_handler = SynthesisHandler(designs_to_synthesise)
      synthesis_handler.run_synthesis(dry_run=args.dry, verbose=args.verbose)
      synthesis_handler.find_and_process_results()
      
      print(synthesis_handler)
      
      pareto = synthesis_handler.find_pareto_optimal()
      if pareto:
          print(f"Pareto Optimal Result:\n{pareto}")
          
  except Exception as e:
      print(f"An error occurred during DSE execution: {e}")
      import traceback
      traceback.print_exc()
  finally:
      # Restore stdout/stderr (cleanliness specifically for python shell usage, though script ending closes file handles)
      sys.stdout = original_stdout
      sys.stderr = original_stderr
