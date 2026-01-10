import os
import glob
import re
import subprocess
import time
import copy
import itertools
import matplotlib
import matplotlib.pyplot as plt
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

class DesignConfig:
  def __init__(self, name, S_q=-1, S_kv=-1, d_kq=-1, d_v=-1, k=-1, scale_width=-1, M1_E=-1, M1_M=-1, M2_E=-1, M2_M=-1, M3_E=-1, M3_M=-1, accum_method1=AccumMethod.Kulisch, accum_method2=AccumMethod.Kulisch, accum_method3=AccumMethod.Kulisch, m1_dsp="yes", m2_dsp="yes", m3_dsp="yes"):
    self.name = name
    
    self.S_q = S_q
    self.S_kv = S_kv
    self.d_kq = d_kq
    self.d_v = d_v
    
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
    # TODO
    return (
      f"--S_q {self.S_q} --S_kv {self.S_kv} --d_kq {self.d_kq} --d_v {self.d_v} "
      # f"--k {self.k} --bit_width {self.bit_width} --out_width {self.out_width} --scale_width {self.scale_width}"
      f"--k {self.k} --scale_width {self.scale_width}"
    )

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
    s += f"  S_q: {self.S_q}\n"
    s += f"  S_kv: {self.S_kv}\n"
    s += f"  d_kq: {self.d_kq}\n"
    s += f"  d_v: {self.d_v}\n"
    s += f"  k: {self.k}\n"
    s += f"  scale_width: {self.scale_width}\n"
    s += f"  M1 bits: {self.M1_bits}\n"
    s += f"  M2 bits: {self.M2_bits}\n"
    s += f"  M3 bits: {self.M3_bits}\n"
    s += f"  Accumulation method 1: {self.accum_method1.value}\n"
    s += f"  Accumulation method 2: {self.accum_method2.value}\n"
    s += f"  Accumulation method 3: {self.accum_method3.value}\n"
    return s
    
  def get_vivado_tclargs(self):
    return f"{self.S_q} {self.S_kv} {self.d_kq} {self.d_v} {self.k} {self.scale_width} {self.M1_bits.exp_bits} {self.M1_bits.mant_bits} {self.M2_bits.exp_bits} {self.M2_bits.mant_bits} {self.M3_bits.exp_bits} {self.M3_bits.mant_bits} {self.accum_method1.value} {self.accum_method2.value} {self.accum_method3.value}"
  
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
    print(cls.get_design_regex())
    print(design_str)
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
  def __init__(self, design_config, power, timing, utilisation, accuracy):
    self.design_config = design_config
    self.power = power
    self.timing = timing
    self.utilisation = utilisation
    self.accuracy = accuracy
    
  def get_aggregated_resource_usage(self, keys=None):
    if keys is None:
      keys = SynthesisHandler.get_available_fpga_resources().keys()
      
    return sum(
      100 * self.utilisation[key] / SynthesisHandler.get_available_fpga_resources(key)
      for key in keys
    ) / len(keys)
    
  @classmethod
  def create_ideal_result(cls, all_results):
    design = DesignConfig("ideal")
    power = {
        "dynamic": 1e10,
        "static": 1e10,
        "total": 1e10
    }
    timing = {
        "no_violation": True,
        "max_freq": 0
    }
    utilisation = copy.deepcopy(SynthesisHandler.get_available_fpga_resources())
    accuracy = 0.0
    
    for result in all_results:
      power['total'] = min(power['total'], result.power['total'])
      power['dynamic'] = min(power['dynamic'], result.power['dynamic'])
      power['static'] = min(power['static'], result.power['static'])
      timing['max_freq'] = max(timing['max_freq'], result.timing['max_freq'])
      for key in SynthesisHandler.get_available_fpga_resources().keys():
        utilisation[key] = min(utilisation[key], result.utilisation[key])
      accuracy = max(accuracy, result.accuracy)
    
    return cls(design_config=design, power=power, timing=timing, utilisation=utilisation, accuracy=accuracy)
    
  @classmethod
  def create_ideal_result_normalised(cls):
    design = DesignConfig("ideal")
    power = {
        "dynamic": 0.0,
        "static": 0.0,
        "total": 0.0
    }
    timing = {
        "no_violation": True,
        "max_freq": 1.0
    }
    utilisation = {key: 0.0 for key in SynthesisHandler.get_available_fpga_resources().keys()}
    accuracy = 1.0
    
    return cls(design_config=design, power=power, timing=timing, utilisation=utilisation, accuracy=accuracy)
    
  @staticmethod
  def normalise_results(results):
    ideal_result = SynthesisResult.create_ideal_result(results)
    results_normalised = copy.deepcopy(results)
    for result in results_normalised:
      result.power['total'] = result.power['total'] / ideal_result.power['total']
      result.timing['max_freq'] = result.timing['max_freq'] / ideal_result.timing['max_freq']
      
      for key in SynthesisHandler.get_available_fpga_resources().keys():
        result.utilisation[key] = result.utilisation[key] / ideal_result.utilisation[key] if ideal_result.utilisation[key] > 0 else 0.0
        
      result.accuracy = result.accuracy / ideal_result.accuracy
        
    return results_normalised
  
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
      
    s += f"Accuracy: {self.accuracy:.2f}%\n"

    return s

class SynthesisHandler:
  def __init__(self, designs_to_synthesise=None, hdl_dir="./src/attention/", clock_period_ns=5):
    self.results = []
    self.designs_to_synthesise = designs_to_synthesise
    self.hdl_dir = hdl_dir
    self.clock_period_ns = clock_period_ns

    # Max frequency for the board, used to filter out results with invalid frequencies
    # Technically, max frequency is 500 MHz, but we use 400 MHz to be safe
    self.board_max_freq = 400 # MHz
    
    self.synth_output_dir = os.path.join(self.hdl_dir, "synth_output")
    
    self._time_format = "%Y%m%d_%H%M"
    
  @staticmethod
  def get_available_fpga_resources(key=None):
    # Device: xcu250figd2104-2L
    AVAILABLE_FPGA_RESOURCES = {
      "LUTs": 1728000,
      "FFs": 3456000,
      "CARRY8": 216000,
      "Muxes": 864000+432000+216000,
      "BRAMs": 2688,
      "DSPs": 12288,
    }
    
    return AVAILABLE_FPGA_RESOURCES if key is None else AVAILABLE_FPGA_RESOURCES.get(key, None)
    
  def check_if_result_exist(self, design, suffix):
    return bool(glob.glob(os.path.join(self.synth_output_dir, f"{design!r}_time_*{suffix}")))
  
  def check_if_results_exist(self, design, suffixes):
    return all(self.check_if_result_exist(design, suffix) for suffix in suffixes)
  
  def check_if_design_is_invalid(self, design):
    # All parameters must be >= 0
    for param in [design.S_q, design.S_kv, design.d_kq, design.d_v, design.k, design.scale_width]:
      if param <= 0:
        return True
      
    for mxfp_bits in [design.M1_bits, design.M2_bits, design.M3_bits]:
      if mxfp_bits.exp_bits <= 0 or mxfp_bits.mant_bits <= 0:
        return True
    
    # S_q, S_kv, d_kq, d_v must powers of 2 (including 2^0 = 1)
    for param in [design.S_q, design.S_kv, design.d_kq, design.d_v]:
      if (param & (param - 1)) != 0:
        return True
      
    # d_kq and d_v must be divisible by k
    if design.d_kq % design.k != 0 or design.d_v % design.k != 0:
      return True
    
    return False
    
  def run_synthesis(self, dry_run=False, verbose=False):
    if not self.designs_to_synthesise:
      print("No designs to synthesise specified.")
      return
    
    if verbose:
      print(f"Starting synthesis for {len(self.designs_to_synthesise)} designs...")
    
    for design in self.designs_to_synthesise:
      if self.check_if_design_is_invalid(design):
        if verbose:
          print(f"Skipping synthesis for {design!r} as design configuration is invalid.")
        continue
      
      if self.check_if_results_exist(design, ["_power.rpt", "_timing.rpt", "_util.rpt"]):
        if verbose:
          print(f"Skipping synthesis for {design!r} as results already exist.")
        continue
      
      run_synth_path = os.path.join(self.hdl_dir, "run_synth_fp.tcl")
      synthesis_cmd = f"vivado -mode batch -source {run_synth_path} -tclargs {design.get_vivado_tclargs()}"
      if verbose:
        print(f"Results for {design!r} not found, running synthesis command: {synthesis_cmd}")
      
      if dry_run:
        if verbose:
          print(f"Dry run mode enabled, skipping actual synthesis, cmd supposed to run:\n{synthesis_cmd}")
        continue
      
      try:
          start_time = time.perf_counter()
          completed_process = subprocess.run(synthesis_cmd, shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
      except subprocess.CalledProcessError as e:
          print(f"Synthesis failed for {design} with return code: {e.returncode}")
      except Exception as e:
          print(f"An unknown error occurred while running synthesis for {design}: {e}")
          
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

    dynamic_power = float(dynamic_match.group(1)) if dynamic_match else None
    static_power = float(static_match.group(1)) if static_match else None

    return dynamic_power, static_power

  def _read_timing_report(self, file_path):
    with open(file_path, 'r') as file:
      text = file.read()
      
    timing_match = re.search(r"\n\s*([-\d\.]+)\s+([-\d\.]+)\s+\d+\s+\d+\s+([-\d\.]+)\s+([-\d\.]+)\s+\d+\s+\d+", text)
    
    wns = float(timing_match.group(1)) if timing_match else None
    tns = float(timing_match.group(2)) if timing_match else None
    whs = float(timing_match.group(3)) if timing_match else None
    ths = float(timing_match.group(4)) if timing_match else None
    
    no_timing_violation = wns >= 0
    if no_timing_violation:
      max_freq = 1e3 / (self.clock_period_ns - wns)
    else:
      # max_freq = 0
      max_freq = 1e3 / (self.clock_period_ns - wns) #if (self.clock_period_ns - wns) > 1 else 0

    return no_timing_violation, max_freq
    
  def _read_utilisation_report(self, file_path):
    with open(file_path, "r") as file:
        text = file.read()

    results = {}

    patterns = {
        "LUTs": r"\|\s*CLB LUTs\*?\s*\|\s*(\d+)",
        "FFs": r"\|\s*CLB Registers\s*\|\s*(\d+)",
        "CARRY8": r"\|\s*CARRY8\s*\|\s*(\d+)",
        "F7_Muxes": r"\|\s*F7 Muxes\s*\|\s*(\d+)",
        "F8_Muxes": r"\|\s*F8 Muxes\s*\|\s*(\d+)",
        "F9_Muxes": r"\|\s*F9 Muxes\s*\|\s*(\d+)",
        "BRAMs": r"\|\s*Block RAM Tile\s*\|\s*(\d+)",
        "DSPs": r"\|\s*DSPs\s*\|\s*(\d+)"
    }

    total_muxes = 0
    for key, pattern in patterns.items():
        match = re.search(pattern, text)
        if key in ["F7_Muxes", "F8_Muxes", "F9_Muxes"]:
            total_muxes += int(match.group(1)) if match else 0
        else:
            results[key] = int(match.group(1)) if match else 0

    results["Muxes"] = total_muxes

    return results
  
  def _read_accuracy_report(self, file_path):
    # TODO early return for now
    return 75.0
    with open(file_path, 'r') as file:
      text = file.read()
      
    accuracy_match = re.search(r"Validation accuracy:\s*(\d+\.\d+)%", text)
    accuracy = float(accuracy_match.group(1))

    return accuracy
  
  def _generate_accuracy_report(self, design, accuracy_report_path):
    accuracy_cmd = f"python bert/bert_sst2.py --silent {design.get_bert_flags()}"
    
    # TODO early return for now
    return
      
    try:
        completed_process = subprocess.run(accuracy_cmd, shell=True, stdout=open(accuracy_report_path, "w"), stderr=subprocess.DEVNULL, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Accuracy measurement failed for {design} with return code: {e.returncode}")
    except Exception as e:
        print(f"An unknown error occurred while running accuracy measurement for {design}: {e}")
    
  def _process_results(self, design_str, date_time):
    file_path = os.path.join(self.synth_output_dir, f"{design_str}_time_{date_time.strftime(self._time_format)}")
    design = DesignConfig.from_str(design_str)
    
    power_report_path = f"{file_path}_power.rpt"
    timing_report_path = f"{file_path}_timing.rpt"
    utilisation_report_path = f"{file_path}_util.rpt"
    accuracy_report_path = f"{file_path}_accuracy.txt"
    
    # Generate missing accuracy report on the fly
    if not self.check_if_result_exist(design, "_accuracy.txt"):
      print(f"Accuracy report not found for {design!r}, generating on the fly...")
      self._generate_accuracy_report(design, accuracy_report_path)
      
    try:
      dynamic_power, static_power = self._read_power_report(power_report_path)
      no_timing_violation, max_freq = self._read_timing_report(timing_report_path)
      utilisation = self._read_utilisation_report(utilisation_report_path)
      accuracy = self._read_accuracy_report(accuracy_report_path)
    except FileNotFoundError as e:
      print(f"Error processing {file_path}: {e} - the report is probably being generated, try again later.")
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
      utilisation=utilisation,
      accuracy=accuracy
    )
    
    # Only include results that have valid max frequency
    if not (max_freq > 0 and max_freq < self.board_max_freq):
      print(f"Skipping result for {result} due to invalid max frequency: {max_freq:.2f} MHz.")
      return

    self.results.append(result)
      
  def _find_results(self, directory):
    pattern = re.compile(DesignConfig.get_filename_regex())
    matches = {}

    for file_path in glob.glob(os.path.join(directory, "*.rpt")):
      filename = os.path.basename(file_path)
      # print(f"\nExtracted filename: {filename}")
      
      # Match the filename against the regex
      m = pattern.match(filename)
      print(pattern)
      if not m:
        print(f"Filename {filename} does not match expected pattern, skipping.")
        continue
      
      matched_str = m.group(1)
      # print(f"Matched string: {matched_str}")
      
      result_date_time = datetime.strptime(m.group(2), self._time_format)
      # print(f"Extracted datetime: {result_date_time}")
      
      # Only store newest synthesis result
      if matched_str not in matches:
        matches[matched_str] = result_date_time
        # print(f"Added new match: {matched_str}")
      elif result_date_time > matches[matched_str]:
        matches[matched_str] = result_date_time
        # print(f"Updated match with newer datetime: {matched_str}")

    return matches
  
  def find_and_process_results(self):  
    matches = self._find_results(self.synth_output_dir)
    for design_str, date_time in matches.items():
      self._process_results(design_str, date_time)
   
  def find_pareto_optimal(self, weights):
    if not self.results:
      raise ValueError("No synthesis results available to find Pareto optimal solution.")
    
    ideal_result = SynthesisResult.create_ideal_result(self.results)
    # print(f"Ideal Result:\n{ideal_result}")
    
    # Normalise results based on the ideal result
    results_normalised = SynthesisResult.normalise_results(self.results)
    
    # Create normalised ideal result
    ideal_result_normalised = SynthesisResult.create_ideal_result_normalised()
      
    # Find the best result by finding a result that is closest to the ideal result in "distance" in the normalised space
    best_distance = 1e10
    best_index = 0
    
    for index, result in enumerate(results_normalised):
      # Compute aggregated resource metric (for plotting)
      actual_res_usage = result.get_aggregated_resource_usage()
      ideal_res_usage = ideal_result_normalised.get_aggregated_resource_usage()

      resource_diff = (actual_res_usage - ideal_res_usage) ** 2
      power_diff = (result.power['total'] - ideal_result_normalised.power['total']) ** 2
      timing_diff = (result.timing['max_freq'] - ideal_result_normalised.timing['max_freq']) ** 2

      distance = (
        power_diff * weights['power'] +
        timing_diff * weights['timing'] +
        resource_diff * weights['utilisation']
      ) ** 0.5
      
      # print(f"Distance for {result.design_config}): {distance:.4f}")
      
      if distance < best_distance:
        best_index = index
        best_distance = distance

    self.pareto_optimal = self.results[best_index]
    return self.pareto_optimal

  def _pareto_front(self, x, y, maximize_y=True):
    points = list(zip(x, y))
    
    # 1. Filter dominated points
    non_dominated = []
    for p in points:
      dominated = False
      for q in points:
        if q == p:
          continue

        better_x = q[0] <= p[0]
        better_y = q[1] >= p[1] if maximize_y else q[1] <= p[1]

        strictly_better_x = q[0] < p[0]
        strictly_better_y = q[1] > p[1] if maximize_y else q[1] < p[1]

        if better_x and better_y and (strictly_better_x or strictly_better_y):
          dominated = True
          break

      if not dominated:
        non_dominated.append(p)

    # 2. Sort by x for plotting
    non_dominated.sort(key=lambda pt: pt[0])

    # 3. Filter out "backward" y steps (enforce monotonicity in y)
    pareto = []
    best_y = -float("inf") if maximize_y else float("inf")
    for pt in non_dominated:
      if (maximize_y and pt[1] > best_y) or (not maximize_y and pt[1] < best_y):
        pareto.append(pt)
        best_y = pt[1]

    return pareto


  def plot_results(self, directory="./plots", plot_file_format="svg"):
    # color_values = np.array([r.design_config.bit_width for r in self.results])
    color_values = np.array([r.design_config.scale_width for r in self.results])
    designs = [r.design_config for r in self.results]
    resource_usages = [synth_result.get_aggregated_resource_usage() for synth_result in self.results]
    powers = [synth_result.power['total'] for synth_result in self.results]
    frequencies = [synth_result.timing['max_freq'] for synth_result in self.results]
    accuracies = [synth_result.accuracy for synth_result in self.results]
    
    self._plot(
      designs=designs,
      x=powers,
      y=frequencies,
      color_values=color_values,
      xlabel="Power (W)",
      ylabel="Max Frequency (MHz)",
      title=f"Max Frequency vs Power",
      filename=f"max_freq_vs_power.{plot_file_format}",
      directory=directory,
    )

  def _plot(self, designs, x, y, color_values, xlabel, ylabel, title, filename, directory, do_pareto_front=True, do_pareto_optimal=True, do_best_fit_line=False):
    # Differentiate designs by block_size k
    marker_map = {
      2: "o",
      4: "^",
    }

    cmap = matplotlib.colormaps["viridis"].resampled(len(np.unique(color_values)))
    bounds = np.arange(color_values.min() - 0.5, color_values.max() + 1.5, 1)
    norm = matplotlib.colors.BoundaryNorm(bounds, cmap.N)

    fig, ax = plt.subplots(figsize=(10, 6))
    plotted_markers = {}

    for design, xi, yi, cval in zip(designs, x, y, color_values):
      block_size = design.k
      label = f"k={block_size}"
      marker = marker_map.get(block_size, "s")
      ax.scatter(
        xi, yi,
        c=[cmap(norm(cval))],
        alpha=1.0,
        s=120,
        marker=marker,
        label=label
      )
      plotted_markers[label] = marker

    ax.set_title(title, fontsize=18)
    ax.set_xlabel(xlabel, fontsize=14)
    ax.set_ylabel(ylabel, fontsize=14)
    ax.tick_params(axis='x', labelsize=12)
    ax.tick_params(axis='y', labelsize=12)
    ax.grid(True)

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, boundaries=bounds,
                        ticks=np.arange(color_values.min(), color_values.max() + 1))
    cbar.set_label("Total bit width", fontsize=14)
    cbar.ax.tick_params(labelsize=12)

    handles, labels = ax.get_legend_handles_labels()
    unique_labels = list(dict.fromkeys(labels))

    black_handles = [
      plt.Line2D([], [], marker=plotted_markers[label],
                color="black", markerfacecolor="black",
                linestyle="", markersize=10)
      for label in unique_labels
    ]
    
    # === Compute and plot Pareto front ===
    if do_pareto_front:
      maximize_y = not ylabel.lower().startswith("power")
      pareto_points = self._pareto_front(x, y, maximize_y=maximize_y)
      pareto_x = [p[0] for p in pareto_points]
      pareto_y = [p[1] for p in pareto_points]

      ax.plot(pareto_x, pareto_y, linestyle="dashdot", color="black", linewidth=1.2)
      
      pareto_front_legend = matplotlib.lines.Line2D([], [], color="black", linestyle="dashdot", linewidth=1.5, label="Pareto front")
      
      black_handles += [pareto_front_legend]
      unique_labels += ["Pareto front"]
    
    # === Highlight pareto optimal point ===
    if do_pareto_optimal and self.pareto_optimal is not None:
      # Compute X and Y of the pareto optimal point for this plot
      if xlabel.startswith("Resource"):
        x_val = self.pareto_optimal.get_aggregated_resource_usage()
      else:
        x_val = self.pareto_optimal.power["total"]

      if ylabel.startswith("Power"):
        y_val = self.pareto_optimal.power["total"]
      else:
        y_val = self.pareto_optimal.timing["max_freq"]

      radius_coeff = 0.04
      radius_x = radius_coeff * (ax.get_xlim()[1] - ax.get_xlim()[0])
      radius_y = radius_coeff * (ax.get_ylim()[1] - ax.get_ylim()[0])

      ellipse = matplotlib.patches.Ellipse(
        (x_val, y_val),
        width=2 * radius_x,
        height=2 * radius_y,
        fill=False,
        linestyle="dotted",
        edgecolor="black",
        linewidth=1.5
      )

      ax.add_patch(ellipse)
      
      ellipse_legend = matplotlib.patches.Ellipse(
        (0, 0),  # position doesn't matter for legend
        width=0.1, height=0.5,  # small size for legend
        fill=False,
        linestyle="dotted",
        edgecolor="black",
        linewidth=1.5,
        label="Ideal* Pareto"
      )
      
      black_handles += [ellipse_legend]
      unique_labels += ["Optimal*"]
      
    # # === Plot best fit line (linear regression) ===
    # if do_best_fit_line and len(x) > 1:
    #   # Fit
    #   coeffs = np.polyfit(x, y, 1)
    #   fit_x = np.linspace(min(x), max(x), 100)
    #   fit_y = np.polyval(coeffs, fit_x)

    #   # Compute R^2
    #   y_mean = np.mean(y)
    #   ss_tot = np.sum((y - y_mean) ** 2)
    #   ss_res = np.sum((y - np.polyval(coeffs, x)) ** 2)
    #   r2 = 1 - (ss_res / ss_tot)

    #   # Plot the line
    #   ax.plot(fit_x, fit_y, color="gray", linestyle="dashdot", linewidth=1.3)

    #   # Create a custom handle with R² in label
    #   best_fit_label = f"Fit, R$^2$ = {r2:.3f}"
    #   best_fit_handle = plt.Line2D([], [], color="gray", linestyle="dashdot", linewidth=1.3, label=best_fit_label)
    #   black_handles += [best_fit_handle]
    #   unique_labels += [best_fit_label]
      
    ax.legend(black_handles, unique_labels, fontsize=14)

    fig.tight_layout()
    fig.savefig(os.path.join(directory, filename))
  
  def __str__(self):
    spacer = "="*60 + "\n"
    return (
      f"\t\t\t{len(self.results)} Synthesis Results:\n" +
      spacer + ("\n" + spacer).join([f"{result!s}" for result in self.results]) + spacer
    )
    
if __name__ == "__main__":
  parser = ArgumentParser(description='Run DSE for attention module synthesis')
  parser.add_argument('--dry', action='store_true', help='Dry run, do not run synthesis')
  parser.add_argument('--verbose', action='store_true', help='Enable verbose output')
  args = parser.parse_args()
  
  designs_to_synthesise = [
    DesignConfig(name, S_q, S_kv, d_kq, d_v, k, scale_width, M1_E, M1_M, M2_E, M2_M, M3_E, M3_M, accum_method)
    for name in ["attention_fp"]
    for S_q in [4]
    for S_kv in [4]
    for d_kq in [4]
    for d_v in [4]
    for k in [2]
    for scale_width in [8]
    for M1_E, M1_M in [(4, 3)]
    for M2_E, M2_M in [(4, 3)]
    for M3_E, M3_M in [(4, 3)]
    for accum_method in [AccumMethod.Kulisch]
  ]
  
  synthesis_handler = SynthesisHandler(designs_to_synthesise)
  synthesis_handler.run_synthesis(dry_run=args.dry, verbose=args.verbose)

  synthesis_handler.find_and_process_results()
  # print(synthesis_handler)

  pareto_optimal = synthesis_handler.find_pareto_optimal(weights={'power': 1.0, 'timing': 1.0, 'utilisation': 1.0})
  print(f"\nPareto Optimal Result:\n{pareto_optimal}")

  synthesis_handler.plot_results(directory="./plots", plot_file_format="png")