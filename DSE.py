import os
import glob
import re
import subprocess
import time
import copy
import itertools
import random
import time
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from enum import Enum
from datetime import datetime
from argparse import ArgumentParser
from concurrent.futures import ProcessPoolExecutor, as_completed
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from gplearn.genetic import SymbolicRegressor

DEBUG_COUNTER = 0

def gplearn_expr_to_math(expr):
    """
    Convert a gplearn prefix expression (e.g. mul(add(X0, X1), log(X2)))
    into a human-readable math expression.
    """

    def split_args(s):
        depth = 0
        for i, c in enumerate(s):
            if c == ',' and depth == 0:
                return s[:i], s[i+1:]
            elif c == '(':
                depth += 1
            elif c == ')':
                depth -= 1
        raise ValueError(f"Cannot split arguments: {s}")

    def parse(e):
        e = e.strip()

        # Variable or constant
        if not '(' in e:
            return e

        op, rest = e.split('(', 1)
        rest = rest[:-1]  # strip trailing ')'

        if op == 'add':
            a, b = split_args(rest)
            return f"({parse(a)} + {parse(b)})"

        if op == 'sub':
            a, b = split_args(rest)
            return f"({parse(a)} − {parse(b)})"

        if op == 'mul':
            a, b = split_args(rest)
            return f"({parse(a)} · {parse(b)})"

        if op == 'div':
            a, b = split_args(rest)
            return f"({parse(a)} ÷ {parse(b)})"

        if op == 'log':
            return f"log({parse(rest)})"

        if op == 'sqrt':
            return f"√({parse(rest)})"

        if op == 'neg':
            return f"−({parse(rest)})"

        # Fallback
        return e

    return parse(expr)

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
    
  def get_total_bits(self):
    return (
      (self.M1_bits.exp_bits + self.M1_bits.mant_bits) +
      (self.M2_bits.exp_bits + self.M2_bits.mant_bits) +
      (self.M3_bits.exp_bits + self.M3_bits.mant_bits)
    )
    
  def get_bert_flags(self):
    return (
      "--model_id 'meta-llama/Llama-3.2-1B' "
      f'--config \'k_quantizer={{"quant":"MXFPQuantizer","man_w":{self.M1_bits.mant_bits},"exp_w":{self.M1_bits.exp_bits},"group_size":{self.k}}}\' '
      f'--config \'s_quantizer={{"quant":"MXFPQuantizer","man_w":{self.M2_bits.mant_bits},"exp_w":{self.M2_bits.exp_bits},"group_size":{self.k}}}\' '
      f'--config \'v_quantizer={{"quant":"MXFPQuantizer","man_w":{self.M3_bits.mant_bits},"exp_w":{self.M3_bits.exp_bits},"group_size":{self.k}}}\' '
      f'--config \'sum_type_attn_s="{self.accum_method1.value}"\' '
      f'--config \'sum_type_smax="{self.accum_method2.value}"\' '
      f'--config \'sum_type_attn_o="{self.accum_method3.value}"\' '
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
    return f"{self.S_q} {self.S_kv} {self.d_kq} {self.d_v} {self.k} {self.scale_width} {self.M1_bits.exp_bits} {self.M1_bits.mant_bits} {self.M2_bits.exp_bits} {self.M2_bits.mant_bits} {self.M3_bits.exp_bits} {self.M3_bits.mant_bits} {self.accum_method1.value} {self.accum_method2.value} {self.accum_method3.value} {self.m1_dsp} {self.m2_dsp} {self.m3_dsp} {self.name}"
  
  def get_tcl_filename(self):
    if self.name == "attention_fp":
      return "run_synth_fp.tcl"
    elif self.name == "matmul_fp":
      return "run_synth_matmul.tcl"
    elif self.name == "mxint_softmax":
      return "run_synth_softmax.tcl"
    
    raise ValueError(f"Unsupported design name: {self.name}")
  
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
    # print(cls.get_design_regex())
    # print(design_str)
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
    
    # TODO placeholder
    self.resource_multipliers = {
      "LUTs": 30.0,
      "FFs": 15.0,
      "BRAMs": 45.0,
      "DSPs": 60.0,
    }
    
  def get_aggregated_resource_usage(self, keys=None, use_multipliers=False):
    if keys is None:
      keys = SynthesisHandler.get_available_fpga_resources().keys()
      
    utilisation_sum = 0.0
    
    for key in keys:
      current_utilisation = self.utilisation[key] / SynthesisHandler.get_available_fpga_resources(key)
      if use_multipliers:
        current_utilisation *= self.resource_multipliers[key]
        
      utilisation_sum += current_utilisation
      
    return utilisation_sum / len(keys)
  
  def get_throughput_multiplier(self):
    return 1 # TODO placeholder
  
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
    accuracy = 1e10
    
    for result in all_results:
      power['total'] = min(power['total'], result.power['total'])
      power['dynamic'] = min(power['dynamic'], result.power['dynamic'])
      power['static'] = min(power['static'], result.power['static'])
      timing['max_freq'] = max(timing['max_freq'], result.timing['max_freq'])
      for key in SynthesisHandler.get_available_fpga_resources().keys():
        utilisation[key] = min(utilisation[key], result.utilisation[key])
      try:
        accuracy = min(accuracy, result.accuracy)
      except Exception as e:
        print(f"Warning: could not compare accuracy for {result.design_config}: {e}")
    
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
    accuracy = 0.0
    
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
      
    s += f"Perplexity: {self.accuracy:.2f}\n" if self.accuracy is not None else "Perplexity: N/A\n"

    return s

class SynthesisHandler:
  def __init__(self, designs_to_synthesise=None, hdl_dir="./src/attention/", synth_output_dir="synth_output", clock_period_ns=5):
    self.results = []
    self.designs_to_synthesise = designs_to_synthesise
    self.hdl_dir = hdl_dir
    self.clock_period_ns = clock_period_ns

    # Max frequency for the board, used to filter out results with invalid frequencies
    # Technically, max frequency is 500 MHz, but we use 400 MHz to be safe
    self.board_max_freq = 500 # MHz
    
    self.synth_output_dir = os.path.join(self.hdl_dir, synth_output_dir)
    
    self._time_format = "%Y%m%d_%H%M"
    
    self.powers = []
    self.resource_usages = []
    self.throughputs = []
    self.accuracies = []
    
  @staticmethod
  def get_available_fpga_resources(key=None):
    # Device: xcv80-lsva4737-2MHP-e-S
    # TODO update according to actual device
    AVAILABLE_FPGA_RESOURCES = {
      "LUTs": 1728000,
      "FFs": 3456000,
      # "CARRY8": 216000,
      # "Muxes": 864000+432000+216000,
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
      if mxfp_bits.exp_bits < 0 or mxfp_bits.mant_bits <= 0: ###########
        return True
    
    # S_q, S_kv, d_kq, d_v must powers of 2 (including 2^0 = 1)
    for param in [design.S_q, design.S_kv, design.d_kq, design.d_v]:
      if (param & (param - 1)) != 0:
        return True
      
    # d_kq and d_v must be divisible by k
    if design.d_kq % design.k != 0 or design.d_v % design.k != 0:
      return True
      
    # S_kq and S_v must be divisible by k
    if design.S_q % design.k != 0 or design.S_kv % design.k != 0:
      return True
    
    return False
  
  @staticmethod
  def run_synthesis_on_design(design, synthesis_cmd, verbose):
    if verbose:
      print(f"Results for {design!r} not found, running synthesis command: {synthesis_cmd}")
      
    start_time = time.perf_counter()
    try:
      _ = subprocess.run(synthesis_cmd, shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
    except subprocess.CalledProcessError as e:
      print(f"Synthesis failed for {design} with return code: {e.returncode}")
    except Exception as e:
      print(f"An unknown error occurred while running synthesis for {design}: {e}")
        
    end_time = time.perf_counter()
    
    if verbose:
      print(f"Synthesis for {design!r} completed in {end_time - start_time:.2f} seconds.")
    
  def run_synthesis(self, dry_run=False, verbose=False):
    if not self.designs_to_synthesise:
      print("No designs to synthesise specified.")
      return
    
    if verbose:
      print(f"Starting synthesis for {len(self.designs_to_synthesise)} designs...")
    
    jobs = []
    with ProcessPoolExecutor() as executor:
      for design_id, design in enumerate(self.designs_to_synthesise):
        # time.sleep(design_id)
        if self.check_if_design_is_invalid(design):
          if verbose:
            print(f"Skipping synthesis for {design!r} as design configuration is invalid.")
          continue
        
        if self.check_if_results_exist(design, ["_power.rpt", "_timing.rpt", "_util.rpt"]):
          if verbose:
            print(f"Skipping synthesis for {design!r} as results already exist.")
          continue
        
        run_synth_path = os.path.join(self.hdl_dir, design.get_tcl_filename())
        # synthesis_cmd = f"vivado -mode batch -source {run_synth_path} -tclargs {design.get_vivado_tclargs()}"
        synthesis_cmd = f"/mnt/applications/Xilinx/24.2/Vivado/2024.2/bin/vivado -mode batch -source {run_synth_path} -tclargs {design.get_vivado_tclargs()}"
        
        if dry_run:
          if verbose:
            print(f"Dry run mode enabled, skipping actual synthesis, cmd supposed to run:\n{synthesis_cmd}")
          continue
        
        # Submit parallel task
        future = executor.submit(self.run_synthesis_on_design, design, synthesis_cmd, verbose)
        jobs.append(future)
        # self.run_synthesis_on_design(design, synthesis_cmd, verbose=verbose)
        
      # Wait for all futures to complete
      for future in as_completed(jobs):
        try:
          future.result()
        except Exception as e:
          print(f"Synthesis subprocess failed with: {e}")
          
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
      
    timing_match = re.search(r"\n\s*([-?\d\.]+)\s+([-?\d\.]+)\s+\d+\s+\d+\s+([-?\d\.]+)\s+([-?\d\.]+)\s+\d+\s+\d+", text)
    
    wns = float(timing_match.group(1)) if timing_match else 0
    tns = float(timing_match.group(2)) if timing_match else 0
    whs = float(timing_match.group(3)) if timing_match else 0
    ths = float(timing_match.group(4)) if timing_match else 0
    
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
        "FFs": r"\|\s*Registers\s*\|\s*(\d+)",
        # "CARRY8": r"\|\s*CARRY8\s*\|\s*(\d+)",
        # "F7_Muxes": r"\|\s*F7 Muxes\s*\|\s*(\d+)",
        # "F8_Muxes": r"\|\s*F8 Muxes\s*\|\s*(\d+)",
        # "F9_Muxes": r"\|\s*F9 Muxes\s*\|\s*(\d+)",
        "BRAMs": r"\|\s*Block RAM Tile\s*\|\s*(\d+)",
        "DSPs": r"\|\s*DSP Slices\s*\|\s*(\d+)"
    }

    # total_muxes = 0
    for key, pattern in patterns.items():
        match = re.search(pattern, text)
        # if key in ["F7_Muxes", "F8_Muxes", "F9_Muxes"]:
        #     total_muxes += int(match.group(1)) if match else 0
        # else:
        #     results[key] = int(match.group(1)) if match else 0
        results[key] = int(match.group(1)) if match else 0

    # results["Muxes"] = total_muxes

    return results
  
  def _read_accuracy_report(self, file_path):
    global DEBUG_COUNTER
    try:
      with open(file_path, 'r') as file:
        text = file.read()
        
      accuracy_match = re.search(r"Perplexity:\s*(\d+\.\d+)", text)
      accuracy = float(accuracy_match.group(1))
    except FileNotFoundError:
      # print(f"Accuracy report file not found: {file_path}")
      # TODO placeholder
      accuracy = random.Random(DEBUG_COUNTER).uniform(1.0, 10.0)
      DEBUG_COUNTER += 1

    return accuracy
  
  def run_accuracy_measurement(self, dry_run=False, verbose=False):
    if not self.designs_to_synthesise:
      print("No designs to measure accuracy for specified.")
      return
    
    for design in self.designs_to_synthesise:
      if self.check_if_result_exist(design, "_accuracy.txt"):
        if verbose:
          print(f"Skipping accuracy measurement for {design!r} as accuracy report already exists.")
        continue
      
      date_time_str = datetime.now().strftime(self._time_format)
      accuracy_report_path = os.path.join(self.synth_output_dir, f"{design!r}_time_{date_time_str}_accuracy.txt")
      
      if verbose:
        print(f"Running accuracy measurement for {design!r}, saving report to {accuracy_report_path}...")
      
      if not dry_run:
        self._generate_accuracy_report(design, accuracy_report_path)
      
  
  def _generate_accuracy_report(self, design, accuracy_report_path):
    accuracy_cmd = f"CUDA_VISIBLE_DEVICES=1 python -u bert/llama_ppl.py {design.get_bert_flags()}"

    try:
        completed_process = subprocess.run(accuracy_cmd, shell=True, stdout=open(accuracy_report_path, "w"), stderr=subprocess.DEVNULL, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Accuracy measurement failed for {design} with return code: {e.returncode}")
    except Exception as e:
        print(f"An unknown error occurred while running accuracy measurement for {design}: {e}")
    
  def _process_result(self, design_str, date_time):
    file_path = os.path.join(self.synth_output_dir, f"{design_str}_time_{date_time.strftime(self._time_format)}")
    design = DesignConfig.from_str(design_str)
    
    power_report_path = f"{file_path}_power.rpt"
    timing_report_path = f"{file_path}_timing.rpt"
    utilisation_report_path = f"{file_path}_util.rpt"
    accuracy_report_path = f"{file_path}_accuracy.txt"
    
    # # Generate missing accuracy report on the fly
    # if not self.check_if_result_exist(design, "_accuracy.txt"):
    #   print(f"Accuracy report not found for {design!r}, generating on the fly...")
    #   self._generate_accuracy_report(design, accuracy_report_path)
      
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
      # print(pattern)
      # print()
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

    print (f"Found {len(matches)} synthesis results in {directory}.")
    return matches
  
  def find_and_process_results(self):  
    matches = self._find_results(self.synth_output_dir)
    for design_str, date_time in matches.items():
      self._process_result(design_str, date_time)
    
    self.designs = [r.design_config for r in self.results]
    self.powers = [r.power['total'] for r in self.results]
    self.resource_usages = [r.get_aggregated_resource_usage(use_multipliers=True) for r in self.results]
    self.LUTs = [r.get_aggregated_resource_usage(["LUTs"], use_multipliers=True) for r in self.results]
    self.FFs = [r.get_aggregated_resource_usage(["FFs"], use_multipliers=True) for r in self.results]
    self.BRAMs = [r.get_aggregated_resource_usage(["BRAMs"], use_multipliers=True) for r in self.results]
    self.DSPs = [r.get_aggregated_resource_usage(["DSPs"], use_multipliers=True) for r in self.results]
    self.throughputs = [r.timing['max_freq'] * r.get_throughput_multiplier() for r in self.results]
    self.accuracies = [r.accuracy for r in self.results]
   
  def find_pareto_optimal(self, weights):
    if not self.results:
      raise ValueError("No synthesis results available to find Pareto optimal solution.")
    
    # ideal_result = SynthesisResult.create_ideal_result(self.results)
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
      # power_diff = (result.power['total'] - ideal_result_normalised.power['total']) ** 2
      timing_diff = (result.timing['max_freq'] - ideal_result_normalised.timing['max_freq']) ** 2
      accuracy_diff = (result.accuracy - ideal_result_normalised.accuracy) ** 2

      distance = (
        timing_diff * weights['timing'] +
        resource_diff * weights['utilisation'] +
        accuracy_diff * weights['accuracy']
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


  def plot_perplexity(self, directory="./plots", plot_file_format="svg"):
    color_values = np.array([r.design_config.get_total_bits() for r in self.results])
    
    self._plot(
      x=self.resource_usages,
      y=self.accuracies,
      color_values=color_values,
      xlabel="Resource Usage (%)",
      ylabel="Perplexity",
      title=f"Perplexity vs Resource Usage",
      filename=f"perplexity_vs_resource_usage.{plot_file_format}",
      directory=directory,
      show_colorbar=False
    )
    
    self._plot(
      x=self.throughputs,
      y=self.accuracies,
      color_values=color_values,
      xlabel="Throughput (T/s)",
      ylabel="Perplexity",
      title=f"Perplexity vs Throughput",
      filename=f"perplexity_vs_throughput.{plot_file_format}",
      directory=directory,
      show_colorbar=True
    )

  def _plot(self, x, y, color_values, xlabel, ylabel, title, filename, directory, do_pareto_front=True, do_pareto_optimal=True, do_best_fit_line=False, show_colorbar=True):
    # Differentiate designs by block_size k
    marker_map = {
      "KULISCH": "o",
      "KAHAN": "^",
      "TWOSUM": "s",
      "FASTTWOSUM": "D",
      "NEUMAIER": "P",
      "KLEIN": "X",
    }
    

    cmap = matplotlib.colormaps["viridis"].resampled(color_values.max() - color_values.min() + 1)
    bounds = np.arange(color_values.min() - 0.5, color_values.max() + 1.5, 1)
    norm = matplotlib.colors.BoundaryNorm(bounds, cmap.N)

    figsize = (7, 6) if show_colorbar else (6, 6)
    fig, ax = plt.subplots(figsize=figsize)
    plotted_markers = {}

    for design, xi, yi, cval in zip(self.designs, x, y, color_values):
      accum_method = design.accum_method1.value
      label = f"{accum_method.capitalize()}"
      marker = marker_map.get(accum_method, "s")
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
    if show_colorbar is True:
      cbar = plt.colorbar(sm, ax=ax, boundaries=bounds,
                          ticks=np.arange(color_values.min(), color_values.max() + 1))
      cbar.set_label("Combined bit widths across stages", fontsize=14)
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
      pareto_points = self._pareto_front(x, y, maximize_y=False) # maximize_y is False for Perplexity minimization
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
        x_val = self.pareto_optimal.get_aggregated_resource_usage() * self.pareto_optimal.get_aggregated_resource_multiplier()
      else:
        x_val = self.pareto_optimal.timing["max_freq"]

      y_val = self.pareto_optimal.accuracy
      # if ylabel.startswith("Power"):
        # y_val = self.pareto_optimal.power["total"]
      # else:
      #   y_val = self.pareto_optimal.timing["max_freq"]

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

    if not show_colorbar:
      ax.legend().set_visible(False)
    else:
      ax.legend(black_handles, unique_labels, fontsize=14)

    fig.tight_layout()
    fig.savefig(os.path.join(directory, filename))
    
  def find_fit(self, degree=2, threshold=1e-3, combine_E_M=True, verbose=True):
    # Create a DataFrame from design parameters
    data = {
      'S': np.array([d.S_q for d in self.designs]),
      'd': np.array([d.d_kq for d in self.designs]),
    }
    if combine_E_M:
      data['(E+M)'] = np.array([d.M1_bits.exp_bits + d.M1_bits.mant_bits for d in self.designs])
    else:
      data['E'] = np.array([d.M1_bits.exp_bits for d in self.designs])
      data['M'] = np.array([d.M1_bits.mant_bits for d in self.designs])
      
    df = pd.DataFrame(data)
    y = np.array(self.LUTs)

    # Generate polynomial features (can include log or exp too)
    poly = PolynomialFeatures(degree=degree, include_bias=False)
    X_poly = poly.fit_transform(df)

    # Fit a linear regression model
    model = LinearRegression()
    model.fit(X_poly, y)

    if verbose:
      print("\nPolynomial model fit")
      if combine_E_M:
        print(f"(Using combined (E+M) feature)")
      feature_names = poly.get_feature_names_out(df.columns)
      formula = ""
      for coef, name in zip(model.coef_, feature_names):
        if coef > threshold:
          formula += f"{coef:.4f} * {name} + "
          
      print("Fitted formula (terms with coef > {:.3f}):".format(threshold))
      # print(f"\ty({', '.join(list(data.keys()))}) = {formula.rstrip(" + ")} + {model.intercept_:.2f}")
      print(f"\tR² score: {model.score(X_poly, y):.4f}\n")
    
    return model, poly
  
  def find_fit_with_gplearn(self, population_size=5000, generations=50, parsimony_coefficient=1e-3):
    # Prepare the design matrix
    X = np.array([[d.S_q, d.d_kq, d.M1_bits.exp_bits + d.M1_bits.mant_bits] for d in self.designs])
    y = np.array(self.LUTs)

    # Normalize the features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Use symbolic regression to fit a model
    model = SymbolicRegressor(
      population_size=population_size,
      generations=generations,
      function_set=['add', 'mul', 'log'],
      stopping_criteria=0.01,
      p_crossover=0.7,
      p_subtree_mutation=0.1,
      p_hoist_mutation=0.05,
      p_point_mutation=0.1,
      max_samples=1.0,
      verbose=1,
      parsimony_coefficient=parsimony_coefficient,
      random_state=124,
      n_jobs=-1,
      feature_names=['S', 'd', '(E+M)']
    )

    model.fit(X_scaled, y)

    print("\nGenetic Symbolic Regression:")
    print(model._program)
    print(gplearn_expr_to_math(model._program.__str__()))
    print(f"\nR² score: {model.score(X_scaled, y):.4f}")
    
  
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
  
  # Combined Sweep:
  # 1. INT (Logarithmic) Sweep: Widths 2-16 (E0, M=width-1)
  # 2. FP (Standard) Sweep: Widths 2-16
  
  # Define FP Configurations for each width (2-16)
  def get_fp_config(w):
      # Returns (Exp, Man) s.t. 1 + Exp + Man = w
      if w == 2: return (1, 0)
      if w == 3: return (2, 0)
      if w == 4: return (2, 1)
      if w == 5: return (2, 2)
      if w == 6: return (3, 2)
      if w == 7: return (3, 3)
      if w == 8: return (4, 3) # E4M3 (Std)
      if w == 9: return (4, 4)
      # For w >= 10, use E5 (converging to FP16)
      return (5, w - 6)

  designs_to_synthesise = []
  
  # # INT Sweep (2-16)
  # designs_to_synthesise += [
  #   DesignConfig(name, S_q, S_kv, d_kq, d_v, k, scale_width, M1_E, M1_M, M2_E, M2_M, M3_E, M3_M, accum_method_1, accum_method_2, accum_method_3, m1_dsp, m2_dsp, m3_dsp)
  #   for name in ["attention_fp"] # Reverted to standard name
  #   for S_q in [4]
  #   for S_kv in [4]
  #   for d_kq in [4]
  #   for d_v in [4]
  #   for k in [2]
  #   for scale_width in [8]
  #   for width in range(2, 17) # INT 2-16
  #   for M1_E, M1_M in [(0, width-1)]
  #   for M2_E, M2_M in [(0, width-1)]
  #   for M3_E, M3_M in [(0, width-1)]
  #   for accum_method_1 in [AccumMethod.Kulisch]
  #   for accum_method_2 in [AccumMethod.Kulisch]
  #   for accum_method_3 in [AccumMethod.Kulisch]
  #   for m1_dsp in ["auto"]
  #   for m2_dsp in ["auto"]
  #   for m3_dsp in ["auto"]
  # ]

  # # FP Sweep (2-16)
  # designs_to_synthesise += [
  #   DesignConfig(name, S_q, S_kv, d_kq, d_v, k, scale_width, M1_E, M1_M, M2_E, M2_M, M3_E, M3_M, accum_method_1, accum_method_2, accum_method_3, m1_dsp, m2_dsp, m3_dsp)
  #   for name in ["attention_fp"] # Reverted to standard name
  #   for S_q in [4]
  #   for S_kv in [4]
  #   for d_kq in [4]
  #   for d_v in [4]
  #   for k in [2]
  #   for scale_width in [8]
  #   for width in range(2, 17) # FP 2-16
  #   for exp_width, man_width in [get_fp_config(width)] 
  #   for M1_E, M1_M in [(exp_width, man_width)]
  #   for M2_E, M2_M in [(exp_width, man_width)]
  #   for M3_E, M3_M in [(exp_width, man_width)]
  #   for accum_method_1 in [AccumMethod.Kulisch]
  #   for accum_method_2 in [AccumMethod.Kulisch]
  #   for accum_method_3 in [AccumMethod.Kulisch]
  #   for m1_dsp in ["auto"]
  #   for m2_dsp in ["auto"]
  #   for m3_dsp in ["auto"]
  # ]

  # # int baseline
  # designs_to_synthesise += [
  #   DesignConfig(name, S, S, d, d, k, scale_width, M_E, M_M, M_E, M_M, M_E, M_M,accum_method_1, accum_method_2, accum_method_3, m1_dsp, m2_dsp, m3_dsp)
  #   for name in ["attention_fp"] # Reverted to standard name
  #   for S in [8]
  #   for d in [8]
  #   for k in [8]
  #   for scale_width in [8]
  #   for M_E, M_M in [(0, 2), (0, 3), (0, 4), (0, 5), (0, 6), (0, 7), (0, 8), (0, 9), (0, 10)]
  #   for accum_method_1 in [AccumMethod.Kulisch]
  #   for accum_method_2 in [AccumMethod.Kulisch]
  #   for accum_method_3 in [AccumMethod.Kulisch]
  #   for m1_dsp in ["auto"]
  #   for m2_dsp in ["auto"]
  #   for m3_dsp in ["auto"]
  # ]

  # # fp baseline (the standard)
  # designs_to_synthesise += [
  #   DesignConfig(name, S, S, d, d, k, scale_width, M_E, M_M, M_E, M_M, M_E, M_M,accum_method_1, accum_method_2, accum_method_3, m1_dsp, m2_dsp, m3_dsp)
  #   for name in ["attention_fp"] # Reverted to standard name
  #   for S in [8]
  #   for d in [8]
  #   for k in [8]
  #   for scale_width in [8]
  #   for M_E, M_M in [(5, 2), (4, 3), (3, 2), (2, 3), (2, 1)]
  #   for accum_method_1 in [AccumMethod.Kulisch]
  #   for accum_method_2 in [AccumMethod.Kulisch]
  #   for accum_method_3 in [AccumMethod.Kulisch]
  #   for m1_dsp in ["auto"]
  #   for m2_dsp in ["auto"]
  #   for m3_dsp in ["auto"]
  # ]
  
  # Analatical model: 
  designs_to_synthesise += [
    DesignConfig(name, S, S, d, d, d, scale_width, M1_E, M1_M, M1_E, M1_M, M2_E, M2_M, accum_method_1, accum_method_1, accum_method_1, m1_dsp, m1_dsp, m1_dsp)
    for name in ["mxint_softmax"]
    for S in [4, 8, 16]
    for d in [4, 8,16]
    # for k in [8]
    for scale_width in [8]
    for M1_E, M1_M in [(2, 3), (3, 3), (3, 4), (4, 4)]
    for M2_E, M2_M in [(2, 3), (3, 3), (3, 4), (4, 4)]
    for accum_method_1 in [AccumMethod.Kulisch]
    for m1_dsp in ["auto"]
  ]

  synthesis_handler = SynthesisHandler(designs_to_synthesise, synth_output_dir="synth_output_softmax")
  for i in range(1):
    synthesis_handler.run_synthesis(dry_run=args.dry, verbose=args.verbose)
    print(f"========================================================\n========================================================\n========================================================\nRUN {i}\n========================================================\n========================================================\n========================================================\n")
  # synthesis_handler.run_accuracy_measurement(dry_run=args.dry, verbose=args.verbose)

  synthesis_handler.find_and_process_results()
  # print(synthesis_handler)

  # pareto_optimal = synthesis_handler.find_pareto_optimal(weights={'timing': 1.0, 'utilisation': 1.0, 'accuracy': 1.0})
  # print(f"\nPareto Optimal Result:\n{pareto_optimal}")

  # # synthesis_handler.plot_perplexity(directory="./plots", plot_file_format="png")
  
  # synthesis_handler.find_fit(degree=2, threshold=1e-3, combine_E_M=True, verbose=args.verbose)
  # synthesis_handler.find_fit(degree=2, threshold=1e-3, combine_E_M=False, verbose=args.verbose)
  # synthesis_handler.find_fit_with_gplearn(population_size=10000, generations=10, parsimony_coefficient=0.0010)
