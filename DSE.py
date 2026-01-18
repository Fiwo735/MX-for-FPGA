import os
import glob
import re
import subprocess
import time
import copy
import itertools
import random
import time
import pickle
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
from deap import base, creator, tools, algorithms

DEBUG_COUNTER = 0

LUTS_BASELINE = 11874317 # TODO
FFS_BASELINE = 2592801 # TODO

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
  Naive = "NAIVE"
  Quant = "QUANT"

class DesignConfig:
  def __init__(self, name, S_q=-1, S_kv=-1, d_kq=-1, d_v=-1, k1=-1, k2=-1, k3=-1, scale_width=-1, M1_E=-1, M1_M=-1, M2_E=-1, M2_M=-1, M3_E=-1, M3_M=-1, accum_method1=AccumMethod.Kulisch, accum_method2=AccumMethod.Kulisch, accum_method3=AccumMethod.Kulisch, m1_dsp="yes", m2_dsp="yes", m3_dsp="yes"):
    self.name = name
    
    self.S_q = S_q
    self.S_kv = S_kv
    self.d_kq = d_kq
    self.d_v = d_v
    
    self.k1 = k1
    self.k2 = k2
    self.k3 = k3
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
  
  def get_total_k(self):
    return self.k1 + self.k2 + self.k3
  
  def get_bert_flags(self):
    out = "--model_id 'meta-llama/Llama-3.2-1B' "

    if self.M1_bits.exp_bits == 0:
      out += f'--config \'k_quantizer={{"quant":"MXINTQuantizer","bit_w":{self.M1_bits.mant_bits},"group_size":{self.k1}}}\' '
    else:
      out += f'--config \'k_quantizer={{"quant":"MXFPQuantizer","man_w":{self.M1_bits.mant_bits},"exp_w":{self.M1_bits.exp_bits},"group_size":{self.k1}}}\' '

    if self.M2_bits.exp_bits == 0:
      out += f'--config \'s_quantizer={{"quant":"MXINTQuantizer","bit_w":{self.M2_bits.mant_bits},"group_size":{self.k2}}}\' '
    else:
      out += f'--config \'s_quantizer={{"quant":"MXFPQuantizer","man_w":{self.M2_bits.mant_bits},"exp_w":{self.M2_bits.exp_bits},"group_size":{self.k2}}}\' '

    if self.M3_bits.exp_bits == 0:
      out += f'--config \'v_quantizer={{"quant":"MXINTQuantizer","bit_w":{self.M3_bits.mant_bits},"group_size":{self.k3}}}\' '
    else:
      out += f'--config \'v_quantizer={{"quant":"MXFPQuantizer","man_w":{self.M3_bits.mant_bits},"exp_w":{self.M3_bits.exp_bits},"group_size":{self.k3}}}\' '
    
    out += f'--config \'sum_type_attn_s="{self.accum_method1.value}"\' '
    out += f'--config \'sum_type_smax="{self.accum_method2.value}"\' '
    out += f'--config \'sum_type_attn_o="{self.accum_method3.value}"\' '

    return out
  
  def _check_all_widths_are(self, e, m, e2=None, m2=None):
    e2 = e if e2 is None else e2
    m2 = m if m2 is None else m2
    return (
      self.M1_bits.exp_bits == e and self.M1_bits.mant_bits == m and
      self.M2_bits.exp_bits == e2 and self.M2_bits.mant_bits == m2 and
      self.M3_bits.exp_bits == e and self.M3_bits.mant_bits == m
    )
    
  def _check_all_k_are(self, k):
    return self.k1 == k and self.k2 == k and self.k3 == k
  
  def _check_all_accum_methods_are(self, method):
    return (
      self.accum_method1 == method and
      self.accum_method2 == method and
      self.accum_method3 == method
    )
    
  def _check_if_model_dims_are_baseline(self):
    return (
      self.S_q == 2048 and
      self.S_kv == 2048 and
      self.d_kq == 64 and
      self.d_v == 64
    )
    
  def is_baseline(self):
    baseline_e_m = [(0, 8), (5, 2), (4, 3), (3, 2), (2, 3), (2, 1)] #  ABLATION: BASELINE & MIXED PRECISION
    # baseline_e_m = [(5, 2), (4, 3), (3, 2), (2, 3)] # ABLATION: MIXED K
    if not any(self._check_all_widths_are(e, m, 5, 10) for e, m in baseline_e_m):
      return False
    
    baseline_k = [32]
    if not any(self._check_all_k_are(k) for k in baseline_k):
      return False
    
    baseline_accum_methods = [AccumMethod.Kulisch]
    if not any(self._check_all_accum_methods_are(method) for method in baseline_accum_methods):
      return False
    
    if not self._check_if_model_dims_are_baseline():
      return False
    
    return True
    
  def is_mixed_precision_ablation(self):
    baseline_k = [32]
    if not any(self._check_all_k_are(k) for k in baseline_k):
      return False
    
    baseline_accum_methods = [AccumMethod.Kulisch]
    if not any(self._check_all_accum_methods_are(method) for method in baseline_accum_methods):
      return False
    
    if not self._check_if_model_dims_are_baseline():
      return False
    
    if self.M1_bits.exp_bits + self.M1_bits.mant_bits > 7:
      return False
    
    if self.M2_bits.exp_bits + self.M2_bits.mant_bits > 7:
      return False
    
    if self.M3_bits.exp_bits + self.M3_bits.mant_bits > 7:
      return False
    
    return True
  
  def is_mixed_k_ablation(self):
    # baseline_e_m = [(0, 8), (5, 2), (4, 3), (3, 2), (2, 3), (2, 1)]
    baseline_e_m = [(2,3)]
    if not any(self._check_all_widths_are(e, m, 5, 10) for e, m in baseline_e_m):
      return False
    
    # Only check for certain k values
    allowed_k = [16, 32, 64]
    if self.k1 not in allowed_k:
      return False
    if self.k2 not in allowed_k:
      return False
    if self.k3 not in allowed_k:
      return False
    
    
    baseline_accum_methods = [AccumMethod.Kulisch]
    if not any(self._check_all_accum_methods_are(method) for method in baseline_accum_methods):
      return False
    
    if not self._check_if_model_dims_are_baseline():
      return False
    
    return True
  
  # designs_to_synthesise = [
  #   DesignConfig(name, S, S, d, d, k1, k2, k3, scale_width, M1_E, M1_M, M2_E, M2_M, M1_E, M1_M, accum_method_1, accum_method_1, accum_method_1, m1_dsp, m1_dsp, m1_dsp)
  #   for name in ["attention_fp"]
  #   for S in [2048]
  #   for d in [64]
  #   for k1 in [32]
  #   for k2 in [32]
  #   for k3 in [32]
  #   for scale_width in [8]
  #   for M1_E, M1_M in [(0, 8), (5, 2), (4, 3), (3, 2), (2, 3), (2, 1)]
  #   for M2_E, M2_M in [(5, 10)]
  #   for accum_method_1 in [AccumMethod.Kahan, AccumMethod.Neumaier, AccumMethod.Klein, AccumMethod.TwoSum, AccumMethod.FastTwoSum]
  #   for m1_dsp in ["auto"]
  # ]
  
  def is_mixed_accum_ablation(self):
    baseline_k = [32]
    if not any(self._check_all_k_are(k) for k in baseline_k):
      return False
    
    e_m = [(0, 8), (5, 2), (4, 3), (3, 2), (2, 3), (2, 1)]
    if not any(self._check_all_widths_are(e, m, 5, 10) for e, m in e_m):
      return False
    
    return True
    

  def __repr__(self):
    return (
      f"{self.name}_S_q_{self.S_q}_S_kv_{self.S_kv}_d_kq_{self.d_kq}_d_v_{self.d_v}_k1_{self.k1}_k2_{self.k2}_k3_{self.k3}_"
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
    s += f"  k1: {self.k1}\n"
    s += f"  k2: {self.k2}\n"
    s += f"  k3: {self.k3}\n"
    s += f"  scale_width: {self.scale_width}\n"
    s += f"  M1 bits: {self.M1_bits}\n"
    s += f"  M2 bits: {self.M2_bits}\n"
    s += f"  M3 bits: {self.M3_bits}\n"
    s += f"  Accumulation method 1: {self.accum_method1.value}\n"
    s += f"  Accumulation method 2: {self.accum_method2.value}\n"
    s += f"  Accumulation method 3: {self.accum_method3.value}\n"
    return s
    
  def get_vivado_tclargs(self):
    return f"{self.S_q} {self.S_kv} {self.d_kq} {self.d_v} {self.k1} {self.k2} {self.k3} {self.scale_width} {self.M1_bits.exp_bits} {self.M1_bits.mant_bits} {self.M2_bits.exp_bits} {self.M2_bits.mant_bits} {self.M3_bits.exp_bits} {self.M3_bits.mant_bits} {self.accum_method1.value} {self.accum_method2.value} {self.accum_method3.value} {self.m1_dsp} {self.m2_dsp} {self.m3_dsp} {self.name}"
  
  def get_tcl_filename(self):
    if self.name == "attention_fp":
      return "run_synth_fp.tcl"
    elif self.name == "matmul_fp":
      return "run_synth_matmul.tcl"
    elif self.name == "mxint_softmax":
      return "run_synth_softmax.tcl"
    
    raise ValueError(f"Unsupported design name: {self.name}")
  
  @staticmethod
  def get_old_filename_regex():
    return r"([^/]+_S_q_\d+_S_kv_\d+_d_kq_\d+_d_v_\d+_k_\d+_scale_width_\d+_M1_E_\d+_M1_M_\d+_M2_E_\d+_M2_M_\d+_M3_E_\d+_M3_M_\d+_ACCUM_METHOD_[A-Z]+_[A-Z]+_[A-Z]+_DSP_[a-zA-Z]+_[a-zA-Z]+_[a-zA-Z]+)_time_(\d+_\d+)"
  
  @staticmethod
  def get_filename_regex():
    return r"([^/]+_S_q_\d+_S_kv_\d+_d_kq_\d+_d_v_\d+_k1_\d+_k2_\d+_k3_\d+_scale_width_\d+_M1_E_\d+_M1_M_\d+_M2_E_\d+_M2_M_\d+_M3_E_\d+_M3_M_\d+_ACCUM_METHOD_[A-Z]+_[A-Z]+_[A-Z]+_DSP_[a-zA-Z]+_[a-zA-Z]+_[a-zA-Z]+)_time_(\d+_\d+)"
  
  @staticmethod
  def get_old_design_regex():
    return r"([^/]+)_S_q_(\d+)_S_kv_(\d+)_d_kq_(\d+)_d_v_(\d+)_k_(\d+)_scale_width_(\d+)_M1_E_(\d+)_M1_M_(\d+)_M2_E_(\d+)_M2_M_(\d+)_M3_E_(\d+)_M3_M_(\d+)_ACCUM_METHOD_([A-Z]+)_([A-Z]+)_([A-Z]+)_DSP_([a-zA-Z]+)_([a-zA-Z]+)_([a-zA-Z]+)"
  
  @staticmethod
  def get_design_regex():
    return r"([^/]+)_S_q_(\d+)_S_kv_(\d+)_d_kq_(\d+)_d_v_(\d+)_k1_(\d+)_k2_(\d+)_k3_(\d+)_scale_width_(\d+)_M1_E_(\d+)_M1_M_(\d+)_M2_E_(\d+)_M2_M_(\d+)_M3_E_(\d+)_M3_M_(\d+)_ACCUM_METHOD_([A-Z]+)_([A-Z]+)_([A-Z]+)_DSP_([a-zA-Z]+)_([a-zA-Z]+)_([a-zA-Z]+)"
  
  
  @classmethod
  def from_str(cls, design_str, use_new_filename=False):
    details = re.search(
      cls.get_design_regex() if use_new_filename else cls.get_old_design_regex(),
      design_str
    ) 
    # print(cls.get_design_regex())
    # print(design_str)
    if not details:
      raise ValueError(f"Design string {design_str} does not match expected pattern.")
    
    if use_new_filename:
      name = details.group(1)
      S_q = int(details.group(2))
      S_kv = int(details.group(3))
      d_kq = int(details.group(4))
      d_v = int(details.group(5))
      k1 = int(details.group(6))
      k2 = int(details.group(7))
      k3 = int(details.group(8))
      scale_width = int(details.group(9))
      M1_E = int(details.group(10))
      M1_M = int(details.group(11))
      M2_E = int(details.group(12))
      M2_M = int(details.group(13))
      M3_E = int(details.group(14))
      M3_M = int(details.group(15))
      accum_method1 = AccumMethod(details.group(16))
      accum_method2 = AccumMethod(details.group(17))
      accum_method3 = AccumMethod(details.group(18))
      m1_dsp = details.group(19)
      m2_dsp = details.group(20)
      m3_dsp = details.group(21)
    
    else:
      name = details.group(1)
      S_q = int(details.group(2))
      S_kv = int(details.group(3))
      d_kq = int(details.group(4))
      d_v = int(details.group(5))
      k1 = int(details.group(6))
      k2 = int(details.group(6))
      k3 = int(details.group(6))
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
    
    return cls(name=name, S_q=S_q, S_kv=S_kv, d_kq=d_kq, d_v=d_v, k1=k1, k2=k2, k3=k3, scale_width=scale_width, M1_E=M1_E, M1_M=M1_M, M2_E=M2_E, M2_M=M2_M, M3_E=M3_E, M3_M=M3_M, accum_method1=accum_method1, accum_method2=accum_method2, accum_method3=accum_method3, m1_dsp=m1_dsp, m2_dsp=m2_dsp, m3_dsp=m3_dsp)

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
      
    utilisation_sum = 0.0
    
    for key in keys:
      utilisation_sum = self.utilisation[key] / SynthesisHandler.get_available_fpga_resources(key)
      
    return utilisation_sum / len(keys)
  
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
  def __init__(self, designs_to_synthesise=None, hdl_dir="./src/attention/", synth_output_dir="synth_output", clock_period_ns=5, max_workers=4):
    self.results = []
    self.designs_to_synthesise = designs_to_synthesise
    self.hdl_dir = hdl_dir
    self.clock_period_ns = clock_period_ns
    self.max_workers = max_workers

    # Max frequency for the board, used to filter out results with invalid frequencies
    # TODO placeholder
    self.board_max_freq = 1200 # MHz 
    
    self.synth_output_dir = os.path.join(self.hdl_dir, synth_output_dir)
    
    self._time_format = "%Y%m%d_%H%M"
    
    self.powers = []
    self.resource_usages = []
    self.accuracies = []
    
    self.pickle_dir = "./synthesis_fits"
    
  @staticmethod
  def get_available_fpga_resources(key=None):
    # Device: xcv80-lsva4737-2MHP-e-S
    AVAILABLE_FPGA_RESOURCES = {
      "LUTs": 2574208,
      "FFs": 5148416,
      # "CARRY8": 216000,
      # "Muxes": 864000+432000+216000,
      "BRAMs": 3741,
      "DSPs": 10848,
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
    with ProcessPoolExecutor(max_workers=self.max_workers) as executor:
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
    max_freq = 1e3 / (self.clock_period_ns - wns)

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

    # if results["DSPs"] > 0:
    #   print(results["DSPs"])
    #   raise(Exception("Debug stop"))
    
    return results
  
  def _read_accuracy_report(self, file_path, verbose):
    global DEBUG_COUNTER
    try:
      with open(file_path, 'r') as file:
        text = file.read()
        
      accuracy_match = re.search(r"Perplexity:\s*(\d+\.\d+)", text)
      
      if accuracy_match is None:
        if verbose:
          print(f"ERROR: Could not find accuracy in report file: {file_path}")
        accuracy = -1.0
      else:
        accuracy = float(accuracy_match.group(1))
        
    except FileNotFoundError:
      if verbose:
        print(f"ERROR: Accuracy report file not found: {file_path}")
      accuracy = -1.0

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
    print(accuracy_cmd)

    try:
        completed_process = subprocess.run(accuracy_cmd, shell=True, stdout=open(accuracy_report_path, "w"), stderr=subprocess.DEVNULL, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Accuracy measurement failed for {design} with return code: {e.returncode}")
    except Exception as e:
        print(f"An unknown error occurred while running accuracy measurement for {design}: {e}")
    
  def _process_result(self, design_str, date_time, predict_resources=False, ablation_check=False, use_new_filename=False, verbose=False):
    file_path = os.path.join(self.synth_output_dir, f"{design_str}_time_{date_time.strftime(self._time_format)}")
    design = DesignConfig.from_str(design_str, use_new_filename=use_new_filename)
    
    power_report_path = f"{file_path}_power.rpt"
    timing_report_path = f"{file_path}_timing.rpt"
    utilisation_report_path = f"{file_path}_util.rpt"
    accuracy_report_path = f"{file_path}_accuracy.txt"
    
    accuracy = self._read_accuracy_report(accuracy_report_path, verbose=verbose)
    
    if predict_resources:
      dynamic_power, static_power = -1, -1
      no_timing_violation, max_freq = None, 1
      utilisation = {
        "LUTs": predict_synthesis_results(self.pickle_dir, "LUTs", design, normalise_S_q=True),
        "FFs": predict_synthesis_results(self.pickle_dir, "FFs", design, normalise_S_q=True),
        "BRAMs": -1,
        "DSPs": -1,
      }
    else:
      try:
        dynamic_power, static_power = self._read_power_report(power_report_path)
        no_timing_violation, max_freq = self._read_timing_report(timing_report_path)
        utilisation = self._read_utilisation_report(utilisation_report_path)
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
    
    if not predict_resources:
      # Only include results that have valid max frequency
      if not (max_freq > 0 and max_freq < self.board_max_freq):
        print(f"WARNING: Skipping result for {result} due to invalid max frequency: {max_freq:.2f} MHz.")
        return
      
    if ablation_check:
      # BASELINE
      # if (not design.is_baseline()):
      #   if verbose:
      #     print(f"Skipping result for {design} as it does not meet ablation check criteria.")
      #   return
      
      # ABLATION: MIXED PRECISION
      # if (not design.is_baseline()) and (not design.is_mixed_precision_ablation()):
      #   if verbose:
      #     print(f"Skipping result for {design} as it does not meet ablation check criteria.")
      #   return
      
      # ABLATION: MIXED K
      # if (not design.is_baseline()) and (not design.is_mixed_k_ablation()):
      #   if verbose:
      #     print(f"Skipping result for {design} as it does not meet ablation check criteria.")
      #   return
      
      # ABLATION: MIXED ACCUM
      if (not design.is_baseline()) and (not design.is_mixed_accum_ablation()):
        if verbose:
          print(f"Skipping result for {design} as it does not meet ablation check criteria.")
        return

    self.results.append(result)
      
  def _find_results(self, directory, report_filter=None, verbose=False):
    matches = {}
    
    if report_filter is not None:
      if report_filter == "accuracy":
        file_ext = "*.txt"
        pattern = re.compile(DesignConfig.get_filename_regex())
      else:
        raise ValueError(f"Unsupported report_filter: {report_filter}")
    else:
      file_ext = "*.rpt"
      pattern = re.compile(DesignConfig.get_old_filename_regex())
    
    for file_path in glob.glob(os.path.join(directory, file_ext)):
      filename = os.path.basename(file_path)
      # print(f"\nExtracted filename: {filename}")
      
      # Match the filename against the regex
      m = pattern.match(filename)
      # print(f"pattern: {pattern}\n")
      
      if not m:
        print(f"WARNING: Filename {filename} does not match expected pattern, skipping.")
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
  
  def find_and_process_results(self, result_dir=None, report_filter=None, predict_resources=False, ablation_check=False, verbose=False):  
    matches = self._find_results(self.synth_output_dir if result_dir is None else result_dir, report_filter=report_filter, verbose=verbose)
    for design_str, date_time in matches.items():
      self._process_result(design_str, date_time, predict_resources=predict_resources, ablation_check=ablation_check, use_new_filename=(report_filter == "accuracy"), verbose=verbose)
    
    if ablation_check:
      print(f"Ablation check enabled, total valid results found: {len(self.results)}")
      # for r in self.results:
      #   print(r.design_config)
    
    self.designs = [r.design_config for r in self.results]
    self.powers = [r.power['total'] for r in self.results]
    self.resource_usages = [r.get_aggregated_resource_usage() for r in self.results]
    # self.LUTs = [r.get_aggregated_resource_usage(["LUTs"]) for r in self.results]
    # self.FFs = [r.get_aggregated_resource_usage(["FFs"]) for r in self.results]
    # self.BRAMs = [r.get_aggregated_resource_usage(["BRAMs"]) for r in self.results]
    # self.DSPs = [r.get_aggregated_resource_usage(["DSPs"]) for r in self.results]
    self.LUTs = [r.utilisation["LUTs"] for r in self.results]
    self.FFs = [r.utilisation["FFs"] for r in self.results]
    self.BRAMs = [r.utilisation["BRAMs"] for r in self.results]
    self.DSPs = [r.utilisation["DSPs"] for r in self.results]
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
      # actual_res_usage = result.get_aggregated_resource_usage()
      # ideal_res_usage = ideal_result_normalised.get_aggregated_resource_usage()

      # resource_diff = (actual_res_usage - ideal_res_usage) ** 2
      # # power_diff = (result.power['total'] - ideal_result_normalised.power['total']) ** 2
      # timing_diff = (result.timing['max_freq'] - ideal_result_normalised.timing['max_freq']) ** 2
      
      actual_LUTs = result.utilisation["LUTs"]
      ideal_LUTs = ideal_result_normalised.utilisation["LUTs"]
      LUTs_diff = (actual_LUTs - ideal_LUTs) ** 2
      actual_FFs = result.utilisation["FFs"]
      ideal_FFs = ideal_result_normalised.utilisation["FFs"]
      FFs_diff = (actual_FFs - ideal_FFs) ** 2
      accuracy_diff = (result.accuracy - ideal_result_normalised.accuracy) ** 2

      distance = (
        LUTs_diff * weights['LUTs'] +
        FFs_diff * weights['FFs'] +
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

  def plot_perplexity(self, directory="./plots", filename_suffix="", plot_file_format="svg"):
    # color_values = np.array([r.design_config.get_total_bits() for r in self.results]) # BASELINE and ABLATION: MIXED PRECISION
    color_values = np.array([r.design_config.get_total_k() for r in self.results]) # ABLATION: MIXED K
    # color_values = np.array([0, 1, 2, 3, 4, 5]) # ABLATION: MIXED ACCUM

    LUTs_mults = np.array(self.LUTs) / LUTS_BASELINE
    FFs_mults = np.array(self.FFs) / FFS_BASELINE
    
    # Create a single figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6), sharey=True, gridspec_kw={'width_ratios': [8, 10]})
    
    _, _, sm1 = self._plot(
      fig=fig,
      ax=ax1,
      x=LUTs_mults,
      y=self.accuracies,
      color_values=color_values,
      xlabel="LUTs (×baseline)",
      ylabel="Perplexity",
      title=f"Perplexity vs LUTs",
      resource="LUTs",
      filename=f"perplexity_vs_LUTs.{plot_file_format}",
      directory=directory,
      show_colorbar=False
    )
    
    _, _, sm1 = self._plot(
      fig=fig,
      ax=ax2,
      x=FFs_mults,
      y=self.accuracies,
      color_values=color_values,
      xlabel="FFs (×baseline)",
      ylabel="Perplexity",
      title=f"Perplexity vs FFs",
      resource="FFs",
      filename=f"perplexity_vs_FFs.{plot_file_format}",
      directory=directory,
      show_colorbar=True
    )
    
    # Save the combined figure
    fig.tight_layout()
    fig.savefig(os.path.join(directory, f"perplexity_combined_{filename_suffix}.{plot_file_format}"))

  def _plot(self, fig, ax, x, y, color_values, xlabel, ylabel, title, resource, filename, directory,
          do_pareto_front=True, do_pareto_optimal=True, do_best_fit_line=False,
          show_colorbar=True):
    
    # Differentiate designs by block_size k
    # marker_map = {
    #   "KULISCH": "o",
    #   "KAHAN": "^",
    #   "TWOSUM": "s",
    #   "FASTTWOSUM": "D",
    #   "NEUMAIER": "P",
    #   "KLEIN": "X",
    #   "NAIVE": "v",
    # }
    
    marker_map = {
      True: "o",   # baseline
      False: "s",  # new
    }
    
    # ABLATION: BASELINE & MIXED PRECISION
    # cmap = matplotlib.colormaps["viridis"].resampled(color_values.max() - color_values.min() + 1)
    # bounds = np.arange(color_values.min() - 0.5, color_values.max() + 1.5, 1)
    # norm = matplotlib.colors.BoundaryNorm(bounds, cmap.N)
    
    # ABLATION: MIXED K
    step = 16
    vmin = int(np.floor(color_values.min() / step) * step)
    vmax = int(np.ceil(color_values.max() / step) * step)
    ticks = np.arange(vmin, vmax + 1, step)
    bounds = np.concatenate(([ticks[0] - step / 2], (ticks[:-1] + ticks[1:]) / 2, [ticks[-1] + step / 2]))
    norm = matplotlib.colors.BoundaryNorm(bounds, len(ticks))
    cmap = matplotlib.colormaps["viridis"].resampled(len(ticks))
    
    # ABLATION: MIXED ACCUM

    


    # figsize = (7, 6) if show_colorbar else (6, 6)
    # fig, ax = plt.subplots(figsize=figsize)
    plotted_markers = {}

    for design, xi, yi, cval in zip(self.designs, x, y, color_values):
      
      # accum_method = design.accum_method1.value
      # label = f"{accum_method.capitalize()}"
      # marker = marker_map.get(accum_method, "s")
      
      # other_label = "Mixed precision" # ABLATION: MIXED PRECISION
      # other_label = "Mixed block size"        # ABLATION: MIXED K
      other_label = "Mixed accumulation method"  # ABLATION: MIXED ACCUM
      
      label = "Baseline" if design.is_baseline() else other_label
      marker = marker_map[design.is_baseline()]
      
      ax.scatter(
        xi, yi,
        c=[cmap(norm(cval))],
        alpha=1.0,
        s=120,
        marker=marker,
        label=label
      )
      plotted_markers[label] = marker

    ax.set_title(title, fontsize=20)
    ax.set_xlabel(xlabel, fontsize=18)
    ax.set_ylabel(ylabel, fontsize=18)
    
    # ax.set_ylim(bottom=9, top=18) # BASELINE
    # ax.set_ylim(bottom=9, top=35) # ABLATION: MIXED PRECISION ONLY
    # ax.set_ylim(bottom=9.75, top=11) # ABLATION: MIXED K
    # ax.set_ylim(bottom=9.75, top=15) # ABLATION: MIXED ACCUM
    
    
    ax.tick_params(axis='x', labelsize=16)
    ax.tick_params(axis='y', labelsize=16)
    ax.grid(True)
    
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    if show_colorbar is True:
      # cbar = plt.colorbar(sm, ax=ax, boundaries=bounds, ticks=np.arange(color_values.min(), color_values.max() + 1)) # BASELINE and ABLATION: MIXED PRECISION
      cbar = plt.colorbar(sm, ax=ax, boundaries=bounds, ticks=ticks) # ABLATION: MIXED K & ACCUM
      # cbar.set_label("Combined bit widths across operators", fontsize=18) # BASELINE and ABLATION: MIXED PRECISION
      # cbar.set_label("Combined block sizes across operators", fontsize=18) # ABLATION: MIXED K
      cbar.set_label("Accumulation method across operators", fontsize=18) # ABLATION: MIXED ACCUM
      cbar.ax.tick_params(labelsize=16)

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
      baseline = LUTS_BASELINE if resource == "LUTs" else FFS_BASELINE
      x_val = self.pareto_optimal.utilisation[resource] / baseline
      # if xlabel.startswith("Resource"):
      #   x_val = self.pareto_optimal.get_aggregated_resource_usage()
      # else:
      #   x_val = self.pareto_optimal.timing["max_freq"]

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
    # fig.savefig(os.path.join(directory, filename))
    return fig, ax, sm
    
  def find_fit(self, y_type, data, degree=2, threshold=1e-3, verbose=True, pickle_suffix=""):
    # # Create a DataFrame from design parameters
    df = pd.DataFrame(data)
    if y_type == "LUTs":
      y = np.array(self.LUTs)
    elif y_type == "FFs":
      y = np.array(self.FFs)
    else:
      raise ValueError(f"Unknown y_type: {y_type}")

    # Generate polynomial features
    poly = PolynomialFeatures(degree=degree, include_bias=False)
    X_poly = poly.fit_transform(df)

    # Fit a linear regression model
    model = LinearRegression()
    model.fit(X_poly, y)

    if verbose:
      print(f"\nPolynomial model fit for {y_type}")
      feature_names = poly.get_feature_names_out(df.columns)
      formula = ""
      for coef, name in zip(model.coef_, feature_names):
        # print(f"  Coefficient for {name}: {coef:.10f}")
        if coef > threshold:
          formula += f"{coef:.10f} * {name} + "
          
      print(f"Fitted formula (terms with coef > {threshold:.3f}):")
      # print(f"\ty({', '.join(list(data.keys()))}) = {formula.rstrip(" + ")} + {model.intercept_:.2f}")
      print(f"\tR² score: {model.score(X_poly, y):.4f}\n")
    
    with open(f"{self.pickle_dir}/fit_model_{y_type}_{pickle_suffix}.pkl", "wb") as f:
      pickle.dump({
          "model": model,
          "poly": poly,
          "feature_names": df.columns.tolist()
      }, f)
  
  def find_fit_with_gplearn(self, y_type, X, population_size=5000, generations=50, parsimony_coefficient=1e-3):
    # Prepare the design matrix
    if y_type == "LUTs":
      y = np.array(self.LUTs)
    elif y_type == "FFs":
      y = np.array(self.FFs)
    else:
      raise ValueError(f"Unknown y_type: {y_type}")

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
    
    
  def genetic_perplexity_search(self):
    def run(k1, k2, k3, e1, e2, e3, m1, m2, m3, accum_method):
      # Simulate delay
      time.sleep(0.1)  # in real use, it's ~180 seconds
      return random.uniform(0, 1)  # Replace with real result
  
    # 1. Create fitness and individual
    creator.create("FitnessMin", base.Fitness, weights=(-1.0,))  # Minimise
    creator.create("Individual", list, fitness=creator.FitnessMin)

    # 2. Initialize toolbox
    toolbox = base.Toolbox()

    # Discrete values
    K_RANGE = [1, 2, 4, 8]
    E_RANGE = [2, 3, 4]
    M_RANGE = [2, 3, 4]
    ACCUM_METHODS = ['kahan', 'neumaier', 'kulisch', 'klein', 'tree', 'vanilla']

    # Register attributes
    toolbox.register("k_val", lambda: random.choice(K_RANGE))
    toolbox.register("e_val", lambda: random.choice(E_RANGE))
    toolbox.register("m_val", lambda: random.choice(M_RANGE))
    toolbox.register("accum", lambda: random.choice(range(len(ACCUM_METHODS))))  # use int index

    # One individual = [k1, k2, k3, e1, e2, e3, m1, m2, m3, accum_method_index]
    toolbox.register("individual", tools.initCycle, creator.Individual,
                    (toolbox.k_val, toolbox.k_val, toolbox.k_val,
                      toolbox.e_val, toolbox.e_val, toolbox.e_val,
                      toolbox.m_val, toolbox.m_val, toolbox.m_val,
                      toolbox.accum),
                    n=1)

    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    # Evaluation
    def evaluate(individual):
        k1, k2, k3, e1, e2, e3, m1, m2, m3, accum_idx = individual
        accum_method = ACCUM_METHODS[accum_idx]
        result = run(k1, k2, k3, e1, e2, e3, m1, m2, m3, accum_method)
        return (result,)  # tuple!

    toolbox.register("evaluate", evaluate)
    toolbox.register("mate", tools.cxTwoPoint)
    toolbox.register("mutate", tools.mutUniformInt,
                    low=[min(K_RANGE)]*3 + [min(E_RANGE)]*3 + [min(M_RANGE)]*3 + [0],
                    up=[max(K_RANGE)]*3 + [max(E_RANGE)]*3 + [max(M_RANGE)]*3 + [len(ACCUM_METHODS)-1],
                    indpb=0.2)
    toolbox.register("select", tools.selTournament, tournsize=3) 
    
    # 3. Run genetic algorithm
    pop_size = 50
    ngen = 20

    pop = toolbox.population(n=pop_size)
    hof = tools.HallOfFame(1)

    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", lambda fits: sum(fits)/len(fits))
    stats.register("min", min)

    algorithms.eaSimple(pop, toolbox, cxpb=0.5, mutpb=0.2,
                        ngen=ngen, stats=stats, halloffame=hof, verbose=True)

    print("Best individual:", hof[0])
    print("Best fitness:", hof[0].fitness.values[0])  
  
  def __str__(self):
    spacer = "="*60 + "\n"
    return (
      f"\t\t\t{len(self.results)} Synthesis Results:\n" +
      spacer + ("\n" + spacer).join([f"{result!s}" for result in self.results]) + spacer
    )
 
def predict_synthesis_results(pickle_dir, y_type, dc, normalise_S_q=False):
    def load_pickled_model(path):
      with open(pickle_dir + "/" + path, "rb") as f:
        saved = pickle.load(f)
        return saved["model"], saved["poly"], saved["feature_names"]
      
    def predict(x, poly, model, feature_names):
      x_df = pd.DataFrame(x, columns=feature_names)
      x_poly = poly.transform(x_df)
      y_pred = model.predict(x_poly)
      return y_pred
      
    # Load models
    model_matmul, poly_matmul, feature_names_matmul = load_pickled_model(f"fit_model_{y_type}_matmul.pkl")
    model_softmax, poly_softmax, feature_names_softmax = load_pickled_model(f"fit_model_{y_type}_softmax.pkl")
    
    # Normalisation scale
    S_q_div_value = dc.S_q if normalise_S_q else 1.0
    k_learned_as = 64  # During model training, k was fixed at 64
    k1_div = (dc.k1 / k_learned_as)**2 if y_type == "LUTs" else 1.0
    k2_div = 1.0 
    k3_div = (dc.k3 / k_learned_as)**2 if y_type == "LUTs" else 1.0
    
    # matmul: if k is increased by 2 then FFs are the same but LUTs decrease by 2 while perplexity is worse (grows)
    
    # Matmul 1 => y(S_q, d_kq, S_kv, (E1+M1))
    # x_matmul1 = np.array([[dc.S_q, dc.d_kq, dc.S_kv, dc.M1_bits.exp_bits + dc.M1_bits.mant_bits]])
    x_matmul1 = np.array([[dc.S_q, dc.d_kq, dc.M1_bits.exp_bits + dc.M1_bits.mant_bits]])
    # x_matmul1 = np.array([[dc.d_kq, dc.S_kv, dc.M1_bits.exp_bits + dc.M1_bits.mant_bits]])
    y_matmul1 = (predict(x_matmul1, poly_matmul, model_matmul, feature_names_matmul)[0] / S_q_div_value) / k1_div
    # print(f"Matmul1 prediction: {y_matmul1}")
    
    # Softmax => y(k2, (E2+M2), (E3+M3))
    x_softmax = np.array([[dc.k2, dc.M2_bits.exp_bits + dc.M2_bits.mant_bits, dc.M3_bits.exp_bits + dc.M3_bits.mant_bits]])
    y_softmax = predict(x_softmax, poly_softmax, model_softmax, feature_names_softmax)[0]
    # print(f"Softmax prediction: {y_softmax}")
    
    # Matmul 2 => y(S_q, S_kv, d_v, (E3+M3))
    # x_matmul2 = np.array([[dc.S_q, dc.S_kv, dc.d_v, dc.M3_bits.exp_bits + dc.M3_bits.mant_bits]])
    x_matmul2 = np.array([[dc.S_q, dc.S_kv, dc.M3_bits.exp_bits + dc.M3_bits.mant_bits]])
    # x_matmul2 = np.array([[dc.S_kv, dc.d_v, dc.M3_bits.exp_bits + dc.M3_bits.mant_bits]])
    y_matmul2 = (predict(x_matmul2, poly_matmul, model_matmul, feature_names_matmul)[0] / S_q_div_value) / k3_div
    # print(f"Matmul2 prediction: {y_matmul2}")
    
    if y_type in ["LUTs", "FFs"]:
      softmax_parallelism = (dc.S_q * dc.S_kv // dc.k2 / S_q_div_value) / k2_div
      prediction = y_matmul1 + softmax_parallelism * y_softmax + y_matmul2
    else:
      raise ValueError(f"Unknown y_type: {y_type}")
    
    return prediction
    
def calibrate_analytical_models(verbose):
  # Analatical model: MATMUL 
  designs_to_synthesise = [
    DesignConfig(name, S, S, d, d, d, d, d, scale_width, M_E, M_M, M_E, M_M, M_E, M_M, accum_method_1, accum_method_1, accum_method_1, m1_dsp, m1_dsp, m1_dsp)
    for name in ["matmul_fp"]
    for S in [2, 4, 8, 16]
    for d in [2, 4, 8, 16]
    # for k in [8]
    for scale_width in [8]
    for M_E, M_M in [(1, 1), (1, 2), (2, 2), (2, 3), (3, 3), (3, 4), (4, 4)]
    for accum_method_1 in [AccumMethod.Kulisch]
    for m1_dsp in ["auto"]
  ]

  synthesis_handler = SynthesisHandler(designs_to_synthesise, synth_output_dir="synth_output_matmul")
  synthesis_handler.find_and_process_results(verbose=verbose)
  
  # print([d.S_q for d in synthesis_handler.designs])
  
  matmul_fit_data = {
    'S':     np.array([d.S_q for d in synthesis_handler.designs]),
    'd':     np.array([d.d_kq for d in synthesis_handler.designs]),
    '(E+M)': np.array([d.M1_bits.exp_bits + d.M1_bits.mant_bits for d in synthesis_handler.designs])
  }

  synthesis_handler.find_fit("LUTs", matmul_fit_data, degree=2, threshold=0, verbose=True, pickle_suffix="matmul")
  synthesis_handler.find_fit("FFs", matmul_fit_data, degree=2, threshold=0, verbose=True, pickle_suffix="matmul")
  
  # matmul_fit_data_gplearn = np.array([[d.S_q, d.d_kq, d.M1_bits.exp_bits + d.M1_bits.mant_bits] for d in synthesis_handler.designs])
  
  # synthesis_handler.find_fit_with_gplearn("LUTs", matmul_fit_data_gplearn,       population_size=5000, generations=20, parsimony_coefficient=0.0001)
  # synthesis_handler.find_fit_with_gplearn("FFs", matmul_fit_data_gplearn,        population_size=5000, generations=20, parsimony_coefficient=0.0001)
  
  # Analatical model: SOFTMAX 
  designs_to_synthesise = [
    DesignConfig(name, S, S, d, d, d, d, d, scale_width, M1_E, M1_M, M1_E, M1_M, M2_E, M2_M, accum_method_1, accum_method_1, accum_method_1, m1_dsp, m1_dsp, m1_dsp)
    for name in ["mxint_softmax"]
    for S in [4, 8, 16]
    for d in [4, 8, 16]
    # for k in [8]
    for scale_width in [8]
    for M1_E, M1_M in [(1, 1), (1, 2), (2, 2), (2, 3), (3, 3), (3, 4), (4, 4)]
    for M2_E, M2_M in [(1, 1), (1, 2), (2, 2), (2, 3), (3, 3), (3, 4), (4, 4)]
    for accum_method_1 in [AccumMethod.Kulisch]
    for m1_dsp in ["auto"]
  ]
  
  synthesis_handler = SynthesisHandler(designs_to_synthesise, synth_output_dir="synth_output_softmax")
  synthesis_handler.find_and_process_results(verbose=verbose)

  softmax_fit_data = {
    'k':     np.array([d.k2 for d in synthesis_handler.designs]),
    '(E2+M2)': np.array([d.M2_bits.exp_bits + d.M2_bits.mant_bits for d in synthesis_handler.designs]),
    '(E3+M3)': np.array([d.M3_bits.exp_bits + d.M3_bits.mant_bits for d in synthesis_handler.designs])
  }
  
  synthesis_handler.find_fit("LUTs", softmax_fit_data, degree=3, threshold=0, verbose=True, pickle_suffix="softmax")
  synthesis_handler.find_fit("FFs", softmax_fit_data, degree=2, threshold=0, verbose=True, pickle_suffix="softmax")
    
  # softmax_fit_data_gplearn = np.array([[d.k2, d.M2_bits.exp_bits + d.M2_bits.mant_bits, d.M3_bits.exp_bits + d.M3_bits.mant_bits] for d in synthesis_handler.designs])
    
  # synthesis_handler.find_fit_with_gplearn("LUTs", softmax_fit_data_gplearn,       population_size=5000, generations=20, parsimony_coefficient=0.0001)
  # synthesis_handler.find_fit_with_gplearn("FFs", softmax_fit_data_gplearn,        population_size=5000, generations=20, parsimony_coefficient=0.0001)
  # synthesis_handler.genetic_perplexity_search()
  
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
    
if __name__ == "__main__":
  parser = ArgumentParser(description='Run DSE for attention module synthesis')
  parser.add_argument('--dry', action='store_true', help='Dry run, do not run synthesis')
  parser.add_argument('--verbose', action='store_true', help='Enable verbose output')
  parser.add_argument('--max-workers', type=int, default=4, help='Maximum number of parallel synthesis processes')
  args = parser.parse_args()
  
  calibrate_analytical_models(args.verbose)

  # Prediction Example
  design_to_predict = DesignConfig(
    name= "attention_fp",
    S_q=4, S_kv=4,
    d_kq=4, d_v=4,
    k1=4, k2=4, k3=4,
    scale_width=8,
    M1_E=0, M1_M=2,
    M2_E=0, M2_M=2,
    M3_E=4, M3_M=2,
    accum_method1=AccumMethod.Kulisch,
    accum_method2=AccumMethod.Kahan,
    accum_method3=AccumMethod.Kahan,
  )
  predicted_luts = predict_synthesis_results("synthesis_fits", "LUTs", design_to_predict)
  actual_luts =  9110
  print(f"\nPredicted LUTs: {predicted_luts}, Actual LUTs: {actual_luts}")

  predicted_ffs = predict_synthesis_results("synthesis_fits", "FFs", design_to_predict)
  actual_ffs =  9125
  print(f"Predicted FFs: {predicted_ffs}, Actual FFs: {actual_ffs}")
  
  
  
  
  synthesis_handler = SynthesisHandler([], synth_output_dir="synth_output")
  synthesis_handler.find_and_process_results(report_filter="accuracy", predict_resources=True, ablation_check=True, verbose=args.verbose)
  
  # print(synthesis_handler)
  
  pareto_point = synthesis_handler.find_pareto_optimal(weights={'LUTs': 1.0, 'FFs': 1.0, 'accuracy': 1.0})
  print(f"Pareto Optimal Design:\n{pareto_point}")
  
  synthesis_handler.plot_perplexity(directory="./plots", filename_suffix="mixed_accum", plot_file_format="png")
  
  

  # # Validation
  # designs_to_synthesise = [
  #   DesignConfig(name, S, S, d, d, d, d, d, scale_width, M_E, M_M, M_E, M_M, M_E, M_M, accum_method_1, accum_method_1, accum_method_1, m1_dsp, m1_dsp, m1_dsp)
  #   for name in ["attention_fp"]
  #   for S in [8]
  #   for d in [8]
  #   # for k in [8]
  #   for scale_width in [8]
  #   for M_E, M_M in [(0, 2), (0, 3), (0, 4), (0, 5), (0, 6), (0, 7), (0, 8), (0, 9), (0, 10), (2, 1), (2, 3), (3, 2), (4, 3), (5, 2)]
  #   for accum_method_1 in [AccumMethod.Kulisch]
  #   for m1_dsp in ["auto"]
  # ]
  
  # synthesis_handler = SynthesisHandler(designs_to_synthesise, synth_output_dir="synth_output")
  # synthesis_handler.find_and_process_results()
  
  # LUTs_diffs = []
  # LUTs_percent_diffs = []
  # FFs_diffs = []
  # FFs_percent_diffs = []
  # for result in synthesis_handler.results:
  #   predicted_luts = predict_synthesis_results("synthesis_fits", "LUTs", result.design_config)
  #   predicted_ffs = predict_synthesis_results("synthesis_fits", "FFs", result.design_config)
    
  #   LUTs_diffs.append((predicted_luts - result.utilisation['LUTs']) ** 2)
  #   FFs_diffs.append((predicted_ffs - result.utilisation['FFs']) ** 2)
    
  #   LUTs_percent_diffs.append(abs(predicted_luts - result.utilisation['LUTs']) / result.utilisation['LUTs'] * 100)
  #   FFs_percent_diffs.append(abs(predicted_ffs - result.utilisation['FFs']) / result.utilisation['FFs'] * 100)
    
  #   # print(f"Design: {result.design_config}")
  #   # print(f"  Actual LUTs: {result.utilisation['LUTs']}, Predicted LUTs: {predicted_luts:.2f}")
  #   # print(f"  Actual FFs: {result.utilisation['FFs']}, Predicted FFs: {predicted_ffs:.2f}")
    
  # rmse_luts = (sum(LUTs_diffs) / len(LUTs_diffs)) ** 0.5
  # rmse_ffs = (sum(FFs_diffs) / len(FFs_diffs)) ** 0.5
  # print(f"\nLUTs Prediction MSE: {rmse_luts:.4f}")
  # print(f"FFs Prediction MSE: {rmse_ffs:.4f}")
  
  # mean_percent_luts = sum(LUTs_percent_diffs) / len(LUTs_percent_diffs)
  # mean_percent_ffs = sum(FFs_percent_diffs) / len(FFs_percent_diffs)
  # print(f"LUTs Mean Percentage Error: {mean_percent_luts:.2f}%")
  # print(f"FFs Mean Percentage Error: {mean_percent_ffs:.2f}%")
  