import re

from DSE.MXFPBits import MXFPBits
from DSE.AccumMethod import AccumMethod

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
  
  def get_quant_flags(self):
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
  
  def is_mixed_accum_ablation(self):
    baseline_k = [32]
    if not any(self._check_all_k_are(k) for k in baseline_k):
      return False
    
    e_m = [(0, 8), (5, 2), (4, 3), (3, 2), (2, 3), (2, 1)]
    if not any(self._check_all_widths_are(e, m, 5, 10) for e, m in e_m):
      return False
    
    return True
  
  def is_joint_ablation(self):
    if self.M2_bits.exp_bits == 5 and self.M2_bits.mant_bits == 10:
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
