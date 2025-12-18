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
from datetime import datetime

class DesignConfig:
  def __init__(self, name, S_q=-1, S_kv=-1, d_kq=-1, d_v=-1, k=-1, bit_width=-1, out_width=-1, scale_width=-1):
    self.name = name
    
    self.S_q = S_q
    self.S_kv = S_kv
    self.d_kq = d_kq
    self.d_v = d_v
    
    self.k = k
    self.bit_width = bit_width
    self.out_width = out_width
    self.scale_width = scale_width

  def __repr__(self):
    return (
      f"{self.name}_S_q_{self.S_q}_S_kv_{self.S_kv}_d_kq_{self.d_kq}_d_v_{self.d_v}_k_{self.k}_"
      f"bit_width_{self.bit_width}_out_width_{self.out_width}_scale_width_{self.scale_width}"
    )
    
  def __str__(self):
    s = f"Design: {self.name}\n"
    s += f"  S_q: {self.S_q}\n"
    s += f"  S_kv: {self.S_kv}\n"
    s += f"  d_kq: {self.d_kq}\n"
    s += f"  d_v: {self.d_v}\n"
    s += f"  k: {self.k}\n"
    s += f"  bit_width: {self.bit_width}\n"
    s += f"  out_width: {self.out_width}\n"
    s += f"  scale_width: {self.scale_width}\n"
    return s
    
  def get_vivado_tclargs(self):
    return f"{self.S_q} {self.S_kv} {self.d_kq} {self.d_v} {self.k} {self.bit_width} {self.out_width} {self.scale_width}"
  
  @staticmethod
  def get_filename_regex():
    return r"([^/]+_S_q_\d+_S_kv_\d+_d_kq_\d+_d_v_\d+_k_\d+_bit_width_\d+_out_width_\d+_scale_width_\d+)_time_(\d+_\d+)"
  
  @staticmethod
  def get_design_regex():
    return r"([^/]+)_S_q_(\d+)_S_kv_(\d+)_d_kq_(\d+)_d_v_(\d+)_k_(\d+)_bit_width_(\d+)_out_width_(\d+)_scale_width_(\d+)"
  
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
    bit_width = int(details.group(7))
    out_width = int(details.group(8))
    scale_width = int(details.group(9))
    
    return cls(name=name, S_q=S_q, S_kv=S_kv, d_kq=d_kq, d_v=d_v, k=k, bit_width=bit_width, out_width=out_width, scale_width=scale_width)

class SynthesisResult:
  def __init__(self, design_config, power, timing, utilisation):
    self.design_config = design_config
    self.power = power
    self.timing = timing
    self.utilisation = utilisation
    
  def get_aggregated_resource_usage(self):
    return sum(
      self.utilisation[key] / SynthesisHandler.get_available_fpga_resources(key)
      for key in SynthesisHandler.get_available_fpga_resources().keys()
    ) / len(SynthesisHandler.get_available_fpga_resources())
    
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
    
    for result in all_results:
      power['total'] = min(power['total'], result.power['total'])
      power['dynamic'] = min(power['dynamic'], result.power['dynamic'])
      power['static'] = min(power['static'], result.power['static'])
      timing['max_freq'] = max(timing['max_freq'], result.timing['max_freq'])
      for key in SynthesisHandler.get_available_fpga_resources().keys():
        utilisation[key] = min(utilisation[key], result.utilisation[key])
    
    return cls(design_config=design, power=power, timing=timing, utilisation=utilisation)
    
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
    
    return cls(design_config=design, power=power, timing=timing, utilisation=utilisation)
    
  @staticmethod
  def normalise_results(results):
    ideal_result = SynthesisResult.create_ideal_result(results)
    results_normalised = copy.deepcopy(results)
    for result in results_normalised:
      result.power['total'] = result.power['total'] / ideal_result.power['total']
      result.timing['max_freq'] = result.timing['max_freq'] / ideal_result.timing['max_freq']
      
      for key in SynthesisHandler.get_available_fpga_resources().keys():
        result.utilisation[key] = result.utilisation[key] / ideal_result.utilisation[key] if ideal_result.utilisation[key] > 0 else 0.0
        
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

    return s

class SynthesisHandler:
  def __init__(self, designs_to_synthesise=None, hdl_dir="./src/attention/attention_int", clock_period_ns=5):
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
    
  def check_if_results_exist(self, design):
    base_pattern = f"{design!r}" + "_time_*"
    power_matches = glob.glob(os.path.join(self.synth_output_dir, base_pattern + "_power.rpt"))
    timing_matches = glob.glob(os.path.join(self.synth_output_dir, base_pattern + "_timing.rpt"))
    util_matches = glob.glob(os.path.join(self.synth_output_dir, base_pattern + "_util.rpt"))
    
    return power_matches and timing_matches and util_matches
  
  def check_if_design_is_invalid(self, design):
    # All parameters must be >= 0
    for param in [design.S_q, design.S_kv, design.d_kq, design.d_v, design.k, design.bit_width, design.out_width, design.scale_width]:
      if param <= 0:
        return True
    
    # S_q, S_kv, d_kq, d_v must powers of 2 (including 2^0 = 1)
    for param in [design.S_q, design.S_kv, design.d_kq, design.d_v]:
      if (param & (param - 1)) != 0:
        return True
      
    # d_kq and d_v must be divisible by k
    if design.d_kq % design.k != 0 or design.d_v % design.k != 0:
      return True
    
    return False
    
  def run_synthesis(self):
    if not self.designs_to_synthesise:
      print("No designs to synthesise specified.")
      return
    
    print(f"Starting synthesis for {len(self.designs_to_synthesise)} designs...")
    
    for design in self.designs_to_synthesise:
      if self.check_if_results_exist(design):
        print(f"Skipping synthesis for {design!r} as results already exist.")
        continue
      
      if self.check_if_design_is_invalid(design):
        print(f"Skipping synthesis for {design!r} as design configuration is invalid.")
        continue
      
      run_synth_path = os.path.join(self.hdl_dir, "run_synth.tcl")
      synthesis_cmd = f"vivado -mode batch -source {run_synth_path} -tclargs {design.get_vivado_tclargs()}"
      print(f"Results not found, running synthesis command: {synthesis_cmd}")
      
      try:
          start_time = time.perf_counter()
          completed_process = subprocess.run(synthesis_cmd, shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
      except subprocess.CalledProcessError as e:
          print(f"Synthesis failed for {design} with return code: {e.returncode}")
      except Exception as e:
          print(f"An unknown error occurred while running synthesis for {design}: {e}")
          
      end_time = time.perf_counter()
      
      print(f"Synthesis for {design} completed in {end_time - start_time:.2f} seconds.")
          
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
    
  def _process_results(self, design_str, date_time):
    file_path = os.path.join(self.synth_output_dir, f"{design_str}_time_{date_time.strftime(self._time_format)}")
    
    power_report_path = f"{file_path}_power.rpt"
    timing_report_path = f"{file_path}_timing.rpt"
    utilisation_report_path = f"{file_path}_util.rpt"

    try:
      dynamic_power, static_power = self._read_power_report(power_report_path)
      no_timing_violation, max_freq = self._read_timing_report(timing_report_path)
      utilisation = self._read_utilisation_report(utilisation_report_path)
    except FileNotFoundError as e:
      print(f"Error processing {file_path}: {e} - the report is probably being generated, try again later.")
      return
    
    design = DesignConfig.from_str(design_str)
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

  def _pareto_front(self, x, y, powers, res_usages, freqs, maximize_y=True):
    points = list(zip(x, y, powers, res_usages, freqs))

    # 1. Filter dominated points
    non_dominated = []
    for p in points:
        dominated = False
        for q in points:
            # q dominates p if it's better in all objectives
            better_power = q[2] <= p[2]
            better_resources = q[3] <= p[3]
            better_freq = q[4] >= p[4]

            if better_power and better_resources and better_freq and q != p:
                if q[2] < p[2] or q[3] < p[3] or q[4] > p[4]:
                    dominated = True
                    break
        if not dominated:
            non_dominated.append(p)

    # 2. Sort by x
    non_dominated.sort(key=lambda r: r[0])

    # 3. Filter to remove "backward" points in y
    pareto = []
    if maximize_y:
        best_y = -float("inf")
        for p in non_dominated:
            if p[1] > best_y:
                pareto.append(p)
                best_y = p[1]
    else:
        best_y = float("inf")
        for p in non_dominated:
            if p[1] < best_y:
                pareto.append(p)
                best_y = p[1]

    return pareto

  def plot_results(self, directory="./plots", plot_file_format="svg"):
    color_values = np.array([r.design_config.bit_width for r in self.results])
  
    self._plot(
      x=res_usages,
      y=powers,
      color_values=color_values,
      xlabel="Resource Usage (average %)",
      ylabel="Power (W)",
      title=f"Power vs Resource Usage (N={elems_counts[0]})",
      filename=f"power_vs_resource_usage.{plot_file_format}",
      directory=directory,
      do_pareto_front=False,
      do_best_fit_line=True,
    )
    
    self._plot(
      x=res_usages,
      y=max_freqs,
      color_values=color_values,
      xlabel="Resource Usage (average %)",
      ylabel="Max Frequency (MHz)",
      title=f"Max Frequency vs Resource Usage (N={elems_counts[0]})",
      filename=f"max_freq_vs_resource_usage.{plot_file_format}",
      directory=directory,
    )
    
    self._plot(
      x=powers,
      y=max_freqs,
      color_values=color_values,
      xlabel="Power (W)",
      ylabel="Max Frequency (MHz)",
      title=f"Max Frequency vs Power (N={elems_counts[0]})",
      filename=f"max_freq_vs_power.{plot_file_format}",
      directory=directory,
    )

  def _plot(self, x, y, color_values, xlabel, ylabel, title, filename, directory, do_pareto_front=True, do_pareto_optimal=True, do_best_fit_line=False):
    marker_map = {
      "attention_int": "o",
      "attention_fp": "^",
    }

    # Prepare data for Pareto front
    powers = [r.power['total'] for r in self.results]
    res_usages = [
      100 * sum(
        r.utilisation[key] / self.get_available_fpga_resources(key)
        for key in self.get_available_fpga_resources().keys()
      )
      for r in self.results
    ]
    freqs = [r.timing['max_freq'] for r in self.results]

    cmap = matplotlib.cm.get_cmap("viridis", len(np.unique(color_values)))
    bounds = np.arange(color_values.min() - 0.5, color_values.max() + 1.5, 1)
    norm = matplotlib.colors.BoundaryNorm(bounds, cmap.N)

    plt.figure(figsize=(10, 6))
    plotted_markers = {}

    for design, xi, yi, cval in zip(designs, x, y, color_values):
      label = design.name
      marker = marker_map.get(label, "s")
      plt.scatter(
        xi, yi,
        c=[cmap(norm(cval))],
        alpha=1.0,
        s=120,
        marker=marker,
        label=label
      )
      plotted_markers[label] = marker

    plt.title(title, fontsize=18)
    plt.xlabel(xlabel, fontsize=14)
    plt.ylabel(ylabel, fontsize=14)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.grid(True)

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, boundaries=bounds,
                        ticks=np.arange(color_values.min(), color_values.max() + 1))
    cbar.set_label("Total bit width", fontsize=14)
    cbar.ax.tick_params(labelsize=12)

    handles, labels = plt.gca().get_legend_handles_labels()
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
      pareto_points = self._pareto_front(x, y, powers, res_usages, freqs, maximize_y=maximize_y)
      pareto_x = [p[0] for p in pareto_points]
      pareto_y = [p[1] for p in pareto_points]

      plt.plot(pareto_x, pareto_y, linestyle="dashdot", color="black", linewidth=1.2)
      
      pareto_front_legend = matplotlib.lines.Line2D([], [], color="black", linestyle="dashdot", linewidth=1.5, label="Pareto front")
      
      black_handles += [pareto_front_legend]
      unique_labels += ["Pareto front"]
    
    # === Highlight pareto optimal point ===
    if do_pareto_optimal and self.pareto_optimal is not None:
      # Compute X and Y of the pareto optimal point for this plot
      if xlabel.startswith("Resource"):
        x_val = 100 * sum(
          self.pareto_optimal.utilisation[key] / self.get_available_fpga_resources(key)
          for key in self.get_available_fpga_resources().keys()
        )
      else:
        x_val = self.pareto_optimal["power"]["total"]

      if ylabel.startswith("Power"):
        y_val = self.pareto_optimal["power"]["total"]
      else:
        y_val = self.pareto_optimal["timing"]["max_freq"]

      ax = plt.gca()

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
      
    # === Plot best fit line (linear regression) ===
    if do_best_fit_line and len(x) > 1:
      # Fit
      coeffs = np.polyfit(x, y, 1)
      fit_x = np.linspace(min(x), max(x), 100)
      fit_y = np.polyval(coeffs, fit_x)

      # Compute R^2
      y_mean = np.mean(y)
      ss_tot = np.sum((y - y_mean) ** 2)
      ss_res = np.sum((y - np.polyval(coeffs, x)) ** 2)
      r2 = 1 - (ss_res / ss_tot)

      # Plot the line
      plt.plot(fit_x, fit_y, color="gray", linestyle="dashdot", linewidth=1.3)

      # Create a custom handle with R² in label
      best_fit_label = f"Fit, R$^2$ = {r2:.3f}"
      best_fit_handle = plt.Line2D([], [], color="gray", linestyle="dashdot", linewidth=1.3, label=best_fit_label)
      black_handles += [best_fit_handle]
      unique_labels += [best_fit_label]
      
    plt.legend(black_handles, unique_labels, fontsize=14)

    plt.tight_layout()
    plt.savefig(os.path.join(directory, filename))
  
  def __str__(self):
    spacer = "="*60 + "\n"
    return (
      "\t\t\tSynthesis Results:\n" +
      spacer + ("\n" + spacer).join([f"{result!s}" for result in self.results]) + spacer
    )
    
if __name__ == "__main__":  
  designs_to_synthesise = [
    DesignConfig(name, S_q, S_kv, d_kq, d_v, k, bit_width, out_width, scale_width)
    for name in ["attention_int"]
    for S_q in [4]
    for S_kv in [4]
    for d_kq in [4, 8]
    for d_v in [8]
    for k in [2]
    for bit_width in [8]
    for out_width in [8]
    for scale_width in [8]
  ]
  
  synthesis_handler = SynthesisHandler(designs_to_synthesise)
  synthesis_handler.run_synthesis()

  synthesis_handler.find_and_process_results()
  print(synthesis_handler)

  pareto_optimal = synthesis_handler.find_pareto_optimal(weights={'power': 1.0, 'timing': 1.0, 'utilisation': 1.0})
  print(f"\nPareto Optimal Result:\n{pareto_optimal}")

  # TODO update plotting
  # synthesis_handler.plot_results(directory="./plots", plot_file_format="svg")