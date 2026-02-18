import os
import glob
import re
import subprocess
import time
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor, as_completed


from DSE.analytical_model import predict_synthesis_results
from DSE.SynthesisResult import SynthesisResult, LUTS_BASELINE, FFS_BASELINE
from DSE.DesignConfig import DesignConfig

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
    self.pickle_dir = "./synthesis_fits"
    
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
    accuracy_cmd = f"CUDA_VISIBLE_DEVICES=1 python -u quant/llama_ppl.py {design.get_quant_flags()}"
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
      # if (not design.is_baseline()) and (not design.is_mixed_accum_ablation()):
      #   if verbose:
      #     print(f"Skipping result for {design} as it does not meet ablation check criteria.")
      #   return
      
      # JOINT ABLATION:
      if (not design.is_joint_ablation()):
        if verbose:
          print(f"Skipping result for {design} as it does not meet ablation check criteria.")
        return
      
      if result.accuracy < 0:
        if verbose:
          print(f"Skipping result for {design} as accuracy could not be determined.")
        return
      
      if result.accuracy > 12:
        if verbose:
          print(f"Skipping result for {design} as accuracy is too high.")
        return
      
      if result.utilisation["LUTs"] > (2.5 * LUTS_BASELINE):
        if verbose:
          print(f"Skipping result for {design} as it exceeds FPGA resource limits.")
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
      
    self.designs = [r.design_config for r in self.results]
    self.powers = [r.power['total'] for r in self.results]
    self.LUTs = [r.utilisation["LUTs"] for r in self.results]
    self.FFs = [r.utilisation["FFs"] for r in self.results]
    self.BRAMs = [r.utilisation["BRAMs"] for r in self.results]
    self.DSPs = [r.utilisation["DSPs"] for r in self.results]
    self.accuracies = [r.accuracy for r in self.results]
  
  def __str__(self):
    spacer = "="*60 + "\n"
    return (
      f"\t\t\t{len(self.results)} Synthesis Results:\n" +
      spacer + ("\n" + spacer).join([f"{result!s}" for result in self.results]) + spacer
    )