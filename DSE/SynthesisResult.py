import copy
from DSE.DesignConfig import DesignConfig

LUTS_BASELINE = 11874317
FFS_BASELINE = 2592801

AVAILABLE_FPGA_RESOURCES = {
  "LUTs": 2574208,
  "FFs": 5148416,
  # "CARRY8": 216000,
  # "Muxes": 864000+432000+216000,
  "BRAMs": 3741,
  "DSPs": 10848,
}

class SynthesisResult:
  def __init__(self, design_config, power, timing, utilisation, accuracy):
    self.design_config = design_config
    self.power = power
    self.timing = timing
    self.utilisation = utilisation
    self.accuracy = accuracy
  
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
    utilisation = copy.deepcopy(AVAILABLE_FPGA_RESOURCES)
    accuracy = 1e10
    
    for result in all_results:
      power['total'] = min(power['total'], result.power['total'])
      power['dynamic'] = min(power['dynamic'], result.power['dynamic'])
      power['static'] = min(power['static'], result.power['static'])
      timing['max_freq'] = max(timing['max_freq'], result.timing['max_freq'])
      for key in AVAILABLE_FPGA_RESOURCES.keys():
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
    utilisation = {key: 0.0 for key in AVAILABLE_FPGA_RESOURCES.keys()}
    accuracy = 0.0
    
    return cls(design_config=design, power=power, timing=timing, utilisation=utilisation, accuracy=accuracy)
    
  @staticmethod
  def normalise_results(results):
    ideal_result = SynthesisResult.create_ideal_result(results)
    results_normalised = copy.deepcopy(results)
    for result in results_normalised:
      result.power['total'] = result.power['total'] / ideal_result.power['total']
      result.timing['max_freq'] = result.timing['max_freq'] / ideal_result.timing['max_freq']
      
      for key in AVAILABLE_FPGA_RESOURCES.keys():
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
      s += f"\t{key}: {value:,} ({(value / AVAILABLE_FPGA_RESOURCES[key]) * 100:.2f}%)\n"
      
    s += f"Perplexity: {self.accuracy:.2f}\n" if self.accuracy is not None else "Perplexity: N/A\n"

    return s