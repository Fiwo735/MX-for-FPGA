from argparse import ArgumentParser
from DSE.AccumMethod import AccumMethod
from DSE.DesignConfig import DesignConfig
from DSE.SynthesisHandler import SynthesisHandler
from DSE.analytical_model import calibrate_analytical_models, predict_synthesis_results
from DSE.Plotter import Plotter


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
  
  plotter = Plotter(synthesis_handler.results)
  plotter.plot_perplexity(directory="./plots", filename_suffix="joint", plot_file_format="png")

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
  