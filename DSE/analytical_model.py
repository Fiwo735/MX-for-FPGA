import pickle
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from gplearn.genetic import SymbolicRegressor

from DSE.DesignConfig import DesignConfig
from DSE.AccumMethod import AccumMethod

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
    
def find_fit(results, y_type, data, pickle_dir, degree=2, threshold=1e-3, verbose=True, pickle_suffix=""):
  # # Create a DataFrame from design parameters
  df = pd.DataFrame(data)
  if y_type == "LUTs":
    y = np.array([r.utilisation["LUTs"] for r in results])
  elif y_type == "FFs":
    y = np.array([r.utilisation["FFs"] for r in results])
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
  
  with open(f"{pickle_dir}/fit_model_{y_type}_{pickle_suffix}.pkl", "wb") as f:
    pickle.dump({
        "model": model,
        "poly": poly,
        "feature_names": df.columns.tolist()
    }, f)
  
def find_fit_with_gplearn(results, y_type, X, population_size=5000, generations=50, parsimony_coefficient=1e-3):
  # Prepare the design matrix
  if y_type == "LUTs":
    y = np.array([r.utilisation["LUTs"] for r in results])
  elif y_type == "FFs":
    y = np.array([r.utilisation["FFs"] for r in results])
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
    
def calibrate_analytical_models(verbose):
  from DSE.SynthesisHandler import SynthesisHandler
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

  find_fit(synthesis_handler.results, "LUTs", matmul_fit_data, pickle_dir=synthesis_handler.pickle_dir, degree=2, threshold=0, verbose=True, pickle_suffix="matmul")
  find_fit(synthesis_handler.results, "FFs", matmul_fit_data, pickle_dir=synthesis_handler.pickle_dir, degree=2, threshold=0, verbose=True, pickle_suffix="matmul")
  
  # matmul_fit_data_gplearn = np.array([[d.S_q, d.d_kq, d.M1_bits.exp_bits + d.M1_bits.mant_bits] for d in synthesis_handler.designs])
  
  # find_fit_with_gplearn(synthesis_handler.results, "LUTs", matmul_fit_data_gplearn,       population_size=5000, generations=20, parsimony_coefficient=0.0001)
  # find_fit_with_gplearn(synthesis_handler.results, "FFs", matmul_fit_data_gplearn,        population_size=5000, generations=20, parsimony_coefficient=0.0001)
  
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
  
  find_fit(synthesis_handler.results, "LUTs", softmax_fit_data, pickle_dir=synthesis_handler.pickle_dir, degree=3, threshold=0, verbose=True, pickle_suffix="softmax")
  find_fit(synthesis_handler.results, "FFs", softmax_fit_data, pickle_dir=synthesis_handler.pickle_dir, degree=2, threshold=0, verbose=True, pickle_suffix="softmax")
    
  # softmax_fit_data_gplearn = np.array([[d.k2, d.M2_bits.exp_bits + d.M2_bits.mant_bits, d.M3_bits.exp_bits + d.M3_bits.mant_bits] for d in synthesis_handler.designs])
    
  # find_fit_with_gplearn(synthesis_handler.results, "LUTs", softmax_fit_data_gplearn,       population_size=5000, generations=20, parsimony_coefficient=0.0001)
  # find_fit_with_gplearn(synthesis_handler.results, "FFs", softmax_fit_data_gplearn,        population_size=5000, generations=20, parsimony_coefficient=0.0001)