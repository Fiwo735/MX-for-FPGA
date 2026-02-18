import os
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from DSE.SynthesisResult import SynthesisResult, LUTS_BASELINE, FFS_BASELINE

class Plotter:
  def __init__(self, results):
    self.results = results
    self.LUTs = [r.utilisation["LUTs"] for r in self.results]
    self.FFs = [r.utilisation["FFs"] for r in self.results]
    self.accuracies = [r.accuracy for r in self.results]
    self.designs = [r.design_config for r in self.results]
    self.pareto_optimal = None
    
  def find_pareto_optimal(self, weights):
    if not self.results:
      raise ValueError("No synthesis results available to find Pareto optimal solution.")
    
    # Normalise results based on the ideal result
    results_normalised = SynthesisResult.normalise_results(self.results)
    
    # Create normalised ideal result
    ideal_result_normalised = SynthesisResult.create_ideal_result_normalised()
      
    # Find the best result by finding a result that is closest to the ideal result in "distance" in the normalised space
    best_distance = 1e10
    best_index = 0
    
    for index, result in enumerate(results_normalised):
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
    color_values = np.array([r.design_config.get_total_bits() for r in self.results]) # BASELINE and ABLATION: MIXED PRECISION
    # color_values = np.array([r.design_config.get_total_k() for r in self.results]) # ABLATION: MIXED K
    # color_values = np.array([0, 1, 2, 3, 4, 5]) # ABLATION: MIXED ACCUM

    LUTs_mults = np.array(self.LUTs) / LUTS_BASELINE
    FFs_mults = np.array(self.FFs) / FFS_BASELINE
    
    # Create a single figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6), sharey=True, gridspec_kw={'width_ratios': [8, 10]})
    
    self._plot(
      fig=fig,
      ax=ax1,
      x=LUTs_mults,
      y=self.accuracies,
      color_values=color_values,
      xlabel="LUTs (×baseline)",
      ylabel="Perplexity",
      title=f"Perplexity vs LUTs",
      resource="LUTs",
      show_colorbar=False
    )
    
    self._plot(
      fig=fig,
      ax=ax2,
      x=FFs_mults,
      y=self.accuracies,
      color_values=color_values,
      xlabel="FFs (×baseline)",
      ylabel="Perplexity",
      title=f"Perplexity vs FFs",
      resource="FFs",
      show_colorbar=True
    )
    
    # Save the combined figure
    fig.tight_layout()
    fig.savefig(os.path.join(directory, f"perplexity_combined_{filename_suffix}.{plot_file_format}"))

  def _plot(self, fig, ax, x, y, color_values, xlabel, ylabel, title, resource,
            do_pareto_front=True, do_pareto_optimal=True, show_colorbar=True):
    
    marker_map = {
      True: "o",   # baseline
      False: "s",  # new
    }
    
    # ABLATION: BASELINE & MIXED PRECISION
    cmap = matplotlib.colormaps["viridis"].resampled(color_values.max() - color_values.min() + 1)
    bounds = np.arange(color_values.min() - 0.5, color_values.max() + 1.5, 1)
    norm = matplotlib.colors.BoundaryNorm(bounds, cmap.N)
    
    # # ABLATION: MIXED K
    # step = 16
    # vmin = int(np.floor(color_values.min() / step) * step)
    # vmax = int(np.ceil(color_values.max() / step) * step)
    # ticks = np.arange(vmin, vmax + 1, step)
    # bounds = np.concatenate(([ticks[0] - step / 2], (ticks[:-1] + ticks[1:]) / 2, [ticks[-1] + step / 2]))
    # norm = matplotlib.colors.BoundaryNorm(bounds, len(ticks))
    # cmap = matplotlib.colormaps["viridis"].resampled(len(ticks))
    
    plotted_markers = {}

    for design, xi, yi, cval in zip(self.designs, x, y, color_values):
      # other_label = "Mixed precision" # ABLATION: MIXED PRECISION
      # other_label = "Mixed block size"        # ABLATION: MIXED K
      # other_label = "Mixed accumulation method"  # ABLATION: MIXED ACCUM
      other_label = "New design" # JOINT ABLATION
      
      # label = "Baseline" if design.is_baseline() else other_label # NO-JOINT ABLATION
      label = other_label # JOINT ABLATION
      # marker = marker_map[design.is_baseline()] # NO-JOINT ABLATION
      marker = marker_map[False] # JOINT ABLATION
      
      ax.scatter(
        xi, yi,
        c=[cmap(norm(cval))],
        alpha=1.0,
        s=120,
        marker=marker,
        # label=label # NO-JOINT ABLATION
      )
      plotted_markers[label] = marker

    ax.set_title(title, fontsize=20)
    ax.set_xlabel(xlabel, fontsize=18)
    ax.set_ylabel(ylabel, fontsize=18)
    
    # ax.set_ylim(bottom=9, top=18) # BASELINE
    # ax.set_ylim(bottom=9, top=35) # ABLATION: MIXED PRECISION ONLY
    # ax.set_ylim(bottom=9.75, top=11) # ABLATION: MIXED K
    # ax.set_ylim(bottom=9.75, top=15) # ABLATION: MIXED ACCUM
    ax.set_ylim(bottom=9.7, top=11.2) # JOINT ABLATION
    
    
    ax.tick_params(axis='x', labelsize=16)
    ax.tick_params(axis='y', labelsize=16)
    ax.grid(True)
    
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    if show_colorbar is True:
      cbar = plt.colorbar(sm, ax=ax, boundaries=bounds, ticks=np.arange(color_values.min(), color_values.max() + 1)) # BASELINE and ABLATION: MIXED PRECISION
      # cbar = plt.colorbar(sm, ax=ax, boundaries=bounds, ticks=ticks) # ABLATION: MIXED K & ACCUM
      cbar.set_label("Combined bit widths across operators", fontsize=18) # BASELINE and ABLATION: MIXED PRECISION
      # cbar.set_label("Combined block sizes across operators", fontsize=18) # ABLATION: MIXED K
      # cbar.set_label("Accumulation method across operators", fontsize=18) # ABLATION: MIXED ACCUM
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
    weights={'LUTs': 1.0, 'FFs': 1.0, 'accuracy': 100.0}
    if do_pareto_optimal and self.find_pareto_optimal(weights=weights) is not None:
      # Compute X and Y of the pareto optimal point for this plot
      baseline = LUTS_BASELINE if resource == "LUTs" else FFS_BASELINE
      x_val = self.pareto_optimal.utilisation[resource] / baseline
      y_val = self.pareto_optimal.accuracy

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

    if not show_colorbar:
      ax.legend().set_visible(False)
    else:
      ax.legend(black_handles, unique_labels, fontsize=14)

    fig.tight_layout()
    