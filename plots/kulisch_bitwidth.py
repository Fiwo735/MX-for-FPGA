import numpy as np
import matplotlib.pyplot as plt

if __name__ == "__main__":
    # Data setup
    exponent_widths = np.arange(1, 9)  # 1 to 8
    element_counts = np.array([64, 32, 16, 8, 4, 2])
    data = np.zeros((len(element_counts), len(exponent_widths)))

    for i, n in enumerate(element_counts):
        for j, e in enumerate(exponent_widths):
            data[i, j] = 2 * (2 ** e + np.log2(n) - 2) + 3

    # Plot
    fig, ax = plt.subplots(figsize=(8, 5))
    im = ax.imshow(data, cmap="viridis", aspect="auto")

    # Tick labels
    ax.set_xticks(np.arange(len(exponent_widths)))
    ax.set_xticklabels(exponent_widths)
    ax.set_yticks(np.arange(len(element_counts)))
    ax.set_yticklabels(element_counts)

    ax.set_xlabel(r"Exponent width ($E$)")
    ax.set_ylabel(r"Element count ($N$)")
    ax.set_title(r"Kulisch Accumulator Bitwidth: $2 \cdot ((2^E - 3) + \log_2 N + 1) + G$")


    # Annotate each cell
    for i in range(len(element_counts)):
        for j in range(len(exponent_widths)):
            value = data[i, j]
            ax.text(j, i, f"{value:.0f}", ha="center", va="center", color="white")

    # Colorbar
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Bit width", fontsize=12)

    # Finalise layout and save
    fig.tight_layout()
    fig.savefig("plots/kulisch_bitwidth.png", dpi=300)
