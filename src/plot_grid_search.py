import json
from pathlib import Path

import matplotlib
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

DPI = 600


def load_grid_results(base_dir='../runs/grid_search'):
    base = Path(base_dir)
    index_path = base / 'index.json'

    with open(index_path, 'r') as f:
        index = json.load(f)

    records = []
    for run in index['runs']:
        if run.get('status') != 'ok':
            continue
        L = run['L']
        h = run['hidden_size']
        result_path = Path(run['run_dir']) / 'results.npz'
        if not result_path.exists():
            continue

        data = np.load(result_path, allow_pickle=True)
        meta = data['metadata'].item()
        best_loss = meta.get('best_loss', np.nan)
        records.append({'L': L, 'hidden': h, 'best_loss': best_loss})

    df = pd.DataFrame(records)
    df.sort_values(by=['L', 'hidden'], inplace=True)
    return df


def load_runs(base_dir: Path):
    index_path = base_dir / "index.json"
    if not index_path.exists():
        raise FileNotFoundError(f"index.json not found at {index_path}")

    with open(index_path, "r") as f:
        index = json.load(f)

    runs = []
    for entry in index.get("runs", []):
        if entry.get("status") != "ok":
            continue
        run_dir = Path(entry.get("run_dir", ""))
        results_path = run_dir / "results.npz"
        if not results_path.exists():
            # be lenient: try alternate naming or skip
            continue
        try:
            data = np.load(results_path, allow_pickle=True)
            losses = np.asarray(data["losses"], dtype=float)
        except Exception as e:
            print(f"Warning: could not load {results_path}: {e}")
            continue

        L = int(entry.get("L"))
        hidden = int(entry.get("hidden_size", -1))
        runs.append({
            "L": L,
            "hidden": hidden,
            "losses": losses,
            "run_dir": str(run_dir)
        })

    if len(runs) == 0:
        raise RuntimeError("No successful runs found in index.json")
    return runs


def plot_all_runs(runs,
                  out_base: Path,
                  figsize=(9, 6),
                  dpi=600,
                  ylog=False,
                  cmap_name="viridis_r",
                  linewidth=1.0,
                  alpha=0.9,
                  network_size=False):
    # Prepare L range and colormap
    if network_size:
        L_values = sorted({r["L"] * r['hidden'] for r in runs})
    else:
        L_values = sorted({r["L"] for r in runs})
    L_min, L_max = min(L_values), max(L_values)

    norm = mpl.colors.Normalize(vmin=L_min, vmax=L_max)
    cmap = matplotlib.colormaps.get_cmap(cmap_name)
    sm = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)  # for colorbar

    # Matplotlib global style for publication-quality
    mpl.rcParams.update({
        "figure.dpi": dpi,
        "savefig.dpi": dpi,
        "font.size": 11,
        "axes.titlesize": 12,
        "axes.labelsize": 11,
        "legend.fontsize": 10,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
    })

    fig, ax = plt.subplots(figsize=figsize)

    # Plot each run individually
    max_epoch = 0
    for r in runs:

        if network_size:
            L = r["L"] * r['hidden']
        else:
            L = r["L"]

        losses = r["losses"]
        epochs = np.arange(1, len(losses) + 1)
        max_epoch = max(max_epoch, len(losses))

        color = cmap(norm(L))
        ax.plot(epochs, losses, linewidth=linewidth, alpha=alpha, color=color, solid_capstyle='round')

    # Axis formatting

    # Grid: light & unobtrusive
    ax.grid(True, linestyle="--", linewidth=0.4, alpha=0.6)

    # Y-scale (log recommended)
    if ylog:
        ax.set_yscale("log")

    # Colorbar for L mapping
    sm.set_array([])  # needed for older matplotlib
    cbar = fig.colorbar(sm, ax=ax, pad=0.02, fraction=0.05)

    ax.set_xlabel("Epoch", fontsize=12)
    ax.set_ylabel("Training Loss (MSE)", fontsize=12)

    if network_size:
        ax.set_title("Training loss per run vs # Parameters", fontsize=13)

        cbar.set_label("# Parameters", fontsize=11)
    else:
        ax.set_title("Training loss per run vs network depth $L$", fontsize=13)

        cbar.set_label("Number of residual blocks (L)", fontsize=11)

    # Try to use integer ticks for L
    if L_max - L_min <= 12:
        cbar.set_ticks(np.arange(L_min, L_max + 1))
    else:
        # choose ~6 ticks
        cbar.set_ticks(np.linspace(L_min, L_max, 6))
    cbar.ax.tick_params(labelsize=10)

    # Save outputs
    out_png = out_base.with_suffix(".png")
    fig.savefig(out_png, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_png}")


def parameter_plots():
    # === User-configurable ===
    BASE_DIR = Path("../runs/grid_search")  # location of index.json and run folders
    FIGSIZE = (9, 6)

    YSCALE_LOG = False  # set False if you prefer linear y-axis
    LINEWIDTH = 1.5
    ALPHA = 1.0  # visibility of each line (1.0 is fully opaque)
    COLORMAP = "viridis_r"  # reversed viridis: small L -> lighter, large L -> darker

    runs = load_runs(BASE_DIR)
    print(f"Found {len(runs)} runs. L values: {sorted({r['L'] for r in runs})}")

    l_size_name = BASE_DIR / "losses_by_L"

    plot_all_runs(runs, l_size_name, figsize=FIGSIZE, dpi=DPI, ylog=YSCALE_LOG,
                  cmap_name=COLORMAP, linewidth=LINEWIDTH, alpha=ALPHA)

    param_size_name = BASE_DIR / "losses_by_parameter_size"
    plot_all_runs(runs, param_size_name, figsize=FIGSIZE, dpi=DPI, ylog=YSCALE_LOG,
                  cmap_name=COLORMAP, linewidth=LINEWIDTH, alpha=ALPHA, network_size=True)


def plot_grid_results(df, save_path='../runs/grid_search/summary_plot.png', dpi=600):
    # Create a pivot table (rows=L, cols=hidden)
    pivot = df.pivot(index='hidden', columns='L', values='best_loss')

    plt.figure(figsize=(6, 4))
    sns.set_context("paper")
    sns.heatmap(
        pivot,
        annot=True,
        fmt=".4f",
        cmap="viridis_r",
        cbar_kws={'label': 'Best Training Loss'},
        linewidths=0.5,
    )
    plt.xlabel("Number of Layers (L)")
    plt.ylabel("Hidden Layer Size (H)")
    plt.title("Deep Lipschitz ResNet Performance")

    plt.tight_layout()
    plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
    plt.close()
    print(f"Saved summary heatmap to {save_path}")


if __name__ == "__main__":
    df = load_grid_results("../runs/grid_search")
    print(df)
    plot_grid_results(df, dpi=DPI)

    parameter_plots()
