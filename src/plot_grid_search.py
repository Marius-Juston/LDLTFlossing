"""Visualization helpers for summarizing grid-search experiments."""

import json
from pathlib import Path

import matplotlib
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.lines import Line2D
from typing import Any, Callable, Dict, Iterable, List, Tuple

DPI = 600


def load_grid_results(base_dir: str = '../runs/grid_search') -> pd.DataFrame:
    """Load per-run metadata into a tidy dataframe.

    Args:
        base_dir: Directory that contains ``index.json`` and run folders.

    Returns:
        pandas.DataFrame: One row per successful configuration with the best
        loss, structural parameters, and model size.
    """
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
        model_size = run['model_size']

        result_path = Path(run['run_dir']) / 'results.npz'
        if not result_path.exists():
            continue

        data = np.load(result_path, allow_pickle=True)
        meta = data['metadata'].item()
        best_loss = meta.get('best_loss', np.nan)

        data = {'L': L, 'hidden': h, 'model_size': model_size, 'best_loss': best_loss}

        if 'num_interior' in run:
            data.update({'num_interior': run['num_interior']})

        records.append(data)

    df = pd.DataFrame(records)
    df.sort_values(by=['L', 'hidden'], inplace=True)
    return df


def load_runs(base_dir: Path) -> List[Dict[str, Any]]:
    """Load every successful training run as a list of dictionaries.

    Args:
        base_dir: Directory containing ``index.json``.

    Returns:
        list[dict]: Metadata dictionaries with losses, model sizes, and
        Lyapunov statistics for each run that finished successfully.

    Raises:
        FileNotFoundError: If ``index.json`` does not exist.
        RuntimeError: If no successful run is discovered.
    """
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

        running_lyapunov_exponents = np.asarray(entry["running_lyapunov_exponents"], dtype=float)
        conditional_lyaponov_exponents = np.asarray(entry["conditional_lyaponov_exponents"], dtype=float)

        L = int(entry.get("L"))
        hidden = int(entry.get("hidden_size", -1))
        model_size = int(entry.get("model_size", -1))
        num_interior = int(entry.get("num_interior", -1))

        final_loss = float(losses[-1]) if losses.size > 0 else float("nan")

        runs.append({
            "L": L,
            "hidden": hidden,
            "losses": losses,
            "final_loss": final_loss,
            "model_size": model_size,
            "num_interior": num_interior,
            "run_dir": str(run_dir),
            'conditional_lyaponov_exponents': conditional_lyaponov_exponents,
            'running_lyapunov_exponents': running_lyapunov_exponents
        })

    if len(runs) == 0:
        raise RuntimeError("No successful runs found in index.json")
    return runs


def _compute_group_means(runs: Iterable[Dict[str, Any]],
                         key: str,
                         group_fn: Callable[[Dict[str, Any]], bool]) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Compute split-mean curves for a ragged time series.

    Args:
        runs: Iterable of run dictionaries returned by :func:`load_runs`.
        key: Name of the time-series entry to summarize.
        group_fn: Callable deciding whether a run counts as *success*.

    Returns:
        tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]: Time/value
        arrays for successful and unsuccessful runs respectively.
    """
    # collect sequences
    success_seqs = []
    fail_seqs = []
    max_len_success = 0
    max_len_fail = 0

    for r in runs:
        arr = r.get(key, None)
        if arr is None:
            continue
        arr = np.asarray(arr, dtype=float).ravel()
        if arr.size == 0:
            continue

        if group_fn(r):
            success_seqs.append(arr)
            max_len_success = max(max_len_success, arr.size)
        else:
            fail_seqs.append(arr)
            max_len_fail = max(max_len_fail, arr.size)

    def _mean_over_ragged(seqs, max_len):
        if max_len == 0 or len(seqs) == 0:
            return np.array([])
        means = []
        for t in range(max_len):
            vals_t = [seq[t] for seq in seqs if seq.size > t]
            if len(vals_t) == 0:
                break
            means.append(np.mean(vals_t))
        return np.asarray(means, dtype=float)

    mean_success = _mean_over_ragged(success_seqs, max_len_success)
    mean_fail = _mean_over_ragged(fail_seqs, max_len_fail)

    T_success = np.arange(1, mean_success.size + 1) if mean_success.size else np.array([])
    T_fail = np.arange(1, mean_fail.size + 1) if mean_fail.size else np.array([])

    return T_success, mean_success, T_fail, mean_fail


def is_success_run(run: Dict[str, Any], threshold: float = 0.2) -> bool:
    """Check whether a run converged below a desired loss threshold."""
    final_loss = run.get("final_loss", float("nan"))
    if np.isnan(final_loss):
        return False
    return final_loss < threshold


def plot_all_runs(runs: Iterable[Dict[str, Any]],
                  out_base: Path,
                  figsize: Tuple[int, int] = (9, 6),
                  dpi: int = 600,
                  ylog: bool = False,
                  cmap_name: str = "viridis_r",
                  linewidth: float = 1.0,
                  alpha: float = 0.9,
                  network_size: bool = False) -> None:
    """Plot training-loss curves (1-D per epoch) colored by depth/size."""
    # Prepare L range and colormap
    if network_size:
        L_values = sorted({r['model_size'] if 'model_size' in r else r["L"] * r['hidden'] for r in runs})
    else:
        L_values = sorted({r['L'] * r['num_interior'] if 'num_interior' in r else r["L"] for r in runs})
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
            L = r['model_size'] if 'model_size' in r else r["L"] * r['hidden']
        else:
            L = r['L'] * r['num_interior'] if 'num_interior' in r else r["L"]

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

        cbar.set_label("Number of hidden layers (L)", fontsize=11)

    # Try to use integer ticks for L
    if L_max - L_min <= 12:
        cbar.set_ticks(np.arange(L_min, L_max + 1))
    else:
        # choose ~6 ticks
        cbar.set_ticks(np.linspace(L_min, L_max, 6))
    # cbar.ax.tick_params(labelsize=10)

    # Save outputs
    out_png = out_base.with_suffix(".png")
    fig.savefig(out_png, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_png}")


SUCCESS_COLOR = 'tab:green'
FAIL_COLOR = 'tab:red'


def plot_lyapunov_runs(runs: Iterable[Dict[str, Any]], out_path: Path,
                       title: str = "Running Lyapunov exponents") -> None:
    """Overlay per-step Lyapunov curves (1-D sequences) and highlight group means."""
    fig, ax = plt.subplots()

    # 1) plot all individual runs, faint
    max_T = 0
    for r in runs:
        lyap = r.get("running_lyapunov_exponents", None)
        if lyap is None:
            continue
        lyap = np.asarray(lyap, dtype=float).ravel()
        if lyap.size == 0:
            continue

        T = np.arange(1, lyap.size + 1)
        max_T = max(max_T, lyap.size)

        color = SUCCESS_COLOR if is_success_run(r) else FAIL_COLOR
        ax.plot(T, lyap, color=color, linewidth=1.0, alpha=0.1)

    # 2) compute and draw the two group means (over ragged sequences)
    T_s, mean_s, T_f, mean_f = _compute_group_means(
        runs,
        key="running_lyapunov_exponents",
        group_fn=is_success_run,
    )

    ax.set_xlabel("Training steps")
    ax.set_ylabel("Lyapunov exponent")
    ax.set_title(title)

    # good practice for dynamical quantities
    ax.axhline(0.0, color="black", linewidth=0.6, alpha=0.5)

    # Legend: make sure both individual & mean are explained

    legend_elems = [
        Line2D([0], [0], color=SUCCESS_COLOR, lw=1., alpha=0.1, label=r"Converged (loss $\approx 0$)"),
        Line2D([0], [0], color=FAIL_COLOR, lw=1., alpha=0.1, label="No learning"),
    ]
    # add mean lines (only if they exist)
    # if mean_s.size:
    #     line_mean_s, = ax.plot(T_s, mean_s, color=SUCCESS_COLOR, linewidth=2.0, alpha=1.0,
    #                            label=r"Converged mean (loss $\approx 0$)")
    #
    #     legend_elems.append(line_mean_s)
    # if mean_f.size:
    #     line_mean_f, = ax.plot(T_f, mean_f, color=FAIL_COLOR, linewidth=2.0, alpha=1.0,
    #                            label="No learning mean")
    #
    #     legend_elems.append(line_mean_f)

    ax.legend(handles=legend_elems, loc='lower left')

    ax.grid(True, linestyle=":", linewidth=0.4, alpha=0.5)

    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight", dpi=DPI)
    plt.close(fig)
    print(f"Saved: {out_path}")


def plot_conditional_lyapunov_runs(runs: Iterable[Dict[str, Any]], out_path: Path,
                                   title: str = "Conditional Lyapunov exponents") -> None:
    """Plot flossing-phase Lyapunov sequences keyed by ``conditional_lyaponov_exponents``."""
    fig, ax = plt.subplots()

    # 1) plot all individual runs, faint
    for r in runs:
        lyap = r.get("conditional_lyaponov_exponents", None)
        if lyap is None:
            continue
        lyap = np.asarray(lyap, dtype=float).ravel()
        if lyap.size == 0:
            continue

        T = np.arange(1, lyap.size + 1)

        color = SUCCESS_COLOR if is_success_run(r) else FAIL_COLOR
        ax.plot(T, lyap, color=color, linewidth=1.0, alpha=0.1)

    # 2) group means
    T_s, mean_s, T_f, mean_f = _compute_group_means(
        runs,
        key="conditional_lyaponov_exponents",
        group_fn=is_success_run,
    )

    ax.set_xlabel("Pre-flossing Training Steps")
    ax.set_ylabel("Lyapunov exponent")
    ax.set_title(title)
    ax.axhline(0.0, color="black", linewidth=0.6, alpha=0.5)

    legend_elems = [
        Line2D([0], [0], color=SUCCESS_COLOR, lw=1.0, alpha=0.1, label=r"Converged (loss $\approx 0$)"),
        Line2D([0], [0], color=FAIL_COLOR, lw=1.0, alpha=0.1, label="No learning"),
    ]

    if mean_s.size:
        line_mean_s, = ax.plot(T_s, mean_s, color=SUCCESS_COLOR, linewidth=2.0, alpha=1.0,
                               label=r"Converged mean (loss $\approx 0$)")

        legend_elems.append(line_mean_s)
    if mean_f.size:
        line_mean_f, = ax.plot(T_f, mean_f, color=FAIL_COLOR, linewidth=2.0, alpha=1.0,
                               label="No learning mean")

        legend_elems.append(line_mean_f)

    ax.legend(handles=legend_elems, loc='lower right')

    ax.grid(True, linestyle=":", linewidth=0.4, alpha=0.5)

    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight", dpi=DPI)
    plt.close(fig)
    print(f"Saved: {out_path}")


def parameter_plots(stacked: bool = False, flossing: bool = False) -> None:
    """Generate publication-ready loss plots for a grid-search directory."""
    # === User-configurable ===
    path = "../runs/grid_search"

    if stacked:
        path += '_stack'

    if flossing:
        path += '_flossing'

    BASE_DIR = Path(path)  # location of index.json and run folders
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

    lyapunov_running_path = BASE_DIR / "lyapunov_running.png"
    lyapunov_conditional_path = BASE_DIR / "lyapunov_conditional.png"

    plot_lyapunov_runs(
        runs,
        out_path=lyapunov_running_path,
        title="Lyapunov exponents of networks during training",
    )

    plot_conditional_lyapunov_runs(
        runs,
        out_path=lyapunov_conditional_path,
        title="Lyapunov exponents of networks during pre-flossing"
    )


def plot_grid_results(stacked: bool = False, flossing: bool = False, dpi: int = 600) -> None:
    """Create a heatmap of best losses across depth/width pairs."""
    foldeR_name = f'grid_search'

    if stacked:
        foldeR_name += '_stack'
    if flossing:
        foldeR_name += '_flossing'

    # Create a pivot table (rows=L, cols=hidden)
    df = load_grid_results(f"../runs/{foldeR_name}")

    pivot = df.pivot(index='hidden', columns='L', values='best_loss')

    if 'num_interior' in df.columns:
        pivot.columns *= df['num_interior'][0]

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

    save_path = f'../runs/{foldeR_name}/summary_plot.png'

    plt.tight_layout()
    plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
    plt.close()
    print(f"Saved summary heatmap to {save_path}")


if __name__ == "__main__":
    stacked = False
    flossing = True

    plot_grid_results(stacked=stacked, flossing=flossing, dpi=DPI)

    parameter_plots(stacked, flossing)
