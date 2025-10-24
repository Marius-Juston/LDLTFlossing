import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


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


def plot_grid_results(df, save_path='../runs/grid_search/summary_plot.png', dpi=600):
    # Create a pivot table (rows=L, cols=hidden)
    pivot = df.pivot(index='hidden', columns='L', values='best_loss')

    plt.figure(figsize=(6, 4))
    sns.set_context("paper")
    sns.heatmap(
        pivot,
        annot=True,
        fmt=".4f",
        cmap="viridis",
        cbar_kws={'label': 'Best Training Loss'},
        linewidths=0.5,
    )
    plt.xlabel("Number of Layers (L)")
    plt.ylabel("Hidden Layer Size")
    plt.title("Deep Lipschitz ResNet Performance")

    plt.tight_layout()
    plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
    plt.close()
    print(f"Saved summary heatmap to {save_path}")


if __name__ == "__main__":
    df = load_grid_results("../runs/grid_search")
    print(df)
    plot_grid_results(df)
