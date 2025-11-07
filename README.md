# Gradient Flossing for LDLT Lipschitz Networks

Research code for training Lipschitz-constrained linear networks with gradient flossing, exhaustive grid searches, and publication-ready visualization pipelines.

## Prerequisites

- Python 3.11
- PyTorch build with CUDA support (training scripts assume at least one NVIDIA GPU)
- Linux is recommended; `torch.compile` defaults to CUDA-optimized backends

## Setup

1. (Optional) Create and activate a virtual environment.
2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Verify GPU visibility with `python -c "import torch; print(torch.cuda.device_count())"`.

All scripts assume execution from the project root so that ``src`` is on `PYTHONPATH`.

## Project Structure

```
LDLTFlossing/
├─ requirements.txt          # Python dependencies
├─ src/
│  ├─ utils.py               # Linear algebra and autograd helpers
│  ├─ linear_train.py        # Training loops + sine benchmarks
│  ├─ grid_search.py         # Multi-GPU hyper-parameter sweeps
│  ├─ plot_grid_search.py    # Visualization utilities
│  └─ models/
│     ├─ linear_layers.py    # Lipschitz-aware linear blocks
│     └─ linear_model.py     # Sequential, residual, and stacked models
├─ test/                     # Reproduction scripts and archived figures
└─ runs/                     # TensorBoard logs and grid-search outputs (generated)
```

## Script Reference

### `src/linear_train.py`

Purpose: train Lipschitz networks on the sine benchmark with optional gradient flossing.

Entry point: running `python src/linear_train.py` executes `sine_training_flossing(L=30)` by default.

Key callables:

| Function | Description | Arguments |
| --- | --- | --- |
| `sine_training(L, hidden, epochs, device_id)` | Plain sequential training without flossing. | `L`: number of hidden blocks; `hidden`: width per block; `epochs`: training epochs; `device_id`: CUDA device index. |
| `sine_training_flossing(L, hidden, epochs, device_id)` | Same benchmark but enables gradient flossing (see `FlossingConfig`). | Same arguments as above plus internal flossing defaults (`weight=0.1`, `flossing_frequency=1`). |
| `sine_training_grouped(L, hidden, L_int, epochs, device_id)` | Trains the stacked sequential architecture. | `L`: number of outer stacks; `hidden`: width; `L_int`: interior depth per stack; remaining args as above. |
| `train(x, y, model, flossing_config, ...)` | General-purpose trainer invoked by the helpers. | `x`, `y`: tensors with shapes `(N, in_features)` and `(N, out_features)`; `model`: any Lipschitz module; `flossing_config`: optional `FlossingConfig`; hyper-parameters such as `batch_size`, `lr`, `epochs`, `termination_error`, `logging_frequency`. |

The module also exposes diagnostic utilities (`alpha_values`, `print_num_parameters`, `initial_train_condition`, etc.) that log tensor statistics to TensorBoard.

### `src/grid_search.py`

Purpose: launch multi-GPU sweeps across depth, width, and flossing options.

Usage:

```bash
python src/grid_search.py
```

The `grid_search` function is the primary entry point and supports the following arguments:

| Argument | Description | Default |
| --- | --- | --- |
| `L_start`, `L_end`, `L_step` | Minimum, maximum, and step for the outer depth (`L`). | `3`, `15`, `2` |
| `hidden` | Tuple of widths to test (applied per layer). | `(8, 16, 32, 64, 128)` |
| `epochs`, `batch_size`, `lr` | Training hyper-parameters forwarded to `train`. | `20`, `64`, `1e-4` |
| `base_save_dir` | Directory that receives per-run metadata (`runs/grid_search`). | `'../runs/grid_search'` |
| `num_gpus` | Number of concurrent worker processes / CUDA devices. | `4` |
| `num_interior` | Interior depth per outer block when `stack=True`. | `5` |
| `stack` | Switch to the stacked architecture. | `False` |
| `flossing` | Enables `FlossingConfig` for every job. | `False` |

Adjust the `if __name__ == '__main__'` block or call `grid_search(...)` from a REPL/script to customize sweeps.

### `src/plot_grid_search.py`

Purpose: post-process grid-search runs into tables, scatter plots, and Lyapunov overlays.

Two helper functions cover the main workflows:

| Function | Description | Arguments |
| --- | --- | --- |
| `plot_grid_results(stacked=False, flossing=False, dpi=600)` | Creates a heatmap of best losses across `(L, hidden)` pairs. | Use the same `stacked`/`flossing` flags that generated the runs; `dpi` controls figure resolution. |
| `parameter_plots(stacked=False, flossing=False)` | Produces per-run training curves plus Lyapunov summaries for success/failure groups. | Flags mirror the training configuration to locate the correct `runs/grid_search[_suffix]` directory. |

Both functions expect an `index.json` under `runs/grid_search*` as produced by `grid_search.py`.

## Typical Workflows

1. **Single-run sanity check** – run `python src/linear_train.py` and inspect TensorBoard logs inside `runs/`.
2. **Hyper-parameter sweep** – edit `src/grid_search.py` (or import `grid_search`) with your target search space, then execute the script. Results accumulate under `runs/grid_search*`.
3. **Visualization** – once runs finish, execute `python src/plot_grid_search.py` (adjusting `stacked`/`flossing`) to generate publication-quality figures.

## Notes

- All training utilities expect CUDA tensors; CPU execution is possible but significantly slower.
- The helper modules in `src/utils.py` and `src/models/` are documented inline with tensor shapes and should serve as the canonical reference for extending the architecture.
