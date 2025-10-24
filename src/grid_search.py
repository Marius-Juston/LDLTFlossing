import json
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

from linear_train import train
from models.linear_model import DeepLipschitzSequential


def worker_run(task):
    """
    Worker function to run one training instance in a separate process.
    `task` is a dict with keys: L, hidden_size, epochs, batch_size, lr, device_id, base_save_dir.
    This function builds the model on the specified GPU and calls the existing `train`.
    It returns a dict with metadata and file paths.
    """

    L = int(task['L'])
    hidden_size = int(task['hidden_size'])
    epochs = int(task['epochs'])
    batch_size = int(task['batch_size'])
    lr = float(task['lr'])
    device_id = int(task['device_id'])
    base_save_dir = Path(task['base_save_dir'])
    dtype = torch.float32

    # assign device
    if torch.cuda.is_available():
        device = torch.device(f'cuda:{device_id}')
        torch.cuda.set_device(device)
    else:
        device = torch.device('cpu')

    # deterministic-ish seed per job
    seed = 1234 + L * 1000 + hidden_size
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Create the model on the worker GPU
    input_features = 1
    output_features = 1
    model = DeepLipschitzSequential(input_features, output_features, (hidden_size,) * L, device=device)

    model = torch.compile(model, mode='reduce-overhead')

    # Create dataset as in sine_training
    x = torch.linspace(-10, 10, 100000 * input_features, device=device, dtype=dtype).reshape((-1, input_features))
    variation = torch.arange(input_features, device=device, dtype=dtype)
    y = torch.sin(x + variation)

    # Timestamped folder per worker run
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    run_dir_name = f'grid_L{L}_h{hidden_size}_gpu{device_id}_{timestamp}'

    run_dir = base_save_dir / run_dir_name
    run_dir.mkdir(parents=True, exist_ok=True)

    # call train (this will save best.pt inside some save_folder)
    try:
        save_folder, best_loss, best_epoch, losses = train(x, y, model,
                                                           learning_prefix=f'grid_search/{run_dir_name}/logs',
                                                           batch_size=batch_size,
                                                           lr=lr,
                                                           termination_error=1e-4,
                                                           epochs=epochs,
                                                           theoretical_lower=0,
                                                           logging_frequency=100)
    except Exception as e:
        # collect exception info and save a small json in run_dir
        err_info = {
            'status': 'failed',
            'exception': repr(e),
            'L': L,
            'hidden_size': hidden_size,
            'device_id': device_id,
            'timestamp': timestamp
        }
        (run_dir / 'error.json').write_text(json.dumps(err_info, indent=2))
        return err_info

    # Save run metadata and losses to .npz for later analysis
    out = {
        'L': L,
        'hidden_size': hidden_size,
        'device_id': device_id,
        'best_loss': None if best_loss is None else float(best_loss),
        'best_epoch': None if best_epoch is None else int(best_epoch),
        'save_folder': save_folder,
    }

    # convert losses into a numpy array and save them
    losses_arr = np.array(losses, dtype=float)
    np.savez_compressed(run_dir / 'results.npz', losses=losses_arr, metadata=out)

    # add a small json for easy reading
    (run_dir / 'metadata.json').write_text(json.dumps(out, indent=2))

    # return a summary dictionary
    summary = dict(out)
    summary['status'] = 'ok'
    summary['run_dir'] = str(run_dir)
    return summary


def grid_search(L_start=3, L_end=15, L_step=2, hidden=(8, 16, 32, 64, 128),
                epochs=20, batch_size=64, lr=1e-4, base_save_dir='../runs/grid_search',
                num_gpus=4):
    """
    Orchestrates a grid search over L in [L_start..L_end] step L_step and hidden sizes.
    Uses up to `num_gpus` processes in parallel (round-robin GPU assignment).
    Results are saved per-run to `base_save_dir`.
    """
    base_save_dir = Path(base_save_dir)
    base_save_dir.mkdir(parents=True, exist_ok=True)

    # Build task list
    L_values = list(range(L_start, L_end + 1, L_step))
    tasks = []
    for L in L_values:
        for h in hidden:
            tasks.append({
                'L': L,
                'hidden_size': h,
                'epochs': epochs,
                'batch_size': batch_size,
                'lr': lr,
                'device_id': None,  # assigned below
                'base_save_dir': str(base_save_dir),
            })

    # assign device ids round-robin
    for idx, t in enumerate(tasks):
        t['device_id'] = idx % num_gpus

    # main index file for quick overview
    index_file = base_save_dir / 'index.json'
    index = {'created': time.strftime('%Y-%m-%d %H:%M:%S'), 'runs': []}

    # Run tasks using ProcessPoolExecutor with max_workers = num_gpus
    futures = {}
    with ProcessPoolExecutor(max_workers=num_gpus) as exe:
        for task in tasks:
            fut = exe.submit(worker_run, task)
            futures[fut] = task

        # collect results as they finish
        for fut in as_completed(futures):
            task = futures[fut]
            try:
                result = fut.result()
            except Exception as e:
                # Shouldn't usually happen because worker_run catches exceptions, but be defensive
                result = {
                    'status': 'exception',
                    'exception': repr(e),
                    'L': task['L'],
                    'hidden_size': task['hidden_size'],
                    'device_id': task['device_id']
                }

            # append to index and save index to disk after each completed run
            index['runs'].append(result)
            index_file.write_text(json.dumps(index, indent=2))

            # print a summary line
            print(
                f"Completed L={task['L']} hidden={task['hidden_size']} on GPU {task['device_id']}: {result.get('status')}")

    print("Grid search finished. Master index saved to:", str(index_file))
    return index_file


if __name__ == '__main__':
    grid_search(L_start=3, L_end=15, L_step=2, hidden=(8, 16, 32, 64, 128), epochs=20,
                batch_size=64, lr=1e-4, base_save_dir='../runs/grid_search', num_gpus=4)
