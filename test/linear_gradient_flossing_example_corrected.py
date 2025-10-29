import os
import random

import numpy as np
import torch
import torch.optim as optim
from matplotlib import pyplot as plt
from torch import nn
from torch.func import functional_call, jacrev, jvp, vmap
from torch.utils.data import TensorDataset, DataLoader

DPI = 600


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def tree_flatten(params_dict):
    names, shapes, chunks = [], [], []
    for n, t in params_dict.items():
        names.append(n)
        shapes.append(t.shape)
        chunks.append(t.reshape(-1))
    vec = torch.cat(chunks)
    meta = (names, shapes, [t.numel() for t in params_dict.values()])
    return vec, meta


def tree_unflatten(vec, meta):
    names, shapes, numels = meta
    out, i = {}, 0
    for n, s, m in zip(names, shapes, numels):
        out[n] = vec[i:i + m].reshape(s)
        i += m
    return out


class Linear(nn.Linear):
    def __init__(self, in_features: int, out_features: int, bias: bool = True, device=None, dtype=None):
        super().__init__(in_features, out_features, bias, device, dtype)
        self.activation = nn.ReLU()

    def forward(self, x):
        linear_out = super().forward(x)
        return self.activation(linear_out)


class Network(nn.Module):
    def __init__(self, Nin, n_hidden, hidden_dim, Nout, device=None, dtype=None):
        super().__init__()
        self.dtype = dtype
        self.device = device
        self.factory_kwargs = dict(device=device, dtype=dtype)
        self.Nout = Nout
        self.hidden_dim = hidden_dim
        self.n_hidden = n_hidden
        self.Nin = Nin

        modules = []
        start = Nin

        for _ in range(n_hidden):
            modules.append(Linear(start, hidden_dim, **self.factory_kwargs))
            start = hidden_dim

        modules.append(nn.Linear(start, Nout, **self.factory_kwargs))

        self.model = nn.Sequential(*modules)

    def forward(self, x):
        return self.model(x)

    def __len__(self):
        return len(self.model)


def train_loop(model, optimizer, num_steps, k, dataloader, device,
               nstepONS,
               curv_coeff=1e-3, nStepTransient=100, enable_flossing=False):
    """
    Curvature-regularized training.
    - Uses k-column block Q (state, no grads across steps).
    - JQ = Q - eta * (H @ Q), with H@Q computed via k HVPs (batched by vmap).
    - QR factorization of JQ yields R; local Lyapunov exponents ~ log|diag(R)| / (nstepONS*eta).
    - Penalize positive exponents (pushes toward stable parameter dynamics).

    curv_coeff: strength of the curvature penalty.
    nStepTransient: start regularization after this many steps.
    """

    model.train()
    criterion = nn.MSELoss(reduction='mean')

    # Buffers (usually empty here)
    BUFFERS = dict(model.named_buffers())

    # Pure functional loss(params, x, y)
    def loss_fn(params, x, y):
        logits = functional_call(model, ({**params, **BUFFERS}), (x,))
        return criterion(logits, y)

    # HVP oracle: forward-over-reverse jvp(jacrev(.))
    def hvp_pytree(params, vparams, x, y):
        g = jacrev(lambda p: loss_fn(p, x, y))  # gradient wrt params
        _, Hv = jvp(g, (params,), (vparams,))  # directional derivative
        return Hv

    compiled_hvp = hvp_pytree

    # Live (leaf) params dict; DO NOT detach/clone — we need 3rd-order grads
    def live_params_dict():
        return {k: p for k, p in model.named_parameters()}

    # Build META once (param shapes don't change)
    params0 = live_params_dict()
    flat0, META = tree_flatten(params0)
    D = flat0.numel()

    # Q is persistent state across steps; keep it detached (no history between steps)
    Q, _ = torch.linalg.qr(torch.randn((D, k), device=device))
    Q = Q.detach()

    S = torch.zeros(k, device=device)  # cumulative sum of logs
    tacc = 0.0  # accumulated "time" (eta * steps)
    steps_since_transient = 0

    step = 0
    loss_all = []
    lyapunov_exponents = []

    Qs = []

    for epoch in range(num_steps):
        running_loss = 0.0
        n_batches = 0

        for x, y in dataloader:
            x = x.to(device)
            y = y.to(device)

            if enable_flossing:
                print(f"[epoch {epoch + 1:03d}][step {step:03d}] Lyapunov Exponents")

                # ----- curvature computation for THIS step (graph-contained) -----
                params = live_params_dict()  # live leaves (requires_grad=True)

                # HVP on a flat vector
                def hvp_flat(v_flat_1d):
                    v_tree = tree_unflatten(v_flat_1d, META)
                    Hv_tree = compiled_hvp(params, v_tree, x, y)  # depends on params
                    Hv_flat, _ = tree_flatten(Hv_tree)
                    return Hv_flat

                lr = optimizer.param_groups[0]['lr']

                # Batch HVPs on Q's columns (H @ Q)
                V = Q.T  # (k, D)
                HQ = vmap(hvp_flat)(V)  # (k, D)
                HQ = HQ.T  # (D, k)

                # J = 1 - eta * H
                # J Q = Q - eta * (H @ Q)

                # Jacobian action JQ = Q - eta * (H @ Q)
                JQ_step = Q - lr * HQ  # differentiable w.r.t. params

                with torch.no_grad():
                    Q = JQ_step  # do NOT re-orthonormalize yet
                steps_since_transient += 1

                # only after nstepONS steps (and past transient), orthonormalize once:
                if step >= nStepTransient and (step + 1) % nstepONS == 0:
                    print(f"[epoch {epoch + 1:03d}][step {step:03d}] Normalizing")

                    # QR on the current block-propagated Q
                    Q_new, R = torch.linalg.qr(Q, mode='reduced')
                    diag_r = R.diagonal().abs().clamp_min(1e-12)

                    # accumulate for the asymptotic spectrum
                    S += torch.log(diag_r)
                    tacc += nstepONS * lr

                    # reset basis to the orthonormal Q (do not keep graph)
                    with torch.no_grad():
                        Q = Q_new.detach()
                    steps_since_transient = 0

            optimizer.zero_grad(set_to_none=True)
            y_pred = model(x)
            data_loss = criterion(y_pred, y)

            # Only activate after transient (optional, avoids early noise)
            if enable_flossing:
                total_loss = data_loss + (0.0 if step >= nStepTransient else 0.0)
            else:
                total_loss = data_loss

            total_loss.backward()
            optimizer.step()

            if enable_flossing:
                Qs.append(Q.detach())

            running_loss += data_loss.item()
            n_batches += 1
            step += 1

        if tacc > 0:
            lyapunov_spectrum = (S / tacc).detach().cpu()
        else:
            lyapunov_spectrum = torch.zeros(k)

        lyapunov_exponents.append(lyapunov_spectrum)

        loss_all.append(running_loss / max(1, n_batches))

        if enable_flossing:
            print(
                f"[epoch {epoch + 1:03d}] loss={loss_all[-1]:.6f} max={lyapunov_spectrum.max()} min={lyapunov_spectrum.min()}")
        else:
            print(f"[epoch {epoch + 1:03d}] loss={loss_all[-1]:.6f}")

    lyapunov_exponents = torch.stack(lyapunov_exponents)
    Qs = torch.stack(Qs)

    results = {
        'loss_all': loss_all,
        'lyapunov_exponents': lyapunov_exponents,
        'Qs': Qs,
    }

    return results


def main():
    set_seed(42)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Model parameters
    Nin = 1
    hidden_dim = 64
    n_hidden = 5
    Nout = Nin
    nle = 16  # here used as number of Hutchinson probes
    Ef = 50  # epochs (made smaller for demo)

    result_file = 'results.pt'

    if os.path.exists(result_file):
        results = torch.load(result_file)
    else:
        # Initialize the Neural Network
        linear_network = Network(Nin, n_hidden, hidden_dim, Nout, device=device).to(device)
        optimizer = optim.SGD(linear_network.parameters(), lr=1e-1)

        # Data
        data_size = 10000
        x_data = torch.linspace(-10, 10, data_size, device=device).reshape(data_size, 1)
        y_data = torch.sin(x_data)  # regression target
        dataset = TensorDataset(x_data, y_data)
        dataloader = DataLoader(dataset, batch_size=256, shuffle=True, pin_memory=(device == 'cpu'))

        # Run training with Hessian diagnostics
        results = train_loop(
            linear_network, optimizer, Ef, nle, dataloader, device, 1, enable_flossing=True
        )

        torch.save(results, result_file)

    losses = results['loss_all']
    x_losses = list(range(len(losses)))
    plt.plot(x_losses, losses)
    plt.savefig('losses.png')
    plt.close()

    losses = results['lyapunov_exponents'].cpu()
    x_losses = list(range(len(losses)))
    plt.plot(x_losses, losses)
    plt.savefig('lyapunov_exponents.png')
    plt.close()

    losses = results['batch_maxes']
    x_losses = list(range(len(losses)))
    plt.plot(x_losses, losses)

    losses = results['batch_mins']
    x_losses = list(range(len(losses)))
    plt.plot(x_losses, losses)

    plt.savefig('batch_mins.png')
    plt.close()

    losses = results['lyapunov_exponents'][-1].cpu()
    x_losses = list(range(len(losses)))
    plt.scatter(x_losses, losses)
    plt.savefig('final_lyapunov_exponents.png')
    plt.close()

    lyapunov_exponents = results['lyapunov_exponents'].cpu()
    median_global = torch.median(lyapunov_exponents)
    q1, q3 = torch.quantile(lyapunov_exponents,
                            torch.tensor([.25, .75], device=lyapunov_exponents.device, dtype=lyapunov_exponents.dtype))
    iqr = q3 - q1

    # Robust limit range (±2 IQR around median, typically covers ~95% if symmetric)
    ymin = median_global - 2 * iqr
    ymax = median_global + 2 * iqr

    # For x-limits, handle possible changing dimensionality
    xmax = lyapunov_exponents.shape[1]
    xlim = (0, xmax)

    # ---- Set up figure -----------------------------------------------------------
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.set_xlim(xlim)
    ax.set_ylim(ymin, ymax)
    ax.set_xlabel("Index")
    ax.set_ylabel("Lyapunov exponent")
    ax.grid(True, alpha=0.3)


if __name__ == '__main__':
    main()
