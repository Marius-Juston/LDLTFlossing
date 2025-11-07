"""Reference implementation of curvature-regularized gradient flossing."""

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
    """Seed Python, NumPy, and PyTorch RNGs for reproducibility."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def tree_flatten(params_dict):
    """Flatten a parameter dict into a single vector plus metadata."""
    names, shapes, chunks = [], [], []
    for n, t in params_dict.items():
        names.append(n)
        shapes.append(t.shape)
        chunks.append(t.reshape(-1))
    vec = torch.cat(chunks)
    meta = (names, shapes, [t.numel() for t in params_dict.values()])
    return vec, meta


def tree_unflatten(vec, meta):
    """Recreate a parameter dict from :func:`tree_flatten` metadata."""
    names, shapes, numels = meta
    out, i = {}, 0
    for n, s, m in zip(names, shapes, numels):
        out[n] = vec[i:i + m].reshape(s)
        i += m
    return out


class Linear(nn.Linear):
    """Linear layer with a ReLU activation for the corrected demo."""
    def __init__(self, in_features: int, out_features: int, bias: bool = True, device=None, dtype=None):
        super().__init__(in_features, out_features, bias, device, dtype)
        self.activation = nn.ReLU()

    def forward(self, x):
        """Forward pass consisting of ``Wx + b`` followed by ReLU."""
        linear_out = super().forward(x)
        return self.activation(linear_out)


class Network(nn.Module):
    """Stack of linear layers tracked for Hessian/ONS diagnostics."""
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
        """Apply the sequential model."""
        return self.model(x)

    def __len__(self):
        return len(self.model)


def train_loop(model, optimizer, num_steps, k, dataloader, device,
               nstepONS,
               curv_coeff=1e-3, nStepTransient=100, enable_flossing=False, train_flossing=False):
    """Train with curvature regularization and optional Lyapunov penalties.

    Args:
        model: Network under training.
        optimizer: Optimizer used for parameter updates.
        num_steps: Number of epochs (outer loops) to execute.
        k: Number of Lyapunov directions to track.
        dataloader: Provides training batches.
        device: Device where computations run.
        nstepONS: Frequency of orthogonalization.
        curv_coeff: Strength of the curvature penalty term.
        nStepTransient: Number of steps to skip before enabling flossing.
        enable_flossing: Whether to evaluate Lyapunov exponents.
        train_flossing: Whether Lyapunov losses contribute gradients.

    Returns:
        dict: Aggregated losses, Lyapunov spectra, and intermediate states.
    """

    model.train()
    criterion = nn.MSELoss(reduction='mean')

    BUFFERS = dict(model.named_buffers())

    def loss_fn(params, x, y):
        logits = functional_call(model, ({**params, **BUFFERS}), (x,))
        return criterion(logits, y)

    def hvp_pytree(params, vparams, x, y):
        g = jacrev(lambda p: loss_fn(p, x, y))
        _, Hv = jvp(g, (params,), (vparams,))
        return Hv

    compiled_hvp = hvp_pytree

    def live_params_dict():
        return {k: p for k, p in model.named_parameters()}

    params0 = live_params_dict()
    flat0, META = tree_flatten(params0)
    D = flat0.numel()

    Q, _ = torch.linalg.qr(torch.randn((D, k), device=device))
    Q = Q.detach()

    S = torch.zeros(k, device=device)
    tacc = 0.0
    steps_since_transient = 0

    step = 0
    loss_all = []
    loss_lyapunov = []
    loss_data = []
    lyapunov_exponents = []

    Qs = []

    for epoch in range(num_steps):
        running_loss = 0.0
        running_loss_lyapunov = 0.0
        running_loss_data = 0.0
        n_batches = 0

        for x, y in dataloader:
            x = x.to(device)
            y = y.to(device)

            if enable_flossing:
                # print(f"[epoch {epoch + 1:03d}][step {step:03d}] Lyapunov Exponents")

                params = live_params_dict()

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

                if train_flossing:
                    Q = JQ_step
                else:
                    with torch.no_grad():
                        Q = JQ_step

                steps_since_transient += 1

                if step >= nStepTransient and (step + 1) % nstepONS == 0:
                    print(f"[epoch {epoch + 1:03d}][step {step:03d}] Normalizing")

                    Q_new, R = torch.linalg.qr(Q, mode='reduced')
                    diag_r = R.diagonal().abs().clamp_min(1e-12)

                    S += torch.log(diag_r)
                    tacc += nstepONS * lr

                    with torch.no_grad():
                        Q = Q_new.detach()
                    steps_since_transient = 0

                    if train_flossing:
                        lyapunov_spectrum = (S / tacc)
                        lyapunov_loss = lyapunov_spectrum.square().mean()
                    else:
                        lyapunov_loss = torch.tensor(0.0, device=device)
                else:
                    lyapunov_loss = torch.tensor(0.0, device=device)

            optimizer.zero_grad(set_to_none=False)
            y_pred = model(x)
            data_loss = criterion(y_pred, y)

            # Only activate after transient (optional, avoids early noise)
            if enable_flossing:
                # FIXME currently not actually training the gradients

                total_loss = data_loss + curv_coeff * (lyapunov_loss if step >= nStepTransient else 0.0)
            else:
                total_loss = data_loss

            total_loss.backward(retain_graph=True)
            optimizer.step()

            if enable_flossing:
                Qs.append(Q.detach().cpu())

                # Detach gradient graph
                with torch.no_grad():
                    Q = Q.detach()

            running_loss += data_loss.item()
            running_loss_lyapunov += lyapunov_loss.item()
            running_loss_data += data_loss.item()

            n_batches += 1
            step += 1

        if tacc > 0:
            lyapunov_spectrum = (S / tacc).detach().cpu()
        else:
            lyapunov_spectrum = torch.zeros(k)

        lyapunov_exponents.append(lyapunov_spectrum.detach().cpu())

        loss_all.append(running_loss / max(1, n_batches))
        loss_data.append(running_loss_data / max(1, n_batches))
        loss_lyapunov.append(running_loss_lyapunov / max(1, n_batches))

        if enable_flossing:
            if train_flossing:
                print(
                    f"[epoch {epoch + 1:03d}] loss={loss_all[-1]:.6f} loss_d={loss_data[-1]:.6f} loss_l={loss_lyapunov[-1]:.6f} max={lyapunov_spectrum.max()} min={lyapunov_spectrum.min()}")
            else:
                print(
                    f"[epoch {epoch + 1:03d}] loss={loss_all[-1]:.6f} loss_d={loss_data[-1]:.6f} max={lyapunov_spectrum.max()} min={lyapunov_spectrum.min()}")
        else:
            print(f"[epoch {epoch + 1:03d}] loss={loss_all[-1]:.6f}")

    if len(lyapunov_exponents)> 0:
        lyapunov_exponents = torch.stack(lyapunov_exponents)

    if len(Qs) > 0:
        Qs = torch.stack(Qs)

    results = {
        'loss_all': loss_all,
        'loss_data': loss_data,
        'loss_lyapunov': loss_lyapunov,
        'lyapunov_exponents': lyapunov_exponents,
        'Qs': Qs,
    }

    return results


def main():
    """Entry point for reproducing the corrected linear flossing experiment."""
    set_seed(42)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Model parameters
    Nin = 1
    hidden_dim = 64
    n_hidden = 5
    Nout = Nin
    nle = 16  # here used as number of Hutchinson probes
    Ef = 100  # epochs (made smaller for demo)

    sample_division = 10

    batch_size = 64

    result_file = 'results.pt'

    if os.path.exists(result_file):
        results = torch.load(result_file)
    else:
        # Initialize the Neural Network
        linear_network = Network(Nin, n_hidden, hidden_dim, Nout, device=device).to(device)
        optimizer = optim.SGD(linear_network.parameters(), lr=1e-2)

        # Data
        data_size = 10000
        x_data = torch.linspace(-10, 10, data_size, device=device).reshape(data_size, 1)
        y_data = torch.sin(x_data)  # regression target
        dataset = TensorDataset(x_data, y_data)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, pin_memory=(device == 'cpu'))

        nstepONS = len(dataset) // (batch_size * sample_division)

        print(nstepONS)

        # Run training with Hessian diagnostics
        results = train_loop(
            linear_network, optimizer, Ef, nle, dataloader, device, nstepONS, enable_flossing=True, train_flossing=False
        )

        torch.save(results, result_file)

    losses = results['loss_all']
    x_losses = list(range(len(losses)))
    plt.plot(x_losses, losses)
    plt.title("Training Loss vs Epoch")
    plt.xlabel("Epoch")
    plt.ylabel("Training Loss")
    plt.savefig('losses.png')
    plt.close()

    losses = results['lyapunov_exponents'].cpu()
    x_losses = list(range(len(losses)))
    plt.plot(x_losses, losses)
    plt.title("Lyapunov Exponents vs Epoch")
    plt.xlabel("Epoch")
    plt.ylabel("Lyapunov Exponents")
    plt.savefig('lyapunov_exponents.png')
    plt.close()

    losses = results['lyapunov_exponents'][-1].cpu()
    x_losses = np.arange(len(losses)) + 1
    plt.scatter(x_losses, losses)
    plt.plot(x_losses, np.zeros_like(x_losses), linestyle='dashed')
    plt.title("Final Lyapunov Exponents")
    plt.xlabel("Lyapunov Exponent index")
    plt.ylabel("Lyapunov Exponent")
    plt.savefig('final_lyapunov_exponents.png')
    plt.close()


if __name__ == '__main__':
    main()
