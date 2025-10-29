import random

import numpy as np
import torch
import torch.optim as optim
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
        g = jacrev(lambda p: loss_fn(p, x, y))   # gradient wrt params
        _, Hv = jvp(g, (params,), (vparams,))    # directional derivative
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

    step = 0
    loss_all = []
    lyapunov_exponents = []

    batch_maxes = []
    batch_mins = []

    for epoch in range(num_steps):
        running_loss = 0.0
        n_batches = 0

        current_max = []
        current_min = []

        for x, y in dataloader:
            x = x.to(device); y = y.to(device)

            if enable_flossing:
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
                V = Q.T            # (k, D)
                HQ = vmap(hvp_flat)(V)   # (k, D)
                HQ = HQ.T          # (D, k)

                # J = 1 - eta * H
                # J Q = Q - eta * (H @ Q)

                # Jacobian action JQ = Q - eta * (H @ Q)
                JQ = Q - lr * HQ  # differentiable w.r.t. params

                # QR factorization (R captures local expansion rates)
                Q_new, R = torch.linalg.qr(JQ, mode='reduced')  # differentiable

                # Local Lyapunov exponents for this block
                diag_r = R.diagonal().abs().clamp_min(1e-12)
                local_exponents = torch.log(diag_r) / (nstepONS * lr)

                lyapunov_exponents.append(local_exponents.detach())

                current_max.append(local_exponents.max().item())
                current_min.append(local_exponents.min().item())

                curv_pen = curv_coeff * local_exponents.square().sum()

            optimizer.zero_grad(set_to_none=True)
            y_pred = model(x)
            data_loss = criterion(y_pred, y)

            # Only activate after transient (optional, avoids early noise)
            if enable_flossing:
                total_loss = data_loss + (curv_pen if step >= nStepTransient else 0.0)
            else:
                total_loss = data_loss

            total_loss.backward()
            optimizer.step()

            if enable_flossing:
                # Update Q state for the next step WITHOUT keeping graph history
                # Otherwise causes
                with torch.no_grad():
                    Q = Q_new.detach()

            running_loss += data_loss.item()
            n_batches += 1
            step += 1

        loss_all.append(running_loss / max(1, n_batches))


        if enable_flossing:
            max_exponent = max(current_max)
            min_exponent = min(current_min)

            batch_maxes.append(max_exponent)
            batch_mins.append(min_exponent)

            print(f"[epoch {epoch + 1:03d}] loss={loss_all[-1]:.6f} max={max_exponent} min={min_exponent}")
        else:
            print(f"[epoch {epoch + 1:03d}] loss={loss_all[-1]:.6f}")

    return loss_all, lyapunov_exponents, None



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
    losses, top_eigs, traceHs = train_loop(
        linear_network, optimizer, Ef, nle, dataloader, device, 1, enable_flossing=True
    )

    # (Optionally) return/plot arrays here


if __name__ == '__main__':
    main()
