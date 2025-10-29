import random

import numpy as np
import torch
import torch.optim as optim
from torch import nn
from torch.func import functional_call, jacrev, jvp
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


def train_loop(model, optimizer, num_steps, k, dataloader, device, learning_rate, nstepONS):
    """
    Adds scalable Hessian integration:
      - HVP via forward-over-reverse: jvp(jacrev(loss))(params)[v]
      - Power iteration for the largest eigenvalue (spectral norm of Hessian)
      - Hutchinson trace estimator (batched vmap of HVPs)

    k is used as the number of Hutchinson probes.
    """
    model.train()
    criterion = nn.MSELoss(reduction='mean')

    # --- Buffers once (e.g., BatchNorm running stats; usually empty here) ---
    BUFFERS = dict(model.named_buffers())

    # --- Pure functional loss(params, x, y) ---
    def loss_fn(params, x, y):
        # functional_call takes a mapping of parameter & buffer names -> tensors
        logits = functional_call(model, ({**params, **BUFFERS}), (x,))
        return criterion(logits, y)

    # --- Hessian-vector product oracle: jvp(jacrev(loss)) ---
    def hvp_pytree(params, vparams, x, y):
        # jacrev(loss_fn) returns gradient wrt params; jvp gives directional derivative
        grad_fn = jacrev(lambda p: loss_fn(p, x, y))
        _, Hv = jvp(grad_fn, (params,), (vparams,))
        return Hv

    # Compile for repeated use (PyTorch 2.9 has faster/steadier compile)
    compiled_loss = torch.compile(loss_fn, fullgraph=True)
    compiled_hvp = torch.compile(hvp_pytree, fullgraph=True)

    # ---- helpers: flat <-> pytree on a reference param dict ----
    def current_params_dict():
        # clone & require grad so the functional AD sees leaves
        return {k: p.detach().clone().requires_grad_(True) for k, p in model.named_parameters()}

    params = current_params_dict()  # freeze a view of current params
    flat_template, META = tree_flatten(params)

    D = flat_template.numel()

    Q, _ = torch.linalg.qr(torch.randn((D, k), device=device))

    I = torch.ones(D, device=device)

    # --- Arrays to log ---
    loss_All = np.zeros(num_steps, dtype=np.float64)
    top_eig_All = np.zeros(num_steps, dtype=np.float64)
    traceH_All = np.zeros(num_steps, dtype=np.float64)

    nStepTransient = 100
    LS = torch.zeros(k, device=device)
    LSall = []
    loss_All = []
    normdhAll = []
    lsidx = -1
    total_time_for_exponents = 0

    step = 0

    # --- Training epochs ---
    for epoch in range(num_steps):

        running_loss = 0.0
        n_batches = 0

        for x, y in dataloader:
            # Standard training step
            optimizer.zero_grad(set_to_none=True)

            x = x.to(device)
            y = y.to(device)

            # ----- Curvature diagnostics at the END of the epoch -----
            params = current_params_dict()  # freeze a view of current params
            flat_template, META = tree_flatten(params)

            def hvp_flat(v_flat):
                v_tree = tree_unflatten(v_flat, META)
                Hv_tree = compiled_hvp(params, v_tree, x, y)
                Hv_flat, _ = tree_flatten(Hv_tree)
                return Hv_flat

            lyapunov_exponents = torch.zeros(k, device=device)

            if step >= nStepTransient:
                print(f"[epoch {epoch + 1:03d}][step {step + 1:03d}] Computing the Hessian")
                hessian = hvp_flat(flat_template)

                J = I - learning_rate * hessian

                Q = torch.diag(J) @ Q

                if (step + 1) % nstepONS == 0:
                    print(f"[epoch {epoch + 1:03d}][step {step + 1:03d}] Renormalizing")
                    lsidx += 1
                    Q, r = torch.linalg.qr(Q)

                    diag_r = torch.abs(torch.diag(r))
                    diag_r[diag_r == 0] = 1e-16
                    local_exponents = torch.log(diag_r) / (nstepONS * learning_rate)
                    LSall.append(local_exponents.detach())
                    LS += torch.log(diag_r)
                    total_time_for_exponents += nstepONS * learning_rate

                    lyapunov_exponents = local_exponents


            y_pred = model(x)

            loss = criterion(y_pred, y)

            lyapunov_loss = lyapunov_exponents.square().sum()

            total_loss = loss + lyapunov_loss

            total_loss.backward()
            optimizer.step()

            running_loss += loss.item()
            n_batches += 1

            step += 1

        loss_All.append( running_loss / max(1, n_batches))

        print(f"[epoch {epoch + 1:03d}] loss={loss_All[epoch]:.6f}")

    return loss_All, top_eig_All, traceH_All


def main():
    set_seed(42)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Model parameters
    Nin = 10
    hidden_dim = 64
    n_hidden = 10
    Nout = Nin
    nle = 16  # here used as number of Hutchinson probes
    Ef = 50  # epochs (made smaller for demo)

    # Initialize the Neural Network
    linear_network = Network(Nin, n_hidden, hidden_dim, Nout, device=device).to(device)
    optimizer = optim.SGD(linear_network.parameters(), lr=1e-2)

    # Data
    data_size = 2048
    x_data = torch.randn(data_size, Nin, device=device)
    y_data = torch.sin(x_data)  # regression target
    dataset = TensorDataset(x_data, y_data)
    dataloader = DataLoader(dataset, batch_size=256, shuffle=True, pin_memory=(device == 'cpu'))

    # Run training with Hessian diagnostics
    losses, top_eigs, traceHs = train_loop(
        linear_network, optimizer, Ef, nle, dataloader, device, 0.01, 1
    )

    # (Optionally) return/plot arrays here


if __name__ == '__main__':
    main()
