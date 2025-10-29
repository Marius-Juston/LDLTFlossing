import random

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.optim as optim
from torch import nn

DPI = 600


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


class Linear(nn.Linear):
    def __init__(self, in_features: int, out_features: int, bias: bool = True, device=None, dtype=None):
        super().__init__(in_features, out_features, bias, device, dtype)

        self.activation = nn.ReLU()

    def forward(self, x):
        self.x = x

        linear_out = super().forward(x)

        self.out = self.activation(linear_out)

        return self.out


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

        self.max_le = Nin

        modules = []

        start = Nin

        for _ in range(n_hidden):
            modules.append(Linear(start, hidden_dim, **self.factory_kwargs))

            start = hidden_dim
            self.max_le = min(start, self.max_le)

        self.max_le = min(Nout, self.max_le)
        modules.append(nn.Linear(start, Nout, **self.factory_kwargs))

        self.model = nn.Sequential(*modules)

    def forward(self, x):
        return self.model(x)

    def __len__(self):
        return len(self.model)


def calculate_lyapunov_spectrum(linear_network, x_data, nle, n_random_samples: int = None,
                                normalization_frequency: int = 1):
    x = x_data.clone()

    if n_random_samples:
        random_sample_indices = torch.randperm(x.shape[0])[:n_random_samples]
        x = x[random_sample_indices]

    x.requires_grad = True

    batch_size = x.shape[0]

    Q, R = torch.linalg.qr(torch.randn(*x.shape, nle, **linear_network.factory_kwargs))
    ls = torch.zeros(batch_size, nle, dtype=torch.float32, device=linear_network.device)

    for step, layer in enumerate(linear_network.model):
        layer_out, jvp_fn  = torch.func.linearize(layer, x)
        D = jvp_fn(x)

        x = layer_out

        Q = D @ Q

        if step % normalization_frequency == 0 and nle > 0:
            Q, R = torch.linalg.qr(Q)
            ls += torch.log(torch.abs(torch.diagonal(R, dim1=-2, dim2=-1))) / normalization_frequency

    return ls


if __name__ == '__main__':
    set_seed(42)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Model parameters
    Nin = 10
    hidden_dim = 64
    n_hidden = 10
    Nout = 14
    nle = 16  # the maximum number of Lyapunov exponents to floss
    Ef = 350  # number of flossing epochs

    num_sample_trajectories = 300  # Number of sample trajectories to compute the Lyapunov exponents on ( depends on available GPU resources )

    # Initialize the Neural Network
    linear_network = Network(Nin, n_hidden, hidden_dim, Nout, device=device)
    optimizer = optim.Adam(linear_network.parameters())

    nle = min(linear_network.max_le, nle)

    print(f"Computing {nle} Lyapunov Exponents")

    # Generate input data
    pIn = 0.5  # input probability
    data_size = 1000

    x_data = torch.randn(data_size, Nin, device=device)

    # Optimization setup
    losses = []
    lyapunov_spectra = []

    lyapunov_spectrum_initial = calculate_lyapunov_spectrum(linear_network, x_data, nle,
                                                            n_random_samples=num_sample_trajectories).numpy(force=True)

    criteria = nn.MSELoss()

    # Can technically make it so that for the specific Lyapunov exponent is what you want it to be
    target_le = torch.zeros(nle, **linear_network.factory_kwargs)

    target_le = target_le.unsqueeze(0).repeat((num_sample_trajectories, 1))

    # Training loop
    for epoch in range(Ef):
        optimizer.zero_grad()

        lyapunov_spectrum = calculate_lyapunov_spectrum(linear_network, x_data, nle,
                                                        n_random_samples=num_sample_trajectories)

        # Calculate the loss (mean square of the first nle Lyapunov exponents)
        loss = criteria(lyapunov_spectrum, target_le)
        print(f"Epoch {epoch}: Loss = {loss.item()}")

        # Backward pass: compute gradients
        loss.backward()

        # Optimization step
        optimizer.step()

        # Store the loss and Lyapunov spectrum
        losses.append(loss.item())
        lyapunov_spectra.append(lyapunov_spectrum.numpy(force=True))

        if epoch % 10 == 0:
            print(f"Epoch {epoch}: Loss = {loss.item()}")

    lyapunov_spectra = np.array(lyapunov_spectra)

    le_mean = lyapunov_spectra.mean(axis=1)
    le_std = lyapunov_spectra.std(axis=1)
    le_x = np.arange(len(lyapunov_spectra))

    fig_size = (5, 5)

    # Set up the initial plot
    fig, ax = plt.subplots(figsize=fig_size)

    fills = [ax.fill_between(le_x, le_mean[:, i] - le_std[:, i], le_mean[:, i] + le_std[:, i], alpha=0.2) for i in
             range(nle)]
    lines = ax.plot(le_x, le_mean)

    for fill, line in zip(fills, lines):
        color = line.get_color()

        color = mcolors.to_rgb(color)
        fill.set_color(color)

    ax.set_xlabel(r"Index $i$")
    ax.set_ylabel(r"Lyapunov Exponent $\lambda_i$ (1/step)")
    ax.set_title(f"Lyapunov Exponent ($k = {nle}$) over Epochs")
    ax.set_xlim(xmin=0, xmax=len(lyapunov_spectra) + 1)

    fig.tight_layout()

    fig.savefig('LinearGradientFlossing_exponents.png', dpi=DPI)

    fig, ax = plt.subplots(figsize=fig_size)

    ax.semilogy(list(range(len(losses))), losses, "r", label="Loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Loss over Epochs")
    ax.set_xlim(xmin=0, xmax=len(losses) + 1)

    ax.legend()

    fig.tight_layout()

    fig.savefig('LinearGradientFlossing_loss.png', dpi=DPI)

    print("Flossing complete.")
