import random

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.optim as optim
from torch import nn


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


def calculate_lyapunov_spectrum(linear_network, x_data, nle):
    ONSstep = 1

    x = torch.zeros_like(x_data, requires_grad=True, **linear_network.factory_kwargs)

    Q, R = torch.linalg.qr(torch.randn(linear_network.Nin, nle, **linear_network.factory_kwargs))
    ls = torch.zeros(nle, dtype=torch.float32, device=linear_network.device)

    for step, layer in enumerate(linear_network.model):
        layer_out = layer(x)

        # D = torch.autograd.grad([layer_out.sum()], x, retain_graph=True)[0]
        # D = torch.func.jacrev(layer, x)
        D = torch.func.jacfwd(layer)(x[0])
        # print(x.shape, layer_out.shape, D.shape)

        x = layer_out

        Q = D @ Q

        if step % ONSstep == 0 and nle > 0:
            Q, R = torch.linalg.qr(Q)
            ls += torch.log(torch.abs(torch.diag(R))) / ONSstep

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

    # Initialize the RNN
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

    # Set up the initial plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))

    lyapunov_spectrum_initial = calculate_lyapunov_spectrum(linear_network, x_data,
                                                            nle).detach().cpu().numpy()
    ax1.plot(lyapunov_spectrum_initial, "r", label="Lyapunov spectrum before flossing")
    ax1.legend()

    # Training loop
    for epoch in range(Ef):
        optimizer.zero_grad()

        lyapunov_spectrum = calculate_lyapunov_spectrum(linear_network, x_data[len(x_data) // 2:], nle)

        # Calculate the loss (mean square of the first nle Lyapunov exponents)
        loss = torch.mean(lyapunov_spectrum ** 2)
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

    ax1.plot(np.arange(len(lyapunov_spectra)), lyapunov_spectra, "k")
    ax1.set_xlabel(r"Index $i$")
    ax1.set_ylabel(r"Lyapunov Exponent $\lambda_i$ (1/step)")
    ax1.legend()

    ax2.semilogy(list(range(len(losses))), losses, "r", label="Loss")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Loss")
    ax2.set_title("Loss over Epochs")
    ax2.legend()

    fig.savefig('Temp.png')

    print("Flossing complete.")
