from typing import Union, List, Sequence, Tuple, Optional

import torch
from torch import nn, Tensor
from torch.nn import ModuleList, Module, ReLU

from models.linear_layers import FirstLipschitzLinearLayer, LastLipschitzLinearLayer, InnerWeightLipschitzLinearLayer, \
    LinearLipschitz
from utils import safe_cholesky


class DeepLipschitzLinearResNet(nn.Module):
    A: FirstLipschitzLinearLayer
    B: LastLipschitzLinearLayer
    layers: ModuleList

    def __init__(self, in_features: int, out_features: int, layers_widths: Sequence[int],
                 activation: Union[Module, List[Module]] = None, lipschitz_constant: float = 1.0,
                 enable_alpha: bool = True,
                 use_safe_cholesky: bool = False,
                 device: Union[str, torch.device] = 'cuda', dtype=None) -> None:
        super().__init__()

        assert len(layers_widths) >= 1, "This deep network assumes that the number of layers is greater "
        assert in_features > 0, "The input width needs to be larger than 0"
        assert out_features > 0, "The output width needs to be larger than 0"

        self.cholesky_function = safe_cholesky if use_safe_cholesky else torch.linalg.cholesky
        self.enable_alpha = enable_alpha
        self.lipschitz_constant = lipschitz_constant
        self.lipschitz_constant_sq = lipschitz_constant * lipschitz_constant

        self.out_features = out_features
        self.in_features = in_features
        self.layer_widths = layers_widths

        self.dtype = dtype
        self.device = device

        self.factory_kwargs = {"device": device, "dtype": dtype}

        if activation is None:
            self.activation = ReLU()
        else:
            self.activation = activation

        self.generate_layers()

    def reset_parameters(self) -> None:
        # Find a better initialization scheme for the current formulation of the system
        self.A.reset_parameters()
        self.B.reset_parameters()

        for layer in self.layers:
            layer.reset_parameters()

    def generate_layers(self) -> None:
        self.A = FirstLipschitzLinearLayer(in_features=self.in_features, out_features=self.out_features,
                                           constant_sq=self.lipschitz_constant_sq, enable_alpha=self.enable_alpha,
                                           **self.factory_kwargs)

        self.B = LastLipschitzLinearLayer(in_features=self.layer_widths[-1], out_features=self.out_features,
                                          bias=False,
                                          enable_alpha=self.enable_alpha,
                                          # Don't need bias on both A and B since they would be doing the same thing
                                          **self.factory_kwargs)

        layer = []

        prev = self.in_features

        for i, out_features in enumerate(self.layer_widths):
            layer.append(
                InnerWeightLipschitzLinearLayer(prev, out_features,
                                                constant_sq=2.0,
                                                enable_alpha=self.enable_alpha,
                                                **self.factory_kwargs))
            prev = out_features

        self.layers = ModuleList(layer)

        # self.spectral_norm = SpectralNorm(self.B.identity)

    def compute_b(self, x: Tensor, sigma_lower: Tensor, a_weight: Tensor, prev_alpha: Tensor, prev_omega_r: Tensor):
        # inner_inverse = torch.cholesky_inverse(prev_r, upper=True)
        # max_singular = torch.sqrt(2 * self.spectral_norm(inner_inverse))

        left = a_weight @ sigma_lower

        scaling = self.B.identity_out + left @ left.T

        R_sigma = self.cholesky_function(scaling, upper=True)

        second, b_weight = self.B(x, R_sigma, prev_alpha, prev_omega_r)

        return second, b_weight

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        first, a_weight, d_inv, prev_alpha, prev_omega_r, prev_constant = self.A(x)

        gamma = self.A.identity
        sigma_lower = None

        current_input = x

        # for i, layer in enumerate(self.layers):
        for layer in self.layers:
            T = gamma @ d_inv

            # d_inv_prev = d_inv

            current_input, c_weight, d_inv, prev_alpha, prev_omega_r, prev_constant = layer(current_input, prev_alpha,
                                                                                            prev_omega_r, prev_constant)

            # with torch.no_grad():
            #     R_inv = torch.linalg.solve_triangular(prev_layer.omega_r, layer.identity, upper=True)
            #
            #     scale = prev_layer.alpha.sqrt() * prev_layer.constant
            #
            #     temp = scale * R_inv
            #     result = temp.T @ d_inv_prev @ temp
            #
            #     assert torch.allclose(layer.identity, result)

            # with torch.no_grad():
            #     val = c_weight @ d_inv_prev @ c_weight.T
            #     eigenvals = torch.linalg.eigvalsh(val)
            #     max_val = eigenvals.max().item()
            #     min_val = eigenvals.min().item()
            #
            #     print(i, max_val <= 2.0, max_val, min_val)

            if sigma_lower is None:
                sigma_lower = T
            else:
                # Instead of using custom CholeskyRankK update this is computationally much faster
                sigma_lower = safe_cholesky(sigma_lower @ sigma_lower.T + T @ T.T, upper=False)
                # sigma_lower = cholesky_rank_k(sigma_lower, T)

            gamma = T @ c_weight.T

            current_input = self.activation(current_input)

        second, b_weight = self.compute_b(current_input, sigma_lower, a_weight, prev_alpha, prev_omega_r)

        return first + second


class DeepLipschitzSequential(nn.Module):
    layers: ModuleList

    def __init__(self, in_features: int, out_features: int, layers_widths: Sequence[int],
                 activation: Union[Module, List[Module]] = None, lipschitz_constant: float = 1.0,
                 dropout=False,
                 device: Union[str, torch.device] = 'cuda', dtype=None) -> None:
        super().__init__()

        self.dropout = dropout
        assert len(layers_widths) >= 1, "This deep network assumes that the number of layers is greater "
        assert in_features > 0, "The input width needs to be larger than 0"
        assert out_features > 0, "The output width needs to be larger than 0"

        self.lipschitz_constant = lipschitz_constant
        self.lipschitz_constant_sq = lipschitz_constant * lipschitz_constant

        self.out_features = out_features
        self.in_features = in_features
        self.layer_widths = layers_widths

        self.dtype = dtype
        self.device = device

        self.factory_kwargs = {"device": device, "dtype": dtype}

        if activation is None:
            self.activation = ReLU()
        else:
            self.activation = activation

        self.generate_layers()

    def reset_parameters(self) -> None:
        for layer in self.layers:
            layer.reset_parameters()

    def generate_layers(self) -> None:
        layer = []

        prev = self.in_features

        for i, out_features in enumerate(self.layer_widths):
            activation = self.activation
            constant = 2.0

            layer.append(
                LinearLipschitz(prev, out_features, constant_sq=constant, activation=activation, **self.factory_kwargs))

            if self.dropout:
                layer.append(nn.Dropout())

            prev = out_features

        layer.append(
            LinearLipschitz(prev, self.out_features, constant_sq=1.0, activation=None, **self.factory_kwargs))

        self.layers = ModuleList(layer)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        current_input = x
        d_sqrt = self.lipschitz_constant * self.layers[0].identity
        prev_alpha = torch.tensor(1.0, **self.factory_kwargs)
        prev_const = torch.tensor(1.0, **self.factory_kwargs)

        for layer in self.layers:
            if isinstance(layer, nn.Dropout):
                current_input = layer(current_input)
            else:
                layer: LinearLipschitz

                current_input, d_sqrt, _ = layer(current_input, d_sqrt, prev_alpha, prev_const)

                prev_const = layer.constant
                prev_alpha = layer.alpha

        return current_input

    def last_weight(self) -> Tuple[torch.Tensor, torch.Tensor]:
        x = torch.zeros((1, self.in_features), **self.factory_kwargs)

        d_sqrt = self.lipschitz_constant * self.layers[0].identity
        prev_alpha = torch.tensor(1.0, **self.factory_kwargs)
        prev_const = torch.tensor(1.0, **self.factory_kwargs)

        for layer in self.layers:
            if isinstance(layer, nn.Dropout):
                x = layer(x)
            else:
                layer: LinearLipschitz
                x, d_sqrt, W = layer(x, d_sqrt, prev_alpha, prev_const)

                prev_const = layer.constant
                prev_alpha = layer.alpha

        return W

    def calculate_lyapunov_spectrum(self, x_data, nle, n_random_samples: int = None,
                                    normalization_frequency: int = None):

        if normalization_frequency is None:
            # Normalize only once at the end
            normalization_frequency = len(self.layers) - 1

        if n_random_samples and n_random_samples < x_data.shape[0]:
            random_sample_indices = torch.randperm(x_data.shape[0])[:n_random_samples]
            x_data = x_data[random_sample_indices]

        batch_size = x_data.shape[0]

        Q, R = torch.linalg.qr(torch.randn(*x_data.shape, nle, **self.factory_kwargs))
        ls = torch.zeros(batch_size, nle, dtype=torch.float32, device=self.device)

        current_input = x_data
        d_sqrt = self.lipschitz_constant * self.layers[0].identity
        prev_alpha = torch.tensor(1.0, **self.factory_kwargs)
        prev_const = torch.tensor(1.0, **self.factory_kwargs)

        for step, layer in enumerate(self.layers):
            current_input_for_jac = current_input.requires_grad_(True)

            if isinstance(layer, nn.Dropout):
                layer_out = layer(current_input)

                # FIXME: Very inefficient, but ¯\_(ツ)_/¯
                D = torch.func.vmap(torch.func.jacfwd(layer))(current_input_for_jac)
            else:
                layer: LinearLipschitz

                # FIXME: Very inefficient, but ¯\_(ツ)_/¯
                D, *_ = torch.func.vmap(
                    torch.func.jacfwd(lambda inp, d, a, c: layer(inp, d, a, c)[0]),
                    in_dims=(0, None, None, None)
                )(current_input_for_jac, d_sqrt, prev_alpha, prev_const)

                layer_out, d_sqrt, _ = layer(current_input, d_sqrt, prev_alpha, prev_const)

                prev_const = layer.constant
                prev_alpha = layer.alpha

            current_input = layer_out

            Q = D @ Q

            if step % normalization_frequency == 0 and nle > 0:
                Q, R = torch.linalg.qr(Q)
                ls += torch.log(torch.abs(torch.diagonal(R, dim1=-2, dim2=-1))) / normalization_frequency

        return ls, current_input


class DeepLipschitzSequentialStack(nn.Module):
    def __init__(self, in_features: int, out_features: int, num_layers: int, num_interior_layers: int, num_hidden: int,
                 activation: Union[Module, List[Module]] = None, interior_activation: Optional[Module] = None,
                 lipschitz_constant: float = 1.0,
                 dropout=False,
                 device: Union[str, torch.device] = 'cuda', dtype=None):
        super().__init__()

        self.interior_activation = interior_activation
        self.num_hidden = num_hidden
        self.num_interior_layers = num_interior_layers
        self.num_layers = num_layers
        assert in_features > 0, "The input width needs to be larger than 0"
        assert out_features > 0, "The output width needs to be larger than 0"

        self.dropout = dropout
        self.lipschitz_constant = lipschitz_constant
        self.lipschitz_constant_sq = lipschitz_constant * lipschitz_constant

        self.out_features = out_features
        self.in_features = in_features

        self.dtype = dtype
        self.device = device

        self.factory_kwargs = {"device": device, "dtype": dtype}

        if activation is None:
            self.activation = ReLU()
        else:
            self.activation = activation

        self.generate_layers()

    def generate_layers(self):
        model = []

        prev = self.in_features

        for i in range(self.num_layers - 1):
            model.append(DeepLipschitzSequential(prev, self.num_hidden, (self.num_hidden,) * self.num_interior_layers,
                                                 activation=self.activation, lipschitz_constant=self.lipschitz_constant,
                                                 dropout=self.dropout, **self.factory_kwargs))

            if self.interior_activation is not None:
                model.append(self.interior_activation)

            prev = self.num_hidden

        model.append(DeepLipschitzSequential(prev, self.out_features, (self.num_hidden,) * self.num_interior_layers,
                                             activation=self.activation, lipschitz_constant=self.lipschitz_constant,
                                             dropout=self.dropout, **self.factory_kwargs))

        self.layers = nn.Sequential(*model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)
