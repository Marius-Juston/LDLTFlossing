import math
from typing import Tuple

import torch
from torch import nn, Tensor
from torch.nn import init
from torch.nn.parameter import Parameter
from torch.nn.utils import parametrize

from utils import PositiveExp


class AbstractLipschitzLinearLayer(nn.Linear):
    identity: torch.Tensor
    identity_out: torch.Tensor
    constant_sq: torch.Tensor
    constant: torch.Tensor
    R: torch.Tensor
    omega_r: torch.Tensor

    def __init__(self, in_features: int,
                 out_features: int,
                 bias: bool = True,
                 constant_sq: float = 1.0,
                 enable_alpha: bool = True,
                 device=None,
                 dtype=None):
        super(AbstractLipschitzLinearLayer, self).__init__(in_features, out_features, bias=bias, device=device,
                                                           dtype=dtype)

        factory_kwargs = {"device": device, "dtype": dtype}

        self.register_buffer(
            "identity", torch.eye(in_features, **factory_kwargs)
        )

        self.register_buffer(
            "identity_out", torch.eye(out_features, **factory_kwargs)
        )

        self.register_buffer(
            "constant_sq", torch.tensor(constant_sq, **factory_kwargs)
        )

        self.register_buffer(
            "constant", self.constant_sq.sqrt()
        )

        initial_alpha = torch.tensor(1.0, **factory_kwargs)

        if enable_alpha:
            self.alpha = Parameter(initial_alpha)
            parametrize.register_parametrization(self, "alpha", PositiveExp(**factory_kwargs))
        else:
            self.register_buffer('alpha', initial_alpha)

    def reset_parameters(self) -> None:
        # Setting a=sqrt(5) in kaiming_uniform is the same as initializing with
        # uniform(-1/sqrt(in_features), 1/sqrt(in_features)). For details, see
        # https://github.com/pytorch/pytorch/issues/57109
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            init.uniform_(self.bias, -bound, bound)

    def scale_w(self):
        omega = self.sum_weight(transpose=True)
        R = torch.linalg.cholesky(omega, upper=True)

        W = torch.linalg.solve_triangular(R, self.weight, upper=True, left=False)

        return omega, W, R

    def sum_weight(self, transpose: bool = False):
        if transpose:
            return self.weight.T @ self.weight + self.identity * self.alpha
        else:
            return self.weight @ self.weight.T + self.identity_out * self.alpha


class FirstLipschitzLinearLayer(AbstractLipschitzLinearLayer):
    def forward(self, input: Tensor) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
        omega, W, R = self.scale_w()

        W = W * self.constant

        d_inv = omega / (self.constant_sq * self.alpha)
        omega_r = R.T

        return torch.nn.functional.linear(input, W, self.bias), W, d_inv, self.alpha, omega_r, self.constant


class InnerWeightLipschitzLinearLayer(AbstractLipschitzLinearLayer):
    def forward(self, input: Tensor, prev_alpha: Tensor, prev_omega_r: Tensor, prev_constant: Tensor) -> \
            Tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
        _, W, _ = self.scale_w()

        W = torch.linalg.solve_triangular(prev_omega_r, W, upper=False, left=False)

        scale = prev_alpha.sqrt() * prev_constant * self.constant
        div_scale = self.constant_sq * self.alpha

        W = scale * W

        omega = self.sum_weight()

        d_inv = omega / div_scale

        omega_r = torch.linalg.cholesky(omega, upper=False)

        return torch.nn.functional.linear(input, W, self.bias), W, d_inv, self.alpha, omega_r, self.constant


class LastLipschitzLinearLayer(AbstractLipschitzLinearLayer):
    def forward(self, input: Tensor, sigma_r: Tensor, prev_alpha: Tensor, prev_omega_r: Tensor) -> Tuple[
        Tensor, Tensor]:
        _, W, _ = self.scale_w()

        prev_alpha = prev_alpha
        prev_r = prev_omega_r

        W = torch.linalg.solve_triangular(prev_r, W, upper=False, left=False)
        W = self.constant * prev_alpha.sqrt() * torch.linalg.solve_triangular(sigma_r, W, upper=True, left=True)

        return torch.nn.functional.linear(input, W, self.bias), W


class LinearLipschitz(AbstractLipschitzLinearLayer):
    sqrt_2: Tensor

    def __init__(self, in_features: int, out_features: int, activation=None, bias: bool = True,
                 constant_sq: float = 1.0,
                 enable_alpha=True,
                 device=None, dtype=None):
        super().__init__(in_features, out_features, bias, constant_sq, enable_alpha, device, dtype)
        self.register_buffer('sqrt_2', torch.tensor(2, device=device, dtype=dtype).sqrt())

        self.activation = activation

    def forward(self, input: Tensor, prev_omega_r: Tensor, prev_alpha: Tensor, pre_constant: Tensor) -> Tuple[
        Tensor, Tensor, Tensor]:
        _, W, _ = self.scale_w()

        W = torch.linalg.solve_triangular(prev_omega_r, W, upper=False, left=False)

        scale = prev_alpha.sqrt() * pre_constant * self.constant

        W = scale * W

        omega = self.sum_weight()

        omega_r = torch.linalg.cholesky(omega, upper=False)

        out = torch.nn.functional.linear(input, W, self.bias)

        if self.activation:
            out = self.activation(out)

        return out, omega_r, W
