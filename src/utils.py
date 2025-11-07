"""Linear-model utility functions and custom autograd helpers.

This module hosts numerically robust matrix primitives, custom autograd
operators, and helper layers used across the gradient-flossing experiments.
Every function follows the same expectations: operate on PyTorch tensors,
avoid silent device transfers, and preserve differentiability whenever
possible.
"""

import logging
import traceback
import types
import warnings
from typing import Tuple, Optional, Any, Iterable, Union

import torch
from torch import Tensor, cholesky_inverse
from torch.autograd import Function
from torch.linalg import eigh


def _project_to_spd(A: Tensor, eps: float) -> Tensor:
    """Project a matrix or batch of matrices onto the SPD cone.

    Args:
        A: Tensor with shape ``(..., n, n)``.
        eps: Minimum eigenvalue retained during the projection.

    Returns:
        Tensor: Symmetric positive definite tensor with the same leading
        dimensions as ``A``.
    """
    A32 = A.to(torch.float32)
    A_sym = (A32 + A32.mT) / 2
    eigvals, eigvecs = torch.linalg.eigh(A_sym)
    eigvals = torch.clamp(eigvals, min=eps)
    return eigvecs @ torch.diag_embed(eigvals) @ eigvecs.mT


@torch.autocast(device_type='cuda', enabled=False)
def safe_cholesky(A: Tensor, upper: bool = False, eps: float = 1e-4, max_tries: int = 5) -> Tensor:
    """Retry ``torch.linalg.cholesky_ex`` with spectral projections and jitter.

    Args:
        A: Hermitian positive semidefinite tensor with shape ``(..., n, n)``.
        upper: Whether to return an upper-triangular factor.
        eps: Minimum eigenvalue retained during spectral projection.
        max_tries: Number of jitter trials attempted before raising.

    Returns:
        Tensor: Triangular factor with the same dtype as ``A``.

    Raises:
        ValueError: If no stabilization strategy succeeds.
    """
    A32 = A.to(torch.float32)
    res, info = torch.linalg.cholesky_ex(A32, upper=upper)
    if not info.any():
        return res.to(A.dtype)

    logging.warning("Cholesky failed, retrying with spectral projection.")
    projected = _project_to_spd(A32, eps)
    res, info = torch.linalg.cholesky_ex(projected, upper=upper)
    if not info.any():
        return res.to(A.dtype)

    identity = torch.eye(A32.shape[-1], dtype=A32.dtype, device=A32.device)
    for attempt in range(max_tries):
        jitter = eps * (10 ** attempt)
        try:
            chol = torch.linalg.cholesky(projected + jitter * identity, upper=upper)
            return chol.to(A.dtype)
        except RuntimeError:
            logging.error(
                "Cholesky retry %s failed with jitter %.2e:\n%s",
                attempt,
                jitter,
                traceback.format_exc(),
            )

    raise ValueError("Unable to produce a numerically stable Cholesky factor.")


@torch.autocast(device_type='cuda', enabled=False)
def amp_solve_triangular(A: Tensor, B: Tensor, **kwargs) -> Tensor:
    """Solve a triangular system without enabling AMP for stability.

    Args:
        A: Triangular tensor with shape ``(..., n, n)``.
        B: Right-hand side tensor with shape ``(..., n, k)``.
        **kwargs: Keyword arguments forwarded to
            :func:`torch.linalg.solve_triangular` (``upper``, ``left``, etc.).

    Returns:
        Tensor: Solution tensor cast back to ``A``'s dtype.
    """
    A32 = A.to(torch.float32)
    B32 = B.to(torch.float32)

    C = torch.linalg.solve_triangular(A32, B32, **kwargs)

    return C.to(A.dtype)


# @torch.compile
def matrix_inv_square_root(x: Tensor) -> Tensor:
    """Return the symmetric inverse square root of ``x``.

    The input is detached before the eigen decomposition so gradients do not
    explode near zero eigenvalues. This is sufficient for all current call
    sites that only need the value, not the gradient, of the inverse square
    root.

    Args:
        x: Hermitian positive definite tensor with shape ``(..., n, n)``.

    Returns:
        Tensor: Symmetric inverse square root ``S`` such that
        ``S @ S`` approximates ``x.inv()``.
    """
    eigvals, eigvecs = eigh(x.detach())
    return (eigvecs * eigvals.rsqrt().unsqueeze(0)) @ eigvecs.T


def matrix_inv_square_root_triangle(x: Tensor, B: Optional[Tensor] = None, upper: bool = False,
                                    left: bool = True) -> Tensor:
    """Return a triangular inverse square root obtained via Cholesky.

    Unlike :func:`matrix_inv_square_root`, this helper keeps the triangular
    factor returned by :func:`torch.linalg.cholesky_ex`. The result therefore is
    not symmetric but is substantially cheaper to compute and differentiable by
    construction.

    Args:
        x: Hermitian positive definite tensor with shape ``(..., n, n)``.
        B: Optional right-hand side with shape ``(..., n, m)``. When ``None``
            the routine solves ``L @ X = I`` and returns the inverse factor.
        upper: Whether to treat the factor as an upper triangular matrix.
        left: Whether to solve ``L @ X = B`` (``True``) or ``X @ L = B``.

    Returns:
        Tensor: Triangular inverse square root ``R`` such that
        ``R.transpose(-1, -2) @ R`` approximates ``x.inv()``.
    """
    L = torch.linalg.cholesky_ex(x, upper=upper).L

    if B is None:
        B = torch.eye(L.shape[0], dtype=x.dtype, device=x.device)
    L_inv = torch.linalg.solve_triangular(L, B, upper=upper, left=left)
    return L_inv


def symmetrize_(x: Tensor) -> Tensor:
    """Return ``0.5 * (x + x.T)`` for tensors shaped ``(..., n, n)``."""
    return (x + x.T) / 2.0


# @torch.compile
def fast_symmetric_positive_definitive_matrix_inverse(x: Tensor) -> Tensor:
    """Invert an SPD matrix using a single Cholesky factorization.

    Args:
        x: Symmetric positive definite tensor with shape ``(..., n, n)``.

    Returns:
        Tensor: Inverse of ``x`` computed via
        :func:`torch.linalg.cholesky` followed by
        :func:`torch.cholesky_inverse`.
    """
    L = torch.linalg.cholesky(x)

    return cholesky_inverse(L)


def get_largest_singular_value_power_iteration(matrix: Tensor, num_iterations: int = 25, v: Optional[Tensor] = None) -> \
        Tuple[Tensor, Tensor]:
    """Approximate the dominant singular value via power iteration.

    Args:
        matrix: Input matrix with shape ``(m, n)`` or batch ``(..., m, n)``.
        num_iterations: Number of power iterations to perform.
        v: Optional initial vector with shape ``(n, 1)``; sampled when omitted.

    Returns:
        Tuple[Tensor, Tensor]: Estimated singular value and the final
        (unnormalized) dominant vector.
    """
    if v is None:
        v = torch.randn(matrix.shape[1], 1, device=matrix.device)

    with torch.no_grad():
        ata = matrix.T @ matrix
        for _ in range(num_iterations):
            v_new = ata @ v
            v = v_new / torch.norm(v_new)

    largest_eigenvalue = (v.T @ ata @ v) / (v.T @ v)
    return torch.sqrt(largest_eigenvalue.maximum(torch.tensor(0.0, device=matrix.device))), v


class CholeskyRankKFunction(Function):
    """Differentiable rank-k update of a Cholesky factor.

    Given a pre-factorized SPD matrix ``A = L L^T`` and a low-rank update
    ``M M^T``, the forward path produces ``L'`` where ``A + M M^T = L' L'^T``.
    The backward pass implements the analytical gradients derived for blocked
    Cholesky updates and is safe to use inside ``torch.autograd``.
    """

    @staticmethod
    def forward(L: Tensor, M: Tensor) -> Tensor:
        """Compute ``chol(L L^T + M M^T)`` while preserving gradients."""
        if L.ndim != 2 or L.shape[0] != L.shape[1]:
            raise ValueError("L must be square matrix (n,n)")
        if M.ndim != 2 or M.shape[0] != L.shape[0]:
            raise ValueError("M must have shape (n,k)")
        k = M.shape[1]
        with torch.no_grad():
            Lp = L.clone().contiguous()
            M_c = M.contiguous()
            for i in range(k):
                col = M_c[:, i]
                Lp = CholeskyRankKFunction._rank_1_update(Lp, col)
            Lp = torch.tril(Lp)  # enforce lower triangular
        return Lp

    @staticmethod
    def setup_context(ctx: Any, inputs: tuple[Any, ...], output: Tensor) -> None:
        L, M = inputs
        Lp = output

        ctx.L = L
        ctx.M = M
        ctx.Lp = Lp

    @staticmethod
    def backward(ctx: Any, grad_output: Optional[Tensor]) -> tuple[Optional[Tensor], Optional[Tensor]]:
        """Differentiate the rank-k update via the closed-form Jacobian."""
        if grad_output is None:
            return None, None

        L = ctx.L
        M = ctx.M
        Lp = ctx.Lp

        needs_L, needs_M = ctx.needs_input_grad
        if not (needs_L or needs_M):
            return None, None

        G = grad_output
        P = Lp.transpose(-1, -2).matmul(G)
        diagP = torch.diagonal(P, dim1=-2, dim2=-1)
        S = torch.tril(P) - 0.5 * torch.diag_embed(diagP)
        n = Lp.shape[-1]
        I = torch.eye(n, device=Lp.device, dtype=Lp.dtype)
        Lp_inv = torch.linalg.solve_triangular(Lp, I, left=True, upper=False)
        tmp = S.matmul(Lp_inv)
        W = torch.linalg.solve_triangular(Lp.transpose(-1, -2), tmp, left=True, upper=True)
        W = 0.5 * (W + W.transpose(-1, -2))
        grad_L = None
        grad_M = None
        if needs_L:
            grad_L = torch.tril(2.0 * W.matmul(L))
        if needs_M:
            grad_M = 2.0 * W.matmul(M)
        return grad_L, grad_M

    @staticmethod
    def _rank_1_update(L: Tensor, x: Tensor) -> Tensor:
        """Classical rank-1 update used internally by the forward pass."""
        n = x.shape[0]

        L = L.contiguous()
        x = x.contiguous()

        for k in range(n):
            r = torch.hypot(L[k, k], x[k])

            c = r / L[k, k]
            s = x[k] / L[k, k]
            L[k, k] = r

            if k + 1 < n:
                lk = L[k + 1:n, k]
                xk = x[k + 1:n]

                L[k + 1:n, k] = (lk + s * xk) / c
                x[k + 1:n] = c * xk - s * L[k + 1:n, k]

        return L


def cholesky_rank_k(L: Tensor, M: Tensor) -> Tensor:
    """User-facing wrapper for :class:`CholeskyRankKFunction`.

    Args:
        L: Lower-triangular factor with shape ``(n, n)``.
        M: Rank-k update matrix with shape ``(n, k)``.

    Returns:
        Tensor: Updated Cholesky factor of ``A + M M^T``.
    """
    return CholeskyRankKFunction.apply(L, M)


class PositiveExp(torch.nn.Module):
    """Exponentiate unconstrained parameters to enforce positivity."""

    def __init__(self, eps: float = 1e-4, device=None,
                 dtype=None) -> None:
        super().__init__()
        self.register_buffer("eps", torch.tensor(eps, dtype=dtype, device=device))

    def forward(self, X: Tensor) -> Tensor:
        """Map unconstrained tensors ``X`` into ``exp(X) + eps``."""
        return X.exp() + self.eps

    def right_inverse(self, A: Tensor) -> Tensor:
        """Invert :meth:`forward` via ``log(A - eps)`` while matching shapes."""
        return (A - self.eps).log()


class PositiveSq(torch.nn.Module):
    """Square unconstrained parameters and shift them above zero."""

    def __init__(self, eps: float = 1e-4, device=None,
                 dtype=None) -> None:
        super().__init__()
        self.register_buffer("eps", torch.tensor(eps, dtype=dtype, device=device))

    def forward(self, X: Tensor) -> Tensor:
        """Return ``eps + X.square()`` preserving the input shape."""
        return self.eps + X.square()

    def right_inverse(self, A: Tensor) -> Tensor:
        """Undo :meth:`forward` through ``sqrt(A - eps)``."""
        return (A - self.eps).sqrt()


@torch.no_grad()
def weight_grad_norm_normalization_(
        parameters: Union[Iterable[Tensor], Tensor],
        norm_type: float = 2.0,
        error_if_nonfinite: bool = False,
        foreach: Optional[bool] = None,
) -> list[float]:
    """Normalize gradients so each parameter block has unit norm.

    Args:
        parameters: Iterable of tensors (or a single tensor) whose gradients
            will be normalized in-place. Each tensor can have arbitrary shape.
        norm_type: P-norm applied to each gradient tensor.
        error_if_nonfinite: Whether to raise if any gradient norm is NaN/Inf.
        foreach: Whether to use the foreach implementations when available.

    Returns:
        list[float]: Collected norms for every gradient that was normalized.
    """
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    else:
        is_generator = isinstance(parameters, types.GeneratorType)
        # prevent generators from being exhausted
        parameters = list(parameters)
        if is_generator and len(parameters) == 0:
            warnings.warn(
                "`parameters` is an empty generator, no gradient clipping will occur.",
                stacklevel=3,
            )

    output_norms = []

    for p in parameters:
        if p.numel() == 1:
            continue

        if p.grad is not None:
            total_norm = torch.nn.utils.get_total_norm(p.grad, norm_type, error_if_nonfinite, foreach)

            p.grad.div_(total_norm)

            output_norms.append(total_norm.item())

    return output_norms
