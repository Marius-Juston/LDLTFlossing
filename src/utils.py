import logging
import traceback
import types
import warnings
from typing import Tuple, Optional, Any

import torch
from torch import Tensor, cholesky_inverse
from torch.autograd import Function
from torch.linalg import eigh


@torch.autocast(device_type='cuda', enabled=False)
def safe_cholesky(A: torch.Tensor, upper=False, eps=1e-3):
    A32 = A.to(torch.float32)

    res, info = torch.linalg.cholesky_ex(A32, upper=upper)

    if info.any():
        logging.warning("Cholesky failed:\n" + traceback.format_exc())

        with torch.no_grad():
            A_sym = (A32 + A32.mT) / 2
            w, V = torch.linalg.eigh(A_sym)

            w_clamped = torch.clamp(w, min=eps)

            D = torch.diag_embed(w_clamped)

            B_pos = V.matmul(D).matmul(V.mT)

            A.data = B_pos.data.to(A.dtype)

            res, info = torch.linalg.cholesky_ex(B_pos, upper=upper)

        if info.any():
            logging.error("eigh failed:\n" + traceback.format_exc())

            raise ValueError("Unable to perform Cholesky.")

    return res.to(A.dtype) if A.dtype != torch.float32 else res


@torch.autocast(device_type='cuda', enabled=False)
def amp_solve_triangular(A, B, **kwargs):
    A32 = A.to(torch.float32)
    B32 = B.to(torch.float32)

    C = torch.linalg.solve_triangular(A32, B32, **kwargs)

    return C.to(A.dtype)


# @torch.compile
def matrix_inv_square_root(x: Tensor) -> Tensor:
    """
    Returns the Matrix's square root inverse x**(-1/2), detaches matrix so that gradients do not cause problems close to 0
    :param x: Matrix to square root inverse
    :return: The inverse square root of x
    """
    eigvals, eigvecs = eigh(x.detach())
    return (eigvecs * eigvals.rsqrt().unsqueeze(0)) @ eigvecs.T


def matrix_inv_square_root_triangle(x: Tensor, B: Tensor = None, upper: bool = False, left: bool = True) -> Tensor:
    """
    Returns the Matrix's square root inverse x**(-1/2), detaches matrix so that gradients do not cause problems close to 0.
    THIS IS NOT THE "UNIQUE" SYMMETRIC MATRIX INV SQUARE ROOT!!!
    :param B:
    :param upper:
    :param left:
    :param x: Matrix to square root inverse
    :return: The inverse square root of x
    """
    L = torch.linalg.cholesky_ex(x, upper=upper).L

    if B is None:
        B = torch.eye(L.shape[0], dtype=x.dtype, device=x.device)
    L_inv = torch.linalg.solve_triangular(L, B, upper=upper, left=left)
    return L_inv


def symmetrize_(x: Tensor) -> Tensor:
    return (x + x.T) / 2.0


# @torch.compile
def fast_symmetric_positive_definitive_matrix_inverse(x: Tensor) -> Tensor:
    """
    With the assumption that x is a symmetric positive definite (PSD) matrix, take the inverse of the matrix
    :param x: The matrix to invert
    :return: The inverse of the matrix
    """
    L = torch.linalg.cholesky(x)

    return cholesky_inverse(L)


def get_largest_singular_value_power_iteration(matrix: Tensor, num_iterations: int = 25, v: Optional[Tensor] = None) -> \
        Tuple[Tensor, Tensor]:
    """
    Computes the largest singular value of a matrix using the Power Iteration method.
    This is a classic and reliable iterative method.

    Args:
        matrix (torch.Tensor): The input matrix.
        num_iterations (int): The number of iterations to run.

    Returns:
        float: The largest singular value.
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
    """
    Inputs:
      L : (n,n) lower-triangular Cholesky factor (A = LL^T)
      M : (n,k) update matrix; A' = A + MM^T
    Output:
      Lp : (n,n) lower-triangular Cholesky factor of A'
    """

    @staticmethod
    def forward(L, M):
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
    def setup_context(ctx: Any, inputs: tuple[Any, ...], output: Any) -> Any:
        L, M = inputs
        Lp = output

        ctx.L = L
        ctx.M = M
        ctx.Lp = Lp

    @staticmethod
    def backward(ctx, grad_output):
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
    def _rank_1_update(L: torch.Tensor, x: torch.Tensor):
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


def cholesky_rank_k(L: torch.Tensor, M: torch.Tensor):
    """
    Wrapper to apply the custom Function; this is the user-facing call.
    """
    return CholeskyRankKFunction.apply(L, M)


class PositiveExp(torch.nn.Module):
    def __init__(self, eps=1e-4, device=None,
                 dtype=None) -> None:
        super().__init__()
        self.register_buffer("eps", torch.tensor(eps, dtype=dtype, device=device))

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        # return X ** 2
        return X.exp() + self.eps

    def right_inverse(self, A: torch.Tensor) -> torch.Tensor:
        return (A - self.eps).log()


class PositiveSq(torch.nn.Module):

    def __init__(self, eps=1e-4, device=None,
                 dtype=None) -> None:
        super().__init__()
        self.register_buffer("eps", torch.tensor(eps, dtype=dtype, device=device))

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        # return X ** 2
        return self.eps + X.square()

    def right_inverse(self, A: torch.Tensor) -> torch.Tensor:
        return (A - self.eps).sqrt()


def safe_cholesky(A: torch.Tensor, upper=False, eps=1e-4, max_tries=5):
    res, info = torch.linalg.cholesky_ex(A, upper=upper)

    if info.any():
        A64 = A.to(torch.float64)

        A_sym = (A64 + A64.mT) / 2

        L, Q = torch.linalg.eigh(A_sym)

        A_sym = torch.dist(Q @ torch.diag_embed(L).clamp(min=1e-4, max=1e3) @ Q.mT, A)

        for i in range(0, max_tries):
            try:
                jitter = eps * (10 ** i)

                return torch.linalg.cholesky(A_sym + jitter * torch.eye(A.size(-1), device=A.device, dtype=A.dtype),
                                             upper=upper).to(A.dtype)
            except RuntimeError:
                logging.error(f"Chokesy failed (trial {i}: {eps}): {traceback.format_exc()}")

        raise ValueError(f"Unable to perform Cholesky.\n{'\n'.join(traceback.format_stack())}")

    return res


@torch.no_grad()
def weight_grad_norm_normalization_(
        parameters,
        norm_type: float = 2.0,
        error_if_nonfinite: bool = False,
        foreach: Optional[bool] = None,
):
    r"""Clip the gradient norm of an iterable of parameters.

    The norm is computed over the norms of the individual gradients of all parameters,
    as if the norms of the individual gradients were concatenated into a single vector.
    Gradients are modified in-place.

    This function is equivalent to :func:`torch.nn.utils.get_total_norm` followed by
    :func:`torch.nn.utils.clip_grads_with_norm_` with the ``total_norm`` returned by ``get_total_norm``.

    Args:
        parameters (Iterable[Tensor] or Tensor): an iterable of Tensors or a
            single Tensor that will have gradients normalized
        max_norm (float): max norm of the gradients
        norm_type (float, optional): type of the used p-norm. Can be ``'inf'`` for
            infinity norm. Default: 2.0
        error_if_nonfinite (bool, optional): if True, an error is thrown if the total
            norm of the gradients from :attr:`parameters` is ``nan``,
            ``inf``, or ``-inf``. Default: False
        foreach (bool, optional): use the faster foreach-based implementation.
            If ``None``, use the foreach implementation for CUDA and CPU native tensors and silently
            fall back to the slow implementation for other device types.
            Default: ``None``

    Returns:
        Total norm of the parameter gradients (viewed as a single vector).
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
