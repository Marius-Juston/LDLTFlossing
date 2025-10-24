import os.path
from typing import Optional, Sequence, List

import numpy as np
import torch
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.image import AxesImage
from matplotlib.patches import Rectangle, ConnectionPatch
from matplotlib.ticker import MaxNLocator
from mpl_toolkits.axes_grid1 import make_axes_locatable
from torch import nn

import utils
from models.linear_layers import LinearLipschitz
from models.linear_model import DeepLipschitzLinearResNet, DeepLipschitzSequential

DEFAULT_DPI = 600

OUTPUT_FORMAT = 'png'


def block(tensors: Sequence[Sequence[torch.Tensor]]) -> torch.Tensor:
    return torch.cat([torch.cat(b, dim=-1) for b in tensors], dim=-2)


def E_real(dimensions: Sequence[int], i: Optional[int], device=None, dtype=None):
    Dn = sum(dimensions)

    factory_kwargs = dict(device=device, dtype=dtype)

    if i is None:
        d1, d2 = dimensions[0], dimensions[-1]
    else:
        d1, d2 = dimensions[i], dimensions[i + 1]

    E = torch.zeros((Dn, d1 + d2), **factory_kwargs)
    id_1 = torch.eye(d1, **factory_kwargs)
    id_2 = torch.eye(d2, **factory_kwargs)

    if i is None:
        E[:d1, :d1] = id_1
        E[-d2:, -d2:] = id_2
    else:
        result = torch.cat(
            (torch.tensor([0], device=device), torch.cumsum(torch.tensor(dimensions, device=device), dim=0)))

        E[result[i]: result[i + 1], :d1] = id_1
        E[result[i + 1]: result[i + 2], -d2:] = id_2

    return E.T


def main_real(A: torch.Tensor, B: torch.Tensor, Cs: List[torch.Tensor], Ls: List[torch.Tensor], Lms, dimensions,
              device=None):
    L = len(Ls)

    factory_kwargs = dict(device=device, dtype=A.dtype)

    E0 = E_real(dimensions, None, **factory_kwargs)

    I = torch.eye(dimensions[0], **factory_kwargs)

    M0 = block([[A.T @ A - I, A.T @ B], [B.T @ A, B.T @ B]])

    M0 = E0.T @ M0 @ E0

    Ms = [M0]

    for i in range(L):
        Ei = E_real(dimensions, i, **factory_kwargs)

        lambda_i = Ls[i]
        Ci = Cs[i]

        CC = torch.zeros((Ci.shape[0] * 2, Ci.shape[1] + Ci.shape[0]), **factory_kwargs)
        CC[:Ci.shape[0], :Ci.shape[1]] = Ci
        CC[-Ci.shape[0]:, -Ci.shape[0]:] = torch.eye(Ci.shape[0], **factory_kwargs)

        L_, m = Lms[i]

        M_sub = block(
            [
                [-2 * L_ * m * lambda_i, (m + L_) * lambda_i],
                [(m + L_) * lambda_i, -2 * lambda_i],
            ]
        )

        Mi = CC.T @ M_sub @ CC
        expanded_Mi = Ei.T @ Mi @ Ei

        Ms.append(expanded_Mi)

    output_M = Ms[0]

    for i in range(1, len(Ms)):
        output_M += Ms[i]

    # eigs = np.linalg.eigh(output_M)

    # print(eigs)

    M = output_M

    return M


def extract_lmi_constants_deep_resnet_new(model: DeepLipschitzLinearResNet):
    x = torch.zeros((1, model.in_features), **model.factory_kwargs)

    first, a_weight, d_inv, prev_alpha, prev_omega_r, prev_constant = model.A(x)

    gamma = model.A.identity
    sigma_lower = None

    current_input = x

    Cs = []
    Ls = []
    Lms = []
    dimensions = [
        a_weight.shape[1],
    ]

    for layer in model.layers:
        T = gamma @ d_inv

        current_input, c_weight, d_inv, prev_alpha, prev_omega_r, prev_constant = layer(current_input, prev_alpha,
                                                                                        prev_omega_r, prev_constant)

        if sigma_lower is None:
            sigma_lower = T
        else:
            sigma_lower = utils.safe_cholesky(sigma_lower @ sigma_lower.T + T @ T.T, upper=False)

        gamma = T @ c_weight.T

        current_input = model.activation(current_input)

        Cs.append(c_weight)
        Ls.append(layer.identity_out)
        Lms.append([1, 0])
        dimensions.append(c_weight.shape[0])

    second, b_weight = model.compute_b(current_input, sigma_lower, a_weight, prev_alpha, prev_omega_r)

    return a_weight, b_weight, Cs, Ls, Lms, dimensions


def extract_lmi_constants_deep_linear(model: DeepLipschitzSequential):
    x = torch.zeros((1, model.in_features), **model.factory_kwargs)

    current_input = x
    b_weight = torch.eye(model.out_features, **model.factory_kwargs)
    a_weight = torch.zeros((model.out_features, model.in_features), **model.factory_kwargs)

    d_sqrt = model.lipschitz_constant * model.layers[0].identity
    prev_alpha = torch.tensor(1.0, **model.factory_kwargs)
    prev_const = torch.tensor(1.0, **model.factory_kwargs)

    Cs = []
    Ls = []
    Lms = []
    dimensions = [
        model.out_features,
    ]

    for layer in model.layers:
        if isinstance(layer, nn.Dropout):
            current_input = layer(current_input)
        else:
            layer: LinearLipschitz

            current_input, d_sqrt, c_weight = layer(current_input, d_sqrt, prev_alpha, prev_const)

            prev_const = layer.constant
            prev_alpha = layer.alpha

            Cs.append(c_weight)
            Ls.append(layer.identity_out)
            Lms.append([1, 0])
            dimensions.append(c_weight.shape[0])

    return a_weight, b_weight, Cs, Ls, Lms, dimensions


def extract_lmi_constants(model):
    if isinstance(model, DeepLipschitzLinearResNet):
        return extract_lmi_constants_deep_resnet_new(model)
    else:
        return extract_lmi_constants_deep_linear(model)


def plot_matrix(M: torch.Tensor, fig_name: str = None, dpi: int = DEFAULT_DPI, mask: bool = True,
                normalize: bool = True, ax: Optional[Axes] = None):
    if mask:
        masked_zerp = torch.abs(M) <= torch.finfo(M.dtype).eps
        no_mask = torch.logical_not(masked_zerp)

        min, max = M[no_mask].min(), M[no_mask].max()
    else:
        min, max = M.min(), M.max()

    vmin, vmax = min, max

    if normalize:
        if vmin == vmin:
            normalized = M / vmax  * 255
        else:
            normalized = (M - min) / (max - min) * 255

        normalized = normalized.round()
        vmin = 0
        vmax = 255
    else:
        normalized = M

    if mask:
        normalized[masked_zerp] = torch.nan

    normalized = normalized.cpu().numpy()

    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 8))
        im = ax.matshow(normalized, cmap='viridis', vmin=vmin, vmax=vmax)
        fig.tight_layout()
        fig.savefig(f"{fig_name}.{OUTPUT_FORMAT}", dpi=dpi)
        plt.close()
    else:
        im = ax.matshow(normalized, cmap='viridis', vmin=vmin, vmax=vmax)

    return im


def freedman_diaconis_bins(x: torch.Tensor):
    x = x.flatten()
    n = x.numel()
    q25, q75 = torch.quantile(x, 0.25), torch.quantile(x, 0.75)
    iqr = q75 - q25
    h = 2 * iqr / (n ** (1 / 3))
    h = h.item()
    if h == 0:
        return int(np.sqrt(n))  # fallback: sqrt rule
    bins = int(torch.ceil((x.max() - x.min()) / h).item())
    return max(1, bins)


def plot_histogram(x: torch.Tensor, fig_name: str, dpi=DEFAULT_DPI):
    fig, ax = plt.subplots()
    counts, edges = torch.histogram(x.cpu(), bins=50, density=True)

    ax.bar(edges[:-1].numpy(), counts.numpy(),
           width=(edges[1] - edges[0]).item())
    ax.set_xlabel("Eigenvalues")
    ax.set_ylabel("Density")
    ax.set_title("Histogram of eigenvalues")
    fig.tight_layout()
    fig.savefig(f"{fig_name}.{OUTPUT_FORMAT}", dpi=dpi)
    plt.close()


def singular_extremes(M: torch.Tensor):
    """Return (sigma_min, sigma_max) for matrix M."""
    s = torch.linalg.svdvals(M)
    return s.min().item(), s.max().item()


def plot_singular_extremes(a_weight, b_weight, Cs, fig_name: str, dpi=DEFAULT_DPI):
    matrices = [a_weight] + Cs + [b_weight]
    labels = ["A"] + [f"C{i + 1}" for i in range(len(Cs))] + ["B"]

    sigmas_min, sigmas_max = [], []

    for M in matrices:
        smin, smax = singular_extremes(M)
        sigmas_min.append(smin)
        sigmas_max.append(smax)
    sigma_diff = [a - b for a, b in zip(sigmas_max, sigmas_min)]

    # Plot
    x = range(len(matrices))

    fig, ax = plt.subplots()
    ax.plot(x, sigmas_min, "o-", label="σ_min", markersize=8)
    ax.plot(x, sigmas_max, "s-", label="σ_max", markersize=8)
    ax.plot(x, sigma_diff, "^-", label="σ_diff", markersize=8)
    ax.set_xticks(x, labels)
    ax.set_xlabel("Matrix")
    ax.set_ylabel("Maximum singular value")
    ax.set_title("Extremal Singular Values of Matrices")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(f"{fig_name}.{OUTPUT_FORMAT}", dpi=dpi)
    plt.close()


def test_sqrt(device, tol=5e-3):
    n = 1000

    A = torch.randn(n, n, device=device)

    W = torch.eye(n, device=device) + A @ A.T

    w_gt = torch.linalg.inv(W)

    w_inv_actual = utils.fast_symmetric_positive_definitive_matrix_inverse(W)
    assert torch.dist(w_gt, w_inv_actual) < tol

    # Symmetric version
    L = utils.matrix_inv_square_root(W)
    w_inv = L @ L.T
    assert torch.dist(w_gt, w_inv) < tol
    w_inv = L.T @ L
    assert torch.dist(w_gt, w_inv) < tol

    # Triangular non-symmetric version
    L = utils.matrix_inv_square_root_triangle(W)
    w_inv_triangle = L.T @ L
    assert torch.dist(w_gt, w_inv_triangle) < tol


def disable_ticks(ax: Axes):
    ax.set_xticklabels([])
    ax.set_xticks([])
    ax.set_yticklabels([])
    ax.set_yticks([])


def add_colorbar(fig: Figure, ax: Axes, im: AxesImage, n_bins=10):
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    cb = fig.colorbar(im, cax=cax)
    cb.locator = MaxNLocator(nbins=n_bins)


def plot_zoomed_in_system(A, B, fig_name: str, dpi=DEFAULT_DPI):
    fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(16, 8))

    ax1.set_title("Full LMI")
    ax2.set_title(r"$B^\top B$")

    im1 = plot_matrix(A, ax=ax1, normalize=False)
    im2 = plot_matrix(B, ax=ax2, normalize=False)

    a_size = np.array(A.shape)
    b_size = np.array(B.shape)

    size = b_size
    pos = a_size - b_size

    rect = Rectangle(pos, *size,
                     linewidth=2, edgecolor="black", facecolor="none")
    ax1.add_patch(rect)

    con1 = ConnectionPatch(xyA=pos, coordsA=ax1.transData,
                           xyB=(0, 0), coordsB=ax2.transData,
                           color="black")

    con2 = ConnectionPatch(xyA=pos + size, coordsA=ax1.transData,
                           xyB=b_size[::-1], coordsB=ax2.transData,
                           color="black")

    fig.add_artist(con1)
    fig.add_artist(con2)

    print(A.min(), A.max())
    print(B.min(), B.max())

    add_colorbar(fig, ax1, im1)
    add_colorbar(fig, ax2, im2)

    disable_ticks(ax1)
    disable_ticks(ax2)

    fig.tight_layout()
    fig.savefig(f"{fig_name}.{OUTPUT_FORMAT}", dpi=dpi)
    plt.close()


def model_artifacting(model_fn, weights, vals=None, fig_name: str = None, dpi=DEFAULT_DPI, factory_kwargs=None):
    if factory_kwargs is None:
        factory_kwargs = {}

    if vals is None:
        vals = [2, 4, 8, 16]

    fig, axs = plt.subplots(ncols=2, nrows=2, figsize=(8, 8))

    for ax, im in zip(axs.flatten(), vals):
        model = model_fn(im, im, weights, **factory_kwargs)

        with torch.no_grad():
            a_weight, b_weight, Cs, Ls, Lms, dimensions = extract_lmi_constants(model)

        b_b = b_weight.T @ b_weight

        plot_matrix(b_b, ax=ax, normalize=False)
        ax.set_title(f"n = {im}", pad=0)
        ax.axis("off")

    fig.tight_layout()
    fig.savefig(f"{fig_name}.{OUTPUT_FORMAT}", dpi=dpi)
    plt.close()


def ldlt_decomposition(M: torch.Tensor, matrix_block_sizes: List[int]):
    # Partition indices
    dimensions = [0, *np.cumsum(matrix_block_sizes)]
    k = len(matrix_block_sizes)

    n = M.shape[0]
    L = torch.eye(n, dtype=M.dtype, device=M.device)
    Ds = [None] * k

    # Work with a mutable copy of M for Schur complement updates
    S = M.clone()

    for r in range(k):
        i0, i1 = dimensions[r], dimensions[r + 1]
        # Extract diagonal block
        D_block = S[i0:i1, i0:i1]
        Ds[r] = D_block.clone()

        # If not the last block, compute L blocks and update Schur complement
        if r < k - 1:
            # Lower blocks wrt D_block
            for s in range(r + 1, k):
                j0, j1 = dimensions[s], dimensions[s + 1]
                # L_{sj} = S_{sj} * D_block^{-1}
                L[j0:j1, i0:i1] = torch.linalg.solve(D_block.T, S[j0:j1, i0:i1].T).T

            # Update Schur complement
            for s in range(r + 1, k):
                for t in range(r + 1, k):
                    j0, j1 = dimensions[s], dimensions[s + 1]
                    k0, k1 = dimensions[t], dimensions[t + 1]

                    S[j0:j1, k0:k1] -= L[j0:j1, i0:i1] @ D_block @ L[k0:k1, i0:i1].T

    return L, Ds


def plot_ldlt_decomposition(M, dimensions, fig_name: str = None, dpi=DEFAULT_DPI):
    L, D = ldlt_decomposition(M, dimensions)

    # Triangular matrix
    fig, ax = plt.subplots(figsize=(8, 8))

    im = plot_matrix(L, ax=ax, normalize=False)

    add_colorbar(fig, ax, im)

    disable_ticks(ax)
    ax.set_title("Unit lower triangular L Matrix")

    fig.tight_layout()
    fig.savefig(f"{fig_name}_l.{OUTPUT_FORMAT}", dpi=dpi)

    # Diagonal matrix D
    fig, ax = plt.subplots(figsize=(8, 8))

    diagonal_block = torch.block_diag(*D)
    im = plot_matrix(diagonal_block, ax=ax, normalize=False)

    add_colorbar(fig, ax, im)

    disable_ticks(ax)
    ax.set_title(r"Block diagonal matrix D")

    fig.tight_layout()
    fig.savefig(f"{fig_name}_d.{OUTPUT_FORMAT}", dpi=dpi)

    plt.close()


def get_maximum_model(model_func, *args, device=None):
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = model_func(*args, device)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)

    for i in range(300):
        optimizer.zero_grad(set_to_none=True)

        a_weight, b_weight, Cs, Ls, Lms, dimensions = extract_lmi_constants(model)

        M = main_real(a_weight, b_weight, Cs, Ls, Lms, dimensions, device=device)

        eigenvalues = torch.linalg.eigvalsh(M)

        if i == 0:
            print(eigenvalues)

        loss = -eigenvalues.sum()

        loss.backward()
        optimizer.step()

        print(loss.item())

    print(eigenvalues)

    return model


def plot_information(M, a_weight, Cs, b_weight, dimensions, figs_folder, suffix=''):
    plot_ldlt_decomposition(M, dimensions, os.path.join(figs_folder, f"ldlt_decomposition{suffix}"))

    eigenvalues, eigenvectors = torch.linalg.eigh(M)

    print(M)

    print(eigenvalues)

    plot_matrix(M, os.path.join(figs_folder, f"full_lmi{suffix}"))

    removed_diag = (M - torch.diag(torch.diag(M)))

    plot_matrix(removed_diag, os.path.join(figs_folder, f"lmi_without_diag{suffix}"))

    binary = (torch.abs(M) > 1e-5).to(torch.uint8)

    plot_matrix(binary, os.path.join(figs_folder, f"binary_lmi{suffix}"), mask=False)

    plot_matrix(eigenvectors, os.path.join(figs_folder, f"eigenvectors_lmi{suffix}"))

    B_zoom = b_weight.T @ b_weight
    plot_matrix(B_zoom, os.path.join(figs_folder, f"zoomed_in_b{suffix}"))

    plot_zoomed_in_system(removed_diag, B_zoom, os.path.join(figs_folder, f"inplace_zoomed_in_b{suffix}"))

    plot_histogram(eigenvalues, os.path.join(figs_folder, f"eigenvalues_lmi{suffix}"))

    plot_singular_extremes(a_weight, b_weight, Cs, os.path.join(figs_folder, f"singular_weights{suffix}"))

    return eigenvalues


def main():
    torch.manual_seed(0)

    model_fnc = DeepLipschitzLinearResNet

    figs_folder = f'figs/linear/{'resnet' if model_fnc == DeepLipschitzLinearResNet else 'fnn'}'
    os.makedirs(figs_folder, exist_ok=True)

    input_features = 1
    output_features = input_features
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    factory_kwargs = dict(device=device, dtype=torch.float32)

    weights = (64, 64, 64, 64, 64)

    if model_fnc == DeepLipschitzLinearResNet:
        model_artifacting(model_fnc, weights, fig_name=os.path.join(figs_folder, "b_artifacting_grid"),
                          factory_kwargs=factory_kwargs)


    torch.no_grad()

    inputs = (input_features, output_features, weights)

    # model = get_maximum_model(model_fnc, *inputs, device=device)

    model = model_fnc(*inputs, **factory_kwargs)

    # save_folder = '/home/mjuston2/Documents/DeepLipschitzResNet/runs/sine_trainer_20250923_003644'
    save_folder = None

    if save_folder is not None:
        data = torch.load(f"{save_folder}/best.pt", map_location=device)
        new_state_dict = {key.replace('_orig_mod.', ''): val for key, val in data.items()}

        model.load_state_dict(new_state_dict)

    with torch.no_grad():
        a_weight, b_weight, Cs, Ls, Lms, dimensions = extract_lmi_constants(model)

    M = main_real(a_weight, b_weight, Cs, Ls, Lms, dimensions, device=device)

    eigenvalues = plot_information(M, a_weight, Cs, b_weight, dimensions, figs_folder)

    assert torch.all(eigenvalues < 1e-3), "Not all the eigenvalues of the LMI are negative"


if __name__ == '__main__':
    main()
