import os
from typing import Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable


def _finite_minmax(arr: np.ndarray) -> Tuple[float, float]:
    arr = np.asarray(arr)
    finite = np.isfinite(arr)
    if not np.any(finite):
        return 0.0, 1.0
    vmin = float(np.min(arr[finite]))
    vmax = float(np.max(arr[finite]))
    if vmin == vmax:
        # avoid singular colorbars
        eps = 1e-12 if vmin == 0.0 else abs(vmin) * 1e-12
        return vmin - eps, vmax + eps
    return vmin, vmax


def save_solution_comparison_heatmaps(
    *,
    x: np.ndarray,
    y: np.ndarray,
    u_pred: np.ndarray,
    u_true: np.ndarray,
    save_path: str,
    x_label: str = "x",
    y_label: str = "y",
    title_prefix: str = "",
    solution_cmap: str = "jet",
    error_cmap: str = "magma",
    figsize: Tuple[float, float] = (15.5, 4.8),
    dpi: int = 300,
    solution_vmin_vmax: Optional[Tuple[float, float]] = None,
    error_vmin_vmax: Optional[Tuple[float, float]] = None,
    error_data: Optional[np.ndarray] = None,
    error_title: str = "Abs Error",
) -> None:
    """
    Save a 1x3 figure: prediction / exact / absolute error as heatmaps.

    Notes on array shapes:
      - x: shape (nx,), plotted on horizontal axis
      - y: shape (ny,), plotted on vertical axis
      - u_pred/u_true: shape (ny, nx) (rows correspond to y, columns to x)
    """
    x = np.asarray(x).reshape(-1)
    y = np.asarray(y).reshape(-1)
    u_pred = np.asarray(u_pred)
    u_true = np.asarray(u_true)

    if u_pred.shape != u_true.shape:
        raise ValueError(f"Shape mismatch: u_pred.shape={u_pred.shape}, u_true.shape={u_true.shape}")
    if u_pred.shape != (y.size, x.size):
        raise ValueError(
            f"Expected u shape (ny,nx)=({y.size},{x.size}), got {u_pred.shape}. "
            "Make sure you pass arrays in (y,x) layout."
        )

    plot_error = np.abs(u_pred - u_true) if error_data is None else np.asarray(error_data)
    if plot_error.shape != u_pred.shape:
        raise ValueError(f"Shape mismatch: error_data.shape={plot_error.shape}, expected {u_pred.shape}")

    if solution_vmin_vmax is None:
        sol_min_1, sol_max_1 = _finite_minmax(u_true)
        sol_min_2, sol_max_2 = _finite_minmax(u_pred)
        sol_vmin = min(sol_min_1, sol_min_2)
        sol_vmax = max(sol_max_1, sol_max_2)
    else:
        sol_vmin, sol_vmax = solution_vmin_vmax

    if error_vmin_vmax is None:
        _, err_max = _finite_minmax(plot_error)
        err_vmin, err_vmax = 0.0, err_max
    else:
        err_vmin, err_vmax = error_vmin_vmax

    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    fig, axs = plt.subplots(1, 3, figsize=figsize)

    def _draw(ax, data, vmin, vmax, cmap, title):
        im = ax.imshow(
            data,
            origin="lower",
            aspect="auto",
            extent=[float(x.min()), float(x.max()), float(y.min()), float(y.max())],
            vmin=vmin,
            vmax=vmax,
            cmap=cmap,
        )
        ax.set_title(title)
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="4.5%", pad=0.06)
        fig.colorbar(im, cax=cax, format="%.1e")

    prefix = (title_prefix + " ") if title_prefix else ""
    _draw(axs[0], u_pred, sol_vmin, sol_vmax, solution_cmap, f"{prefix}Prediction")
    _draw(axs[1], u_true, sol_vmin, sol_vmax, solution_cmap, f"{prefix}Exact")
    _draw(axs[2], plot_error, err_vmin, err_vmax, error_cmap, f"{prefix}{error_title}")

    plt.tight_layout()
    fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


