from __future__ import annotations

import argparse
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

_REPO_ROOT = Path(__file__).resolve().parent.parent
_SCRIPTS_DIR = _REPO_ROOT / "scripts"
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

from merge_10runs_training_curves import load_run_dir, read_run_training_curve


METHOD_ORDER = ["PINN", "PINN-RAR", "QPINN", "QPINN-RAR"]
# Display names used in titles/legends.
METHOD_DISPLAY_NAME = {
    "PINN": "PINN",
    "PINN-RAR": "PINN-RAR",
    "QPINN": "QPINN",
    "QPINN-RAR": "QPINN-RAR",
}
METHOD_COLOR = {
    "PINN": "#E69F00",
    "PINN-RAR": "#009E73",
    "QPINN": "#0072B2",
    "QPINN-RAR": "#CC79A7",
}
METHOD_ALIASES = {
    "DV": "QPINN-RAR",
    "pinn-rar-s": "PINN-RAR",
}


def _legend_right_column(
    fig: plt.Figure,
    handles: list[object],
    labels: list[str],
    *,
    title: Optional[str] = None,
    right: float = 0.82,
) -> None:
    """
    Put legend in a dedicated right-side column, so it never overlaps plots.

    right: the figure fraction reserved for subplots (0..1). Legend goes in (right..1).
    """
    if not handles:
        return

    # Reserve space for the legend column.
    fig.subplots_adjust(right=right)

    # Create a legend-only axes in the reserved right column.
    ax_leg = fig.add_axes([right + 0.01, 0.08, 1.0 - right - 0.02, 0.84])
    ax_leg.axis("off")
    ax_leg.legend(
        handles,
        labels,
        loc="upper left",
        framealpha=0.9,
        title=title,
    )


def _setup_matplotlib_cn_font() -> None:
    plt.rcParams["font.sans-serif"] = [
        "Microsoft YaHei",
        "SimHei",
        "Noto Sans CJK SC",
        "Arial Unicode MS",
        "DejaVu Sans",
    ]
    plt.rcParams["axes.unicode_minus"] = False


@dataclass(frozen=True)
class RunInfo:
    equation: str
    run_id: str
    seed: Optional[int]
    method: str
    csv_path: Path


def _read_text_head(p: Path, *, max_bytes: int = 20_000) -> str:
    with p.open("rb") as f:
        b = f.read(max_bytes)
    return b.decode("utf-8", errors="ignore")


def _parse_seed_from_log_head(text: str) -> Optional[int]:
    m = re.search(r"Random seed set to:\s*(\d+)", text)
    if m:
        return int(m.group(1))
    m = re.search(r"^\s*seed\s*:\s*(\d+)\s*$", text, flags=re.MULTILINE)
    if m:
        return int(m.group(1))
    return None


def _parse_method_from_log_head(text: str) -> Optional[str]:
    m_rar = re.search(r"^\s*use_rar\s*:\s*(True|False)\s*$", text, flags=re.MULTILINE)
    use_rar = m_rar is not None and m_rar.group(1) == "True"

    # QPINN family
    if re.search(r"\bUsing\s+DV\s+Solver\b", text) or re.search(r"^\s*solver\s*:\s*DV\s*$", text, re.MULTILINE):
        return "QPINN-RAR" if use_rar else "QPINN"

    # Classical2 (PINN family)
    if re.search(r"\bUsing\s+Classical\s+Solver2\b", text) or re.search(
        r"^\s*solver\s*:\s*Classical2\s*$", text, re.MULTILINE
    ):
        return "PINN-RAR" if use_rar else "PINN"

    return None


def _canonical_method(method: str) -> str:
    return METHOD_ALIASES.get(method, method)


def iter_run_infos(
    checkpoints_dir: Path,
    *,
    equation: Optional[str] = None,
    csv_name: str = "training_curve_iteration_normalized.csv",
) -> Iterable[RunInfo]:
    if equation:
        eq_dirs = [checkpoints_dir / equation]
    else:
        eq_dirs = [p for p in checkpoints_dir.iterdir() if p.is_dir()]

    for eq_dir in eq_dirs:
        if not eq_dir.exists() or not eq_dir.is_dir():
            continue
        for run_dir in sorted(eq_dir.iterdir()):
            if not run_dir.is_dir():
                continue
            csv_path = run_dir / csv_name
            log_path = run_dir / "output.log"
            if not log_path.exists():
                continue

            head = _read_text_head(log_path)
            method = _parse_method_from_log_head(head)
            if method is None:
                continue
            method = _canonical_method(method)
            seed = _parse_seed_from_log_head(head)
            yield RunInfo(
                equation=eq_dir.name,
                run_id=run_dir.name,
                seed=seed,
                method=method,
                csv_path=csv_path if csv_path.exists() else run_dir,
            )


def _read_curve(csv_path: Path) -> pd.DataFrame:
    if csv_path.is_dir():
        run = load_run_dir(csv_path, csv_path.parent.parent)
        metric_fields, by_it = read_run_training_curve(run)
        rows = []
        for it in sorted(by_it.keys()):
            row = {"iteration": it}
            row.update(by_it[it])
            rows.append(row)
        df = pd.DataFrame(rows, columns=["iteration"] + metric_fields)
    else:
        df = pd.read_csv(csv_path)

    # 把常用列（包括梯度范数 grad-norm）都转成数值型，方便后续插值/取均值
    for c in ["iteration_normalized", "iteration", "loss", "rel_l2_error", "grad-norm"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    if "iteration_normalized" not in df.columns and "iteration" in df.columns:
        it = pd.to_numeric(df["iteration"], errors="coerce")
        max_it = it.max(skipna=True)
        if pd.notna(max_it) and max_it > 0:
            df.insert(0, "iteration_normalized", it / max_it)
    return df


def _pick_10_runs(runs: list[RunInfo]) -> list[RunInfo]:
    # If there are more than 10, pick 10 deterministically (by seed then run_id).
    def key(r: RunInfo) -> tuple:
        seed = r.seed if r.seed is not None else 10**18
        return (seed, r.run_id)

    runs_sorted = sorted(runs, key=key)
    return runs_sorted[:10]


def plot_equation(
    equation_dir: Path,
    runs: list[RunInfo],
    *,
    out_dir: Path,
    metric: str,
    yscale: str,
) -> Path:
    _setup_matplotlib_cn_font()

    eq = equation_dir.name
    by_method: dict[str, list[RunInfo]] = {m: [] for m in METHOD_ORDER}
    for r in runs:
        if r.method in by_method:
            by_method[r.method].append(r)

    fig, axes = plt.subplots(1, 4, figsize=(19.5, 4.8), dpi=180, sharex=True, sharey=True)
    fig.suptitle(f"{eq} | {metric} | 10 runs per method (x = iteration_normalized)", y=1.02)

    for ax, method in zip(axes, METHOD_ORDER):
        mruns = by_method.get(method, [])
        mruns = _pick_10_runs(mruns)
        color = METHOD_COLOR.get(method, "#333333")
        label = METHOD_DISPLAY_NAME.get(method, method)

        # Plot 10 individual runs
        for ri, r in enumerate(mruns):
            df = _read_curve(r.csv_path)
            x = df.get("iteration_normalized")
            y = df.get(metric)
            if x is None or y is None:
                continue
            x_np = x.to_numpy(dtype=float)
            y_np = y.to_numpy(dtype=float)
            m = np.isfinite(x_np) & np.isfinite(y_np)
            if yscale == "log":
                m = m & (y_np > 0)
            if m.sum() < 2:
                continue
            ax.plot(
                x_np[m],
                y_np[m],
                color=color,
                alpha=0.22,
                linewidth=1.0,
                label="_nolegend_",
            )

        # Mean curve (align on a common x-grid)
        if mruns:
            x_grid = np.linspace(0.0, 1.0, 600, dtype=float)
            ys = []
            for r in mruns:
                df = _read_curve(r.csv_path)
                x = df.get("iteration_normalized")
                y = df.get(metric)
                if x is None or y is None:
                    continue
                x = x.to_numpy(dtype=float)
                y = y.to_numpy(dtype=float)
                m = np.isfinite(x) & np.isfinite(y)
                if yscale == "log":
                    m = m & (y > 0)
                if m.sum() < 2:
                    continue
                xs = x[m]
                ys0 = y[m]
                order = np.argsort(xs)
                xs = xs[order]
                ys0 = ys0[order]
                # de-dup x for np.interp
                xs_u, idx = np.unique(xs, return_index=True)
                ys_u = ys0[idx]
                if xs_u.size < 2:
                    continue
                if yscale == "log":
                    yi = np.power(10.0, np.interp(x_grid, xs_u, np.log10(ys_u), left=np.nan, right=np.nan))
                else:
                    yi = np.interp(x_grid, xs_u, ys_u, left=np.nan, right=np.nan)
                ys.append(yi)

            if ys:
                y_stack = np.vstack(ys)
                y_mean = np.nanmean(y_stack, axis=0)
                ax.plot(x_grid, y_mean, color=color, linewidth=2.2, alpha=0.95, label=label)

        ax.set_title(f"{label} (n={len(mruns)})")
        ax.set_xlabel("iteration_normalized")
        ax.set_yscale(yscale)
        ax.grid(False)
        ax.legend(
            loc="upper right",
            framealpha=0.9,
            ncol=1,
            borderaxespad=0.4,
            labelspacing=0.25,
            handlelength=2.0,
        )

    axes[0].set_ylabel(metric)

    fig.subplots_adjust(left=0.10, bottom=0.08, top=0.94, right=0.98, hspace=0.22, wspace=0.20)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{eq}_{metric}_10runs_by_method.png"
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_equation_loss_and_error_one_fig(
    equation_dir: Path,
    runs: list[RunInfo],
    *,
    out_dir: Path,
    yscale_loss: str = "log",
    yscale_error: str = "log",
) -> Path:
    _setup_matplotlib_cn_font()

    eq = equation_dir.name
    by_method: dict[str, list[RunInfo]] = {m: [] for m in METHOD_ORDER}
    for r in runs:
        if r.method in by_method:
            by_method[r.method].append(r)

    metrics = [("loss", yscale_loss), ("rel_l2_error", yscale_error)]

    fig, axes = plt.subplots(
        2,
        4,
        figsize=(19.5, 8.8),
        dpi=180,
        sharex=True,
        constrained_layout=False,
    )
    fig.suptitle(f"{eq} | 10 runs per method (x = iteration_normalized)", y=1.01)

    for row, (metric, yscale) in enumerate(metrics):
        for col, method in enumerate(METHOD_ORDER):
            ax = axes[row, col]
            mruns = by_method.get(method, [])
            mruns = _pick_10_runs(mruns)
            color = METHOD_COLOR.get(method, "#333333")
            label = METHOD_DISPLAY_NAME.get(method, method)

            # Plot 10 individual runs
            for ri, r in enumerate(mruns):
                df = _read_curve(r.csv_path)
                x = df.get("iteration_normalized")
                y = df.get(metric)
                if x is None or y is None:
                    continue
                x_np = x.to_numpy(dtype=float)
                y_np = y.to_numpy(dtype=float)
                m = np.isfinite(x_np) & np.isfinite(y_np)
                if yscale == "log":
                    m = m & (y_np > 0)
                if m.sum() < 2:
                    continue
                ax.plot(
                    x_np[m],
                    y_np[m],
                    color=color,
                    alpha=0.22,
                    linewidth=1.0,
                    label="_nolegend_",
                )

            # Mean curve (align on a common x-grid)
            if mruns:
                x_grid = np.linspace(0.0, 1.0, 600, dtype=float)
                ys = []
                for r in mruns:
                    df = _read_curve(r.csv_path)
                    x = df.get("iteration_normalized")
                    y = df.get(metric)
                    if x is None or y is None:
                        continue
                    x = x.to_numpy(dtype=float)
                    y = y.to_numpy(dtype=float)
                    m = np.isfinite(x) & np.isfinite(y)
                    if yscale == "log":
                        m = m & (y > 0)
                    if m.sum() < 2:
                        continue
                    xs = x[m]
                    ys0 = y[m]
                    order = np.argsort(xs)
                    xs = xs[order]
                    ys0 = ys0[order]
                    xs_u, idx = np.unique(xs, return_index=True)
                    ys_u = ys0[idx]
                    if xs_u.size < 2:
                        continue
                    if yscale == "log":
                        yi = np.power(
                            10.0,
                            np.interp(x_grid, xs_u, np.log10(ys_u), left=np.nan, right=np.nan),
                        )
                    else:
                        yi = np.interp(x_grid, xs_u, ys_u, left=np.nan, right=np.nan)
                    ys.append(yi)

                if ys:
                    y_stack = np.vstack(ys)
                    y_mean = np.nanmean(y_stack, axis=0)
                    ax.plot(x_grid, y_mean, color=color, linewidth=2.2, alpha=0.95, label=label)

            if row == 0:
                ax.set_title(f"{label} (n={len(mruns)})")
            ax.set_xlabel("iteration_normalized")
            ax.set_yscale(yscale)
            ax.grid(False)
            ax.legend(
                loc="upper right",
                framealpha=0.9,
                ncol=1,
                borderaxespad=0.4,
                labelspacing=0.25,
                handlelength=2.0,
            )

            if col == 0:
                ax.set_ylabel(metric)

    fig.subplots_adjust(left=0.07, bottom=0.08, top=0.93, right=0.98, hspace=0.22, wspace=0.20)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{eq}_loss_and_rel_l2_error_10runs_by_method.png"
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_equation_loss_and_error_overlay_one_fig(
    equation_dir: Path,
    runs: list[RunInfo],
    *,
    out_dir: Path,
    yscale_loss: str = "log",
    yscale_error: str = "log",
) -> Path:
    """Overlay PINN/PINN-RAR/QPINN/QPINN-RAR on the same axes for easier comparison."""
    _setup_matplotlib_cn_font()

    eq = equation_dir.name
    by_method: dict[str, list[RunInfo]] = {m: [] for m in METHOD_ORDER}
    for r in runs:
        if r.method in by_method:
            by_method[r.method].append(r)

    metrics = [("loss", yscale_loss), ("rel_l2_error", yscale_error)]

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(12.8, 4.6),
        dpi=180,
        sharex=True,
        constrained_layout=False,
    )
    fig.suptitle(f"{eq} | 10 runs per method (x = iteration_normalized)", y=1.02)

    for col, (metric, yscale) in enumerate(metrics):
        ax = axes[col]
        ax.set_yscale(yscale)
        ax.set_ylabel(metric)
        ax.grid(False)

        # Keep legend handles per-axis to avoid cross-metric handle reuse.
        mean_handles: dict[str, object] = {}

        for method in METHOD_ORDER:
            mruns = _pick_10_runs(by_method.get(method, []))
            color = METHOD_COLOR.get(method, "#333333")

            # Plot 10 individual runs (faint)
            for r in mruns:
                df = _read_curve(r.csv_path)
                x = df.get("iteration_normalized")
                y = df.get(metric)
                if x is None or y is None:
                    continue
                x_np = x.to_numpy(dtype=float)
                y_np = y.to_numpy(dtype=float)
                m = np.isfinite(x_np) & np.isfinite(y_np)
                if yscale == "log":
                    m = m & (y_np > 0)
                if m.sum() < 2:
                    continue
                ax.plot(
                    x_np[m],
                    y_np[m],
                    color=color,
                    alpha=0.16,
                    linewidth=0.9,
                )

            # Mean curve (thick)
            if mruns:
                x_grid = np.linspace(0.0, 1.0, 600, dtype=float)
                ys = []
                for r in mruns:
                    df = _read_curve(r.csv_path)
                    x = df.get("iteration_normalized")
                    y = df.get(metric)
                    if x is None or y is None:
                        continue
                    x = x.to_numpy(dtype=float)
                    y = y.to_numpy(dtype=float)
                    m = np.isfinite(x) & np.isfinite(y)
                    if yscale == "log":
                        m = m & (y > 0)
                    if m.sum() < 2:
                        continue
                    xs = x[m]
                    ys0 = y[m]
                    order = np.argsort(xs)
                    xs = xs[order]
                    ys0 = ys0[order]
                    xs_u, idx = np.unique(xs, return_index=True)
                    ys_u = ys0[idx]
                    if xs_u.size < 2:
                        continue
                    if yscale == "log":
                        yi = np.power(
                            10.0,
                            np.interp(x_grid, xs_u, np.log10(ys_u), left=np.nan, right=np.nan),
                        )
                    else:
                        yi = np.interp(x_grid, xs_u, ys_u, left=np.nan, right=np.nan)
                    ys.append(yi)

                if ys:
                    y_stack = np.vstack(ys)
                    y_mean = np.nanmean(y_stack, axis=0)
                    (mean_line,) = ax.plot(
                        x_grid,
                        y_mean,
                        color=color,
                        linewidth=2.4,
                        alpha=0.98,
                    )
                    # keep only one handle per method (for legend)
                    mean_handles[method] = mean_line

        ax.set_title(metric, pad=8)
        ax.set_xlabel("iteration_normalized")

        # Legend inside each subplot, upper-right, 3 lines (one entry per line)
        handles, labels = [], []
        for method in METHOD_ORDER:
            h = mean_handles.get(method)
            if h is None:
                continue
            handles.append(h)
            labels.append(METHOD_DISPLAY_NAME.get(method, method))
        if handles:
            ax.legend(
                handles,
                labels,
                loc="upper right",
                framealpha=0.9,
                ncol=1,
                borderaxespad=0.4,
                labelspacing=0.25,
                handlelength=2.0,
            )

    fig.subplots_adjust(left=0.08, bottom=0.14, top=0.88, right=0.98, wspace=0.22)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{eq}_loss_and_rel_l2_error_10runs_overlay.png"
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    return out_path


def _interp_runs_on_grid(
    mruns: list[RunInfo],
    *,
    metric: str,
    yscale: str,
    x_grid: np.ndarray,
) -> np.ndarray:
    ys: list[np.ndarray] = []
    for r in mruns:
        df = _read_curve(r.csv_path)
        x = df.get("iteration_normalized")
        y = df.get(metric)
        if x is None or y is None:
            continue
        x = x.to_numpy(dtype=float)
        y = y.to_numpy(dtype=float)
        m = np.isfinite(x) & np.isfinite(y)
        if yscale == "log":
            m = m & (y > 0)
        if m.sum() < 2:
            continue
        xs = x[m]
        ys0 = y[m]
        order = np.argsort(xs)
        xs = xs[order]
        ys0 = ys0[order]
        xs_u, idx = np.unique(xs, return_index=True)
        ys_u = ys0[idx]
        if xs_u.size < 2:
            continue
        if yscale == "log":
            yi = np.power(10.0, np.interp(x_grid, xs_u, np.log10(ys_u), left=np.nan, right=np.nan))
        else:
            yi = np.interp(x_grid, xs_u, ys_u, left=np.nan, right=np.nan)
        ys.append(yi)
    if not ys:
        return np.empty((0, x_grid.size), dtype=float)
    return np.vstack(ys)


def _mean_and_band_from_stack(
    y_stack: np.ndarray,
    *,
    yscale: str,
    band: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns (y_mean, y_low, y_high) for a given band type.

    band:
      - "std": mean ± std (in log10 space if yscale=="log")
      - "minmax": [min, max] across runs (in original space)
    """
    if y_stack.size == 0:
        n = 0 if y_stack.ndim == 0 else y_stack.shape[-1]
        nan = np.full((n,), np.nan, dtype=float)
        return nan, nan, nan

    if band not in {"std", "minmax"}:
        raise ValueError(f"Unknown band: {band}")

    # Reduce along runs (axis=0) without triggering "All-NaN slice" warnings.
    # For columns where all values are NaN, keep NaN in the result.
    def _reduce_cols(func) -> np.ndarray:
        cols_ok = np.any(~np.isnan(y_stack), axis=0)
        out = np.full((y_stack.shape[1],), np.nan, dtype=float)
        if np.any(cols_ok):
            out[cols_ok] = func(y_stack[:, cols_ok], axis=0)
        return out

    if band == "minmax":
        y_mean = _reduce_cols(np.nanmean)
        y_low = _reduce_cols(np.nanmin)
        y_high = _reduce_cols(np.nanmax)
        return y_mean, y_low, y_high

    # std band
    if yscale == "log":
        with np.errstate(divide="ignore", invalid="ignore"):
            z = np.log10(y_stack)
        z = np.where(np.isfinite(z), z, np.nan)
        cols_ok = np.any(~np.isnan(z), axis=0)
        z_mean = np.full((z.shape[1],), np.nan, dtype=float)
        z_std = np.full((z.shape[1],), np.nan, dtype=float)
        if np.any(cols_ok):
            z_mean[cols_ok] = np.nanmean(z[:, cols_ok], axis=0)
            z_std[cols_ok] = np.nanstd(z[:, cols_ok], axis=0)
        y_mean = np.power(10.0, z_mean)
        y_low = np.power(10.0, z_mean - z_std)
        y_high = np.power(10.0, z_mean + z_std)
        return y_mean, y_low, y_high

    y_mean = _reduce_cols(np.nanmean)
    y_std = _reduce_cols(np.nanstd)
    return y_mean, y_mean - y_std, y_mean + y_std


def _plot_mean_with_band(
    ax: plt.Axes,
    *,
    x: np.ndarray,
    y_mean: np.ndarray,
    y_low: np.ndarray,
    y_high: np.ndarray,
    color: str,
    band_alpha: float,
    min_count: Optional[np.ndarray] = None,
    min_required: int = 10,
    line_width: float = 2.5,
) -> object:
    """
    Plot mean + shaded band, robust to NaNs at the tail.

    Matplotlib's fill_between can create a visible vertical "cap" at the boundary
    between finite and NaN regions. We avoid this by splitting into contiguous
    finite segments (and optionally requiring enough runs contributing).
    """
    finite = np.isfinite(x) & np.isfinite(y_mean) & np.isfinite(y_low) & np.isfinite(y_high)
    if min_count is not None:
        finite = finite & (min_count >= min_required)

    if not np.any(finite):
        # Plot nothing; return a dummy invisible line handle for legend safety.
        (h,) = ax.plot([], [], color=color, linewidth=line_width, alpha=0.98)
        return h

    idx = np.flatnonzero(finite)
    # Split into contiguous segments where consecutive indices differ by 1.
    breaks = np.where(np.diff(idx) != 1)[0]
    starts = np.r_[0, breaks + 1]
    ends = np.r_[breaks + 1, idx.size]

    last_handle: object | None = None
    for s, e in zip(starts, ends):
        seg = idx[s:e]
        ax.fill_between(
            x[seg],
            y_low[seg],
            y_high[seg],
            color=color,
            alpha=band_alpha,
            linewidth=0.0,
        )
        (last_handle,) = ax.plot(
            x[seg],
            y_mean[seg],
            color=color,
            linewidth=line_width,
            alpha=0.98,
        )

    # last_handle is always set because finite has at least one True.
    return last_handle  # type: ignore[return-value]


def plot_equation_loss_and_error_overlay_with_band_one_fig(
    equation_dir: Path,
    runs: list[RunInfo],
    *,
    out_dir: Path,
    band: str,
    yscale_loss: str = "log",
    yscale_error: str = "log",
) -> Path:
    """Overlay by method with a range band (std or min-max)."""
    _setup_matplotlib_cn_font()

    eq = equation_dir.name
    by_method: dict[str, list[RunInfo]] = {m: [] for m in METHOD_ORDER}
    for r in runs:
        if r.method in by_method:
            by_method[r.method].append(r)

    metrics = [("loss", yscale_loss), ("rel_l2_error", yscale_error)]

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(13.0, 4.8),
        dpi=180,
        sharex=True,
        constrained_layout=False,
    )

    if band == "std":
        band_title = "mean ± 1 std"
    elif band == "minmax":
        band_title = "min–max range"
    else:
        raise ValueError(f"Unknown band: {band}")

    fig.suptitle(f"{eq} | {band_title} (x = iteration_normalized)", y=1.02)
    x_grid = np.linspace(0.0, 1.0, 600, dtype=float)

    for col, (metric, yscale) in enumerate(metrics):
        ax = axes[col]
        ax.set_yscale(yscale)
        ax.set_ylabel(metric)
        ax.grid(False)

        # Keep legend handles per-axis to avoid cross-metric handle reuse.
        mean_handles: dict[str, object] = {}

        for method in METHOD_ORDER:
            mruns = _pick_10_runs(by_method.get(method, []))
            if not mruns:
                continue
            color = METHOD_COLOR.get(method, "#333333")

            y_stack = _interp_runs_on_grid(mruns, metric=metric, yscale=yscale, x_grid=x_grid)
            y_mean, y_low, y_high = _mean_and_band_from_stack(y_stack, yscale=yscale, band=band)
            # Require enough runs contributing to avoid tail artifacts where many runs are NaN.
            counts = np.sum(~np.isnan(y_stack), axis=0) if y_stack.size else None
            min_required = 2 if band == "std" else 1
            mean_line = _plot_mean_with_band(
                ax,
                x=x_grid,
                y_mean=y_mean,
                y_low=y_low,
                y_high=y_high,
                color=color,
                band_alpha=0.18 if band == "std" else 0.14,
                min_count=counts,
                min_required=min_required,
                line_width=2.5,
            )
            mean_handles[method] = mean_line

        ax.set_title(metric, pad=8)
        ax.set_xlabel("iteration_normalized")

        handles, labels = [], []
        for method in METHOD_ORDER:
            h = mean_handles.get(method)
            if h is None:
                continue
            handles.append(h)
            labels.append(METHOD_DISPLAY_NAME.get(method, method))
        if handles:
            ax.legend(
                handles,
                labels,
                loc="upper right",
                framealpha=0.9,
                ncol=1,
                borderaxespad=0.4,
                labelspacing=0.25,
                handlelength=2.0,
            )

    fig.subplots_adjust(left=0.08, bottom=0.14, top=0.88, right=0.98, wspace=0.22)
    out_dir.mkdir(parents=True, exist_ok=True)
    suffix = "std" if band == "std" else "minmax"
    out_path = out_dir / f"{eq}_loss_and_rel_l2_error_10runs_overlay_{suffix}.png"
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    return out_path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoints_dir", type=str, default="checkpoints", help="checkpoints 根目录")
    ap.add_argument("--equation", type=str, default=None, help="只画指定方程目录（例如 heat3d_exact）")
    ap.add_argument(
        "--csv_name",
        type=str,
        default="training_curve_iteration_normalized.csv",
        help="每个 run 目录下的曲线 CSV 文件名",
    )
    ap.add_argument("--out_dirname", type=str, default="plots_10runs", help="输出子目录名（写入到方程目录下）")
    ap.add_argument(
        "--layout",
        type=str,
        default="overlay",
        choices=["overlay", "by_method"],
        help="overlay=四种方法叠加在同一图里; by_method=每种方法单独子图",
    )
    args = ap.parse_args()

    checkpoints_dir = Path(args.checkpoints_dir)
    if not checkpoints_dir.exists():
        raise FileNotFoundError(f"checkpoints_dir 不存在: {checkpoints_dir}")

    runs = list(iter_run_infos(checkpoints_dir, equation=args.equation, csv_name=args.csv_name))
    if not runs:
        print("未找到可用的 run（需要 run_dir/output.log + run_dir/<csv_name>）")
        return 1

    by_eq: dict[str, list[RunInfo]] = {}
    for r in runs:
        by_eq.setdefault(r.equation, []).append(r)

    for eq, eq_runs in sorted(by_eq.items(), key=lambda kv: kv[0]):
        eq_dir = checkpoints_dir / eq
        out_dir = eq_dir / args.out_dirname
        if args.layout == "by_method":
            out = plot_equation_loss_and_error_one_fig(
                eq_dir,
                eq_runs,
                out_dir=out_dir,
                yscale_loss="log",
                yscale_error="log",
            )
            print(f"Saved: {out}")
        else:
            out = plot_equation_loss_and_error_overlay_one_fig(
                eq_dir,
                eq_runs,
                out_dir=out_dir,
                yscale_loss="log",
                yscale_error="log",
            )
            print(f"Saved: {out}")

            out_std = plot_equation_loss_and_error_overlay_with_band_one_fig(
                eq_dir,
                eq_runs,
                out_dir=out_dir,
                band="std",
                yscale_loss="log",
                yscale_error="log",
            )
            print(f"Saved: {out_std}")

            out_minmax = plot_equation_loss_and_error_overlay_with_band_one_fig(
                eq_dir,
                eq_runs,
                out_dir=out_dir,
                band="minmax",
                yscale_loss="log",
                yscale_error="log",
            )
            print(f"Saved: {out_minmax}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
