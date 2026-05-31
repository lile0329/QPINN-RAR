from __future__ import annotations

import os
import re

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def _interp_1d_for_plot(
    x: np.ndarray,
    y: np.ndarray,
    x_new: np.ndarray,
    *,
    yscale: str,
) -> np.ndarray:
    """
    用于绘图的 1D 插值：自动忽略 NaN；log 轴下在 log10 空间做线性插值，视觉更平滑。
    要求 x_new 单调递增；若有效点 < 2，则返回全 NaN（交给调用方回退到原始散点/折线）。
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    x_new = np.asarray(x_new, dtype=float)

    m = np.isfinite(x) & np.isfinite(y)
    if m.sum() < 2:
        return np.full_like(x_new, np.nan, dtype=float)

    x0 = x[m]
    y0 = y[m]
    order = np.argsort(x0)
    x0 = x0[order]
    y0 = y0[order]

    # 去重：np.interp 要求 x0 严格单调递增
    x_unique, idx = np.unique(x0, return_index=True)
    y_unique = y0[idx]
    if x_unique.size < 2:
        return np.full_like(x_new, np.nan, dtype=float)

    if yscale == "log":
        # y 可能已经被置为 NaN（<=0），这里再保险一下
        my = np.isfinite(y_unique) & (y_unique > 0)
        if my.sum() < 2:
            return np.full_like(x_new, np.nan, dtype=float)
        x_unique = x_unique[my]
        y_unique = y_unique[my]
        order2 = np.argsort(x_unique)
        x_unique = x_unique[order2]
        y_unique = y_unique[order2]
        y_log = np.log10(y_unique)
        y_new_log = np.interp(x_new, x_unique, y_log, left=np.nan, right=np.nan)
        return np.power(10.0, y_new_log)

    return np.interp(x_new, x_unique, y_unique, left=np.nan, right=np.nan)


def _setup_matplotlib_cn_font() -> None:
    """
    与原脚本一致：设置中文字体，避免乱码。
    """
    plt.rcParams["font.sans-serif"] = [
        "Microsoft YaHei",
        "SimHei",
        "Noto Sans CJK SC",
        "Arial Unicode MS",
        "DejaVu Sans",
    ]
    plt.rcParams["axes.unicode_minus"] = False


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

METHOD_BAND_STYLE = {
    "PINN": {"hatch": "\\\\\\", "alpha": 0.16},
    "PINN-RAR": {"hatch": "...", "alpha": 0.18},
    "QPINN": {"hatch": "///", "alpha": 0.18},
    "QPINN-RAR": {"hatch": "xxx", "alpha": 0.16},
}

METHOD_ORDER = ["PINN", "PINN-RAR", "QPINN", "QPINN-RAR"]
METHOD_ALIASES = {
    "DV": "QPINN-RAR",
    "pinn-rar-s": "PINN-RAR",
}


def _parse_run_info_from_log(log_path: str) -> tuple[int | None, str | None]:
    seed: int | None = None
    solver: str | None = None
    use_rar: bool | None = None

    with open(log_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            if seed is None:
                m = re.search(r"Random seed set to:\s*(\d+)", line)
                if m:
                    seed = int(m.group(1))
                else:
                    m = re.search(r"^\s*seed\s*:\s*(\d+)\s*$", line)
                    if m:
                        seed = int(m.group(1))

            if solver is None:
                m = re.search(r"^\s*solver\s*:\s*([A-Za-z0-9_]+)\s*$", line)
                if m:
                    solver = m.group(1).strip()
                elif re.search(r"\bUsing\s+DV\s+Solver\b", line):
                    solver = "DV"
                elif re.search(r"\bUsing\s+Classical\s+Solver2\b", line):
                    solver = "Classical2"

            if use_rar is None:
                m = re.search(r"^\s*use_rar\s*:\s*(True|False)\s*$", line)
                if m:
                    use_rar = m.group(1) == "True"

            if seed is not None and solver is not None and use_rar is not None:
                break

    if solver is None:
        return seed, None
    if solver.upper() == "DV":
        return seed, "QPINN-RAR" if use_rar else "QPINN"
    if solver.lower() in {"classical2", "classical"}:
        return seed, "PINN-RAR" if use_rar else "PINN"
    return seed, METHOD_ALIASES.get(solver, solver)


def _iter_normalized_run_curves(eq_dir: str, metric_suffix: str) -> dict[str, list[tuple[int | None, str, np.ndarray, np.ndarray]]]:
    by_method: dict[str, list[tuple[int | None, str, np.ndarray, np.ndarray]]] = {m: [] for m in METHOD_ORDER}

    for name in sorted(os.listdir(eq_dir)):
        run_dir = os.path.join(eq_dir, name)
        if not os.path.isdir(run_dir):
            continue
        csv_path = os.path.join(run_dir, "training_curve_iteration_normalized.csv")
        log_path = os.path.join(run_dir, "output.log")
        if not os.path.exists(csv_path) or not os.path.exists(log_path):
            continue

        seed, method = _parse_run_info_from_log(log_path)
        if method not in by_method:
            continue

        df = pd.read_csv(csv_path)
        if metric_suffix not in df.columns:
            continue
        if "iteration_normalized" in df.columns:
            x = pd.to_numeric(df["iteration_normalized"], errors="coerce").to_numpy(dtype=float)
        elif "iteration" in df.columns:
            it = pd.to_numeric(df["iteration"], errors="coerce").to_numpy(dtype=float)
            max_it = np.nanmax(it)
            if not np.isfinite(max_it) or max_it <= 0:
                continue
            x = it / max_it
        else:
            continue

        # Make stale/rounded normalized files robust: every run ends at x=1.
        max_x = np.nanmax(x)
        if np.isfinite(max_x) and max_x > 0:
            x = x / max_x

        y = pd.to_numeric(df[metric_suffix], errors="coerce").to_numpy(dtype=float)
        by_method[method].append((seed, name, x, y))

    def run_key(item: tuple[int | None, str, np.ndarray, np.ndarray]) -> tuple[int, str]:
        seed, run_id, _, _ = item
        return (seed if seed is not None else 10**18, run_id)

    return {m: sorted(curves, key=run_key)[:10] for m, curves in by_method.items()}


def _interp_run_on_grid(x: np.ndarray, y: np.ndarray, x_grid: np.ndarray, *, yscale: str) -> np.ndarray:
    m = np.isfinite(x) & np.isfinite(y)
    if yscale == "log":
        m = m & (y > 0)
    if m.sum() < 2:
        return np.full_like(x_grid, np.nan, dtype=float)

    xs = x[m]
    ys = y[m]
    order = np.argsort(xs)
    xs = xs[order]
    ys = ys[order]
    xs_u, idx = np.unique(xs, return_index=True)
    ys_u = ys[idx]
    if xs_u.size < 2:
        return np.full_like(x_grid, np.nan, dtype=float)

    if yscale == "log":
        yi = np.interp(x_grid, xs_u, np.log10(ys_u), left=np.nan, right=np.nan)
        return np.power(10.0, yi)
    return np.interp(x_grid, xs_u, ys_u, left=np.nan, right=np.nan)


def _plot_normalized_run_ranges_on_ax(
    ax: plt.Axes,
    *,
    eq_dir: str,
    metric_suffix: str,
    title: str,
    yscale: str,
    plot_mean: bool = True,
    show_ylabel: bool = True,
    show_xlabel: bool = True,
    show_legend: bool = True,
) -> None:
    grouped = _iter_normalized_run_curves(eq_dir, metric_suffix)
    x_grid = np.linspace(0.0, 1.0, 600, dtype=float)

    for method in METHOD_ORDER:
        curves = grouped.get(method, [])
        if not curves:
            continue

        stack = np.vstack([
            _interp_run_on_grid(x, y, x_grid, yscale=yscale)
            for _, _, x, y in curves
        ])
        valid = np.isfinite(stack)
        count = valid.sum(axis=0)
        full = count == len(curves)

        y_mean = np.full_like(x_grid, np.nan, dtype=float)
        y_min = np.full_like(x_grid, np.nan, dtype=float)
        y_max = np.full_like(x_grid, np.nan, dtype=float)
        if np.any(full):
            y_mean[full] = np.nanmean(stack[:, full], axis=0)
            y_min[full] = np.nanmin(stack[:, full], axis=0)
            y_max[full] = np.nanmax(stack[:, full], axis=0)

        color = METHOD_COLOR.get(method, None)
        label = METHOD_DISPLAY_NAME.get(method, method)
        band_style = METHOD_BAND_STYLE.get(method, {})
        band_alpha = float(band_style.get("alpha", 0.18))
        band_hatch = band_style.get("hatch", None)

        if plot_mean:
            ax.fill_between(
                x_grid,
                y_min,
                y_max,
                facecolor=color,
                alpha=band_alpha,
                edgecolor=color,
                hatch=band_hatch,
                linewidth=0.0,
                label="_nolegend_",
            )
            ax.plot(x_grid, y_mean, color=color, linewidth=2.0, label=f"{label} (mean)")
        else:
            ax.fill_between(
                x_grid,
                y_min,
                y_max,
                facecolor=color,
                alpha=max(band_alpha, 0.20),
                edgecolor=color,
                hatch=band_hatch,
                linewidth=0.0,
                label=f"{label} (min-max)",
            )

    ax.set_title(title)
    if show_xlabel:
        ax.set_xlabel("normalized iteration (0 = start, 1 = final)")
    if show_ylabel:
        ax.set_ylabel(metric_suffix)
    ax.set_xlim(0.0, 1.0)
    ax.set_yscale(yscale)
    ax.grid(False)

    if show_legend:
        handles, labels = ax.get_legend_handles_labels()
        if handles:
            ax.legend(handles, labels, loc="best", framealpha=0.9)


def _group_columns_by_method(columns: list[str], metric_suffix: str) -> dict[str, list[str]]:
    """
    {method}__seed{N}__{metric_suffix}
    例如：
      QPINN__seed3__loss
      PINN-RAR__seed20__rel_l2_error
    """
    pat = re.compile(rf"^(?P<method>.+?)__seed\d+__{re.escape(metric_suffix)}$")
    grouped: dict[str, list[str]] = {}
    for c in columns:
        m = pat.match(c)
        if not m:
            continue
        method = METHOD_ALIASES.get(m.group("method"), m.group("method"))
        grouped.setdefault(method, []).append(c)
    return grouped


def _normalize_iteration(iteration: np.ndarray) -> np.ndarray:
    """
    将原始 iteration 线性归一化到 [0, 1] 区间。
    0 = 最初的迭代（最小 iteration），1 = 收敛迭代（最大 iteration）。
    """
    x = pd.to_numeric(iteration, errors="coerce").to_numpy(dtype=float)
    # 去掉 NaN 后再取 min/max，防止全 NaN 抛错
    finite_mask = np.isfinite(x)
    if not finite_mask.any():
        # 全是 NaN 的极端情况，就直接返回全 NaN
        return np.full_like(x, np.nan)

    x_finite = x[finite_mask]
    x_min = np.min(x_finite)
    x_max = np.max(x_finite)

    if x_max == x_min:
        # 所有 iteration 一样时，统一映射到 0
        x_norm = np.zeros_like(x, dtype=float)
    else:
        x_norm = (x - x_min) / (x_max - x_min)

    # 保持原来 NaN 位置为 NaN
    x_norm[~finite_mask] = np.nan
    return x_norm


def _plot_range_on_ax(
    ax: plt.Axes,
    *,
    df: pd.DataFrame,
    metric_suffix: str,
    title: str,
    yscale: str,
    plot_mean: bool = True,
    min_count: int = 10,
    show_count: bool = False,
    show_ylabel: bool = True,
    show_xlabel: bool = True,
    show_legend: bool = True,
) -> None:
    """
    在指定 ax 上绘制：横坐标为归一化后的 iteration ∈ [0, 1]。
    用于“单张图”拼子图时复用。
    """
    if "iteration" not in df.columns:
        raise ValueError("CSV 缺少 'iteration' 列")

    grouped = _group_columns_by_method([c for c in df.columns if c != "iteration"], metric_suffix)
    if not grouped:
        raise ValueError(f"没有找到形如 {{method}}__seed{{N}}__{metric_suffix} 的列")

    target_methods = METHOD_ORDER
    existing_methods = [m for m in target_methods if m in grouped]
    if not existing_methods:
        raise ValueError(f"在 CSV 中未找到目标方法列：{target_methods}")

    ax2 = None
    for method in existing_methods:
        cols = grouped[method]
        y = df[cols].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)

        # 对“该方法有有效数据”的迭代点单独做一次 0–1 归一化，
        # 保证每条曲线（以及其阴影带）自己的 x 轴都从 0 走到 1。
        it_raw = pd.to_numeric(df["iteration"], errors="coerce").to_numpy(dtype=float)

        if yscale == "log":
            y = np.where(y > 0, y, np.nan)

        finite = np.isfinite(y)
        has_any = finite.any(axis=1)
        y_cnt = finite.sum(axis=1)
        support = has_any
        if plot_mean and min_count > 1:
            support = y_cnt >= min_count

        it_finite_mask = np.isfinite(it_raw) & support
        if not it_finite_mask.any():
            x_method = np.full_like(it_raw, np.nan, dtype=float)
        else:
            it_min = np.min(it_raw[it_finite_mask])
            it_max = np.max(it_raw[it_finite_mask])
            if it_max == it_min:
                x_method = np.zeros_like(it_raw, dtype=float)
            else:
                x_method = (it_raw - it_min) / (it_max - it_min)
            x_method[~it_finite_mask] = np.nan

        y_min = np.min(np.where(finite, y, np.inf), axis=1)
        y_min = np.where(has_any, y_min, np.nan)

        y_max = np.max(np.where(finite, y, -np.inf), axis=1)
        y_max = np.where(has_any, y_max, np.nan)

        y_sum = np.sum(np.where(finite, y, 0.0), axis=1)
        y_mean = np.divide(y_sum, y_cnt, out=np.full_like(y_sum, np.nan, dtype=float), where=y_cnt > 0)

        if min_count > 1:
            enough = y_cnt >= min_count
            y_mean = np.where(enough, y_mean, np.nan)

        color = METHOD_COLOR.get(method, None)
        label = METHOD_DISPLAY_NAME.get(method, method)
        band_style = METHOD_BAND_STYLE.get(method, {})
        band_alpha = float(band_style.get("alpha", 0.18))
        band_hatch = band_style.get("hatch", None)

        # 关键：对断点（NaN 段）做插值，让曲线与误差带连续。
        # 为了不引入额外依赖，这里用线性插值；log 轴在 log10 空间插值更“平滑”。
        # 若有效点不足，则回退到原始 x/y（保持原逻辑）。
        x_grid = np.linspace(0.0, 1.0, 600, dtype=float)
        y_mean_i = _interp_1d_for_plot(x_method, y_mean, x_grid, yscale=yscale)
        y_min_i = _interp_1d_for_plot(x_method, y_min, x_grid, yscale=yscale)
        y_max_i = _interp_1d_for_plot(x_method, y_max, x_grid, yscale=yscale)

        can_interp = np.isfinite(y_mean_i).any() and np.isfinite(y_min_i).any() and np.isfinite(y_max_i).any()
        if can_interp:
            # 确保上下界关系不被数值插值打破
            y_lo = np.minimum(y_min_i, y_max_i)
            y_hi = np.maximum(y_min_i, y_max_i)
            x_plot = x_grid
            y_mean_plot = y_mean_i
            y_min_plot = y_lo
            y_max_plot = y_hi
        else:
            x_plot = x_method
            y_mean_plot = y_mean
            y_min_plot = y_min
            y_max_plot = y_max

        if plot_mean:
            ax.fill_between(
                x_plot,
                y_min_plot,
                y_max_plot,
                facecolor=color,
                alpha=band_alpha,
                edgecolor=color,
                linewidth=0.8,
                hatch=band_hatch,
                zorder=1,
            )
            ax.plot(x_plot, y_mean_plot, color=color, linewidth=2.0, label=f"{label} (mean)")
        else:
            ax.fill_between(
                x_plot,
                y_min_plot,
                y_max_plot,
                facecolor=color,
                alpha=max(band_alpha, 0.20),
                edgecolor=color,
                linewidth=0.8,
                hatch=band_hatch,
                label=f"{label} (min–max)",
                zorder=1,
            )

        if show_count:
            if ax2 is None:
                ax2 = ax.twinx()
                ax2.set_ylabel("valid runs (count)")
                ax2.grid(False)
            ax2.plot(
                x_method,
                y_cnt.astype(float),
                color=color,
                linewidth=1.0,
                alpha=0.35,
                linestyle="-",
                label=f"{label} (count)",
            )

    ax.set_title(title)
    if show_xlabel:
        ax.set_xlabel("normalized iteration (0 = 初始, 1 = 收敛)")
    if show_ylabel:
        ax.set_ylabel(metric_suffix)
    ax.set_yscale(yscale)
    ax.grid(False)

    if show_legend:
        h1, l1 = ax.get_legend_handles_labels()
        if ax2 is not None:
            h2, l2 = ax2.get_legend_handles_labels()
            handles, labels = h1 + h2, l1 + l2
        else:
            handles, labels = h1, l1

        ax.legend(
            handles,
            labels,
            loc="lower left",
            bbox_to_anchor=(0.01, 0.01),
            borderaxespad=0.0,
            framealpha=0.9,
        )


def _plot_range(
    df: pd.DataFrame,
    metric_suffix: str,
    title: str,
    out_path: str,
    yscale: str,
    plot_mean: bool = True,
    min_count: int = 10,
    show_count: bool = False,
) -> None:
    """
    单个 equation 单张图输出。
    """
    _setup_matplotlib_cn_font()
    fig, ax = plt.subplots(figsize=(10, 5.5), dpi=160)
    _plot_range_on_ax(
        ax,
        df=df,
        metric_suffix=metric_suffix,
        title=title,
        yscale=yscale,
        plot_mean=plot_mean,
        min_count=min_count,
        show_count=show_count,
        show_ylabel=True,
        show_xlabel=True,
        show_legend=True,
    )

    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved: {out_path}")


def _plot_all_equations_one_figure_mean_normiter(
    *,
    eq_dirs: list[str],
    metric_suffix: str,
    yscale: str,
    out_path: str,
) -> None:
    """
    把多个 equation 画到同一张 figure（多子图）里。
    只画 mean（以及对应的 min–max 阴影带），横坐标使用 normalized iteration。
    """
    _setup_matplotlib_cn_font()
    n = len(eq_dirs)
    fig, axs = plt.subplots(1, n, figsize=(5.4 * n, 4.8), dpi=170, squeeze=False, sharex=True)
    axs1 = axs[0]

    all_handles: list = []
    all_labels: list[str] = []

    for i, eq_dir in enumerate(eq_dirs):
        ax = axs1[i]
        base = os.path.basename(eq_dir)
        csv_path = os.path.join(eq_dir, f"merged_{metric_suffix}.csv")
        if not os.path.exists(csv_path):
            ax.set_title(f"{base} | [missing csv]")
            ax.axis("off")
            continue

        df = pd.read_csv(csv_path)
        _plot_range_on_ax(
            ax,
            df=df,
            metric_suffix=metric_suffix,
            title=base,
            yscale=yscale,
            plot_mean=True,
            min_count=10,
            show_count=False,
            show_ylabel=(i == 0),
            show_xlabel=True,
            show_legend=False,  # 统一 legend 放在整张图底部
        )

        h, l = ax.get_legend_handles_labels()
        for hh, ll in zip(h, l):
            if ll not in all_labels:
                all_handles.append(hh)
                all_labels.append(ll)

    fig.suptitle(f"All equations | {metric_suffix} (mean with min–max band, normalized iteration)", y=1.03)
    if all_handles:
        fig.legend(
            all_handles,
            all_labels,
            loc="lower center",
            ncol=min(4, len(all_labels)),
            framealpha=0.9,
            bbox_to_anchor=(0.5, -0.02),
        )

    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


def _plot_for_equation_dir(eq_dir: str) -> None:
    # loss
    loss_csv = os.path.join(eq_dir, "merged_loss.csv")
    if os.path.exists(loss_csv):
        df_loss = pd.read_csv(loss_csv)
        base = os.path.basename(eq_dir)
        _plot_range(
            df=df_loss,
            metric_suffix="loss",
            title=f"{base} | Loss range (min–max shaded, normalized iteration)",
            out_path=os.path.join(eq_dir, f"loss_range_normiter_{base}.png"),
            yscale="log",
        )
        _plot_range(
            df=df_loss,
            metric_suffix="loss",
            title=f"{base} | Loss range (min–max shaded, no mean, normalized iteration)",
            out_path=os.path.join(eq_dir, f"loss_range_no_mean_normiter_{base}.png"),
            yscale="log",
            plot_mean=False,
        )
    else:
        print(f"[skip] not found: {loss_csv}")

    # relative error
    rel_csv = os.path.join(eq_dir, "merged_rel_l2_error.csv")
    if os.path.exists(rel_csv):
        df_rel = pd.read_csv(rel_csv)
        base = os.path.basename(eq_dir)
        _plot_range(
            df=df_rel,
            metric_suffix="rel_l2_error",
            title=f"{base} | Relative L2 error range (min–max shaded, normalized iteration)",
            out_path=os.path.join(eq_dir, f"rel_l2_error_range_normiter_{base}.png"),
            yscale="log",
        )
        _plot_range(
            df=df_rel,
            metric_suffix="rel_l2_error",
            title=f"{base} | Relative L2 error range (min–max shaded, no mean, normalized iteration)",
            out_path=os.path.join(eq_dir, f"rel_l2_error_range_no_mean_normiter_{base}.png"),
            yscale="log",
            plot_mean=False,
        )
    else:
        print(f"[skip] not found: {rel_csv}")


def _plot_all_equations_one_figure_mean_normiter(
    *,
    eq_dirs: list[str],
    metric_suffix: str,
    yscale: str,
    out_path: str,
) -> None:
    """Plot from per-run normalized CSVs, so x=1 is each run's final iteration."""
    _setup_matplotlib_cn_font()
    n = len(eq_dirs)
    fig, axs = plt.subplots(1, n, figsize=(5.4 * n, 4.8), dpi=170, squeeze=False, sharex=True)
    axs1 = axs[0]

    all_handles: list = []
    all_labels: list[str] = []

    for i, eq_dir in enumerate(eq_dirs):
        ax = axs1[i]
        base = os.path.basename(eq_dir)
        _plot_normalized_run_ranges_on_ax(
            ax,
            eq_dir=eq_dir,
            metric_suffix=metric_suffix,
            title=base,
            yscale=yscale,
            plot_mean=True,
            show_ylabel=(i == 0),
            show_xlabel=True,
            show_legend=False,
        )

        h, l = ax.get_legend_handles_labels()
        for hh, ll in zip(h, l):
            if ll not in all_labels:
                all_handles.append(hh)
                all_labels.append(ll)

    fig.suptitle(f"All equations | {metric_suffix} (10-run mean with min-max band, normalized per run)", y=1.03)
    if all_handles:
        fig.legend(
            all_handles,
            all_labels,
            loc="lower center",
            ncol=min(4, len(all_labels)),
            framealpha=0.9,
            bbox_to_anchor=(0.5, -0.02),
        )

    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


def _plot_for_equation_dir(eq_dir: str) -> None:
    """Plot each equation from per-run normalized CSVs."""
    base = os.path.basename(eq_dir)
    _setup_matplotlib_cn_font()

    for metric_suffix, yscale, pretty in [
        ("loss", "log", "Loss"),
        ("rel_l2_error", "log", "Relative L2 error"),
    ]:
        fig, ax = plt.subplots(figsize=(10, 5.5), dpi=160)
        _plot_normalized_run_ranges_on_ax(
            ax,
            eq_dir=eq_dir,
            metric_suffix=metric_suffix,
            title=f"{base} | {pretty} range (min-max shaded, normalized per run)",
            yscale=yscale,
            plot_mean=True,
        )
        fig.tight_layout()
        out_path = os.path.join(eq_dir, f"{metric_suffix}_range_normiter_{base}.png")
        fig.savefig(out_path, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved: {out_path}")

        fig, ax = plt.subplots(figsize=(10, 5.5), dpi=160)
        _plot_normalized_run_ranges_on_ax(
            ax,
            eq_dir=eq_dir,
            metric_suffix=metric_suffix,
            title=f"{base} | {pretty} range (min-max shaded, no mean, normalized per run)",
            yscale=yscale,
            plot_mean=False,
        )
        fig.tight_layout()
        out_path = os.path.join(eq_dir, f"{metric_suffix}_range_no_mean_normiter_{base}.png")
        fig.savefig(out_path, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved: {out_path}")


def main() -> None:
    here = os.path.dirname(os.path.abspath(__file__))
    equations = ["burgers_exact", "diffusion_exact", "heat3d_exact"]

    # 1) 汇总：多个 equation 放到同一张图（mean 口径）
    eq_dirs = [os.path.join(here, eq) for eq in equations if os.path.isdir(os.path.join(here, eq))]
    if eq_dirs:
        _plot_all_equations_one_figure_mean_normiter(
            eq_dirs=eq_dirs,
            metric_suffix="loss",
            yscale="log",
            out_path=os.path.join(here, "loss_range_all_equations_one_figure_mean_normiter.png"),
        )
        _plot_all_equations_one_figure_mean_normiter(
            eq_dirs=eq_dirs,
            metric_suffix="rel_l2_error",
            yscale="log",
            out_path=os.path.join(here, "rel_l2_error_range_all_equations_one_figure_mean_normiter.png"),
        )

    # 2) 兼容保留：每个 equation 单独出图（与原脚本一致）
    for eq in equations:
        eq_dir = os.path.join(here, eq)
        if not os.path.isdir(eq_dir):
            print(f"[skip] not a dir: {eq_dir}")
            continue
        _plot_for_equation_dir(eq_dir)


if __name__ == "__main__":
    main()

