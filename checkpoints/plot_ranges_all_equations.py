import os
import re

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def _setup_matplotlib_cn_font() -> None:
    """
    Windows 下 Matplotlib 默认字体常不含中文，导致标题/标签显示为方块。
    这里提供一个通用的中文字体回退列表；若某个字体不存在，Matplotlib 会自动尝试下一个。
    """
    plt.rcParams["font.sans-serif"] = [
        "Microsoft YaHei",  # 微软雅黑（Win 常见）
        "SimHei",  # 黑体（Win 常见）
        "Noto Sans CJK SC",  # 思源黑体（若安装）
        "Arial Unicode MS",  # 若安装
        "DejaVu Sans",  # 最后兜底
    ]
    plt.rcParams["axes.unicode_minus"] = False  # 避免负号显示为方块


METHOD_DISPLAY_NAME = {
    "PINN": "PINN",
    "PINN-RAR": "PINN-RAR",
    "QPINN": "QPINN",
    "QPINN-RAR": "QPINN-RAR",
}

METHOD_COLOR = {
    # Colorblind-friendly (Okabe–Ito palette)
    # https://jfly.uni-koeln.de/color/
    "PINN": "#E69F00",  # orange
    "PINN-RAR": "#009E73",  # bluish green
    "QPINN": "#0072B2",  # blue
    "QPINN-RAR": "#CC79A7",  # reddish purple
}

# 阴影带样式：用不同 hatch + 边框来提升重叠区域的可区分度
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

# 当图例条目较多时，默认放到图外右侧，避免挡线
LEGEND_OUTSIDE_MIN_ITEMS = 5


def _group_columns_by_method(columns: list[str], metric_suffix: str) -> dict[str, list[str]]:
    """
    Expected column format:
      {method}__seed{N}__{metric_suffix}
    e.g.
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
    _setup_matplotlib_cn_font()
    if "iteration" not in df.columns:
        raise ValueError("CSV 缺少 'iteration' 列")

    grouped = _group_columns_by_method([c for c in df.columns if c != "iteration"], metric_suffix)
    if not grouped:
        raise ValueError(f"没有找到形如 {{method}}__seed{{N}}__{metric_suffix} 的列")

    target_methods = METHOD_ORDER
    existing_methods = [m for m in target_methods if m in grouped]
    if not existing_methods:
        raise ValueError(f"在 CSV 中未找到目标方法列：{target_methods}")

    x = pd.to_numeric(df["iteration"], errors="coerce").to_numpy()

    fig, ax = plt.subplots(figsize=(10, 5.5), dpi=160)
    ax2 = None
    for method in existing_methods:
        cols = grouped[method]
        y = df[cols].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)

        # 对 log 轴做个保护：把非正数/NaN 变成 NaN，避免报错或把 0 画进去
        if yscale == "log":
            y = np.where(y > 0, y, np.nan)

        # 避免出现 “All-NaN slice encountered / Mean of empty slice” 的 RuntimeWarning：
        # 对每一行（同一 iteration 的多 seed）做安全归约；若该行全是 NaN，则结果置为 NaN。
        finite = np.isfinite(y)
        has_any = finite.any(axis=1)

        y_min = np.min(np.where(finite, y, np.inf), axis=1)
        y_min = np.where(has_any, y_min, np.nan)

        y_max = np.max(np.where(finite, y, -np.inf), axis=1)
        y_max = np.where(has_any, y_max, np.nan)

        y_sum = np.sum(np.where(finite, y, 0.0), axis=1)
        y_cnt = finite.sum(axis=1)
        y_mean = np.divide(y_sum, y_cnt, out=np.full_like(y_sum, np.nan, dtype=float), where=y_cnt > 0)

        # 当某些 run 提前结束时，尾部 iteration 的有效样本数会下降。
        # mean 曲线默认只保留 10 个 run 都有效的位置，避免尾部被少数 run 拉偏。
        if min_count > 1:
            enough = y_cnt >= min_count
            y_mean = np.where(enough, y_mean, np.nan)

        color = METHOD_COLOR.get(method, None)
        label = METHOD_DISPLAY_NAME.get(method, method)
        band_style = METHOD_BAND_STYLE.get(method, {})
        band_alpha = float(band_style.get("alpha", 0.18))
        band_hatch = band_style.get("hatch", None)

        if plot_mean:
            ax.fill_between(
                x,
                y_min,
                y_max,
                facecolor=color,
                alpha=band_alpha,
                edgecolor=color,
                linewidth=0.8,
                hatch=band_hatch,
                zorder=1,
            )
            ax.plot(x, y_mean, color=color, linewidth=2.0, label=f"{label} (mean)")
        else:
            # no-mean: only show the min–max band
            ax.fill_between(
                x,
                y_min,
                y_max,
                facecolor=color,
                alpha=max(band_alpha, 0.20),
                edgecolor=color,
                linewidth=0.8,
                hatch=band_hatch,
                label=f"{label} (min–max)",
                zorder=1,
            )

        # 可选：在右轴画出每个 iteration 的有效样本数（可帮助解释尾部口径变化）
        if show_count:
            if ax2 is None:
                ax2 = ax.twinx()
                ax2.set_ylabel("valid runs (count)")
                ax2.grid(False)
            # 不使用虚线，改为实线展示样本数
            ax2.plot(
                x,
                y_cnt.astype(float),
                color=color,
                linewidth=1.0,
                alpha=0.35,
                linestyle="-",
                label=f"{label} (count)",
            )

    ax.set_title(title)
    ax.set_xlabel("iteration")
    ax.set_ylabel(metric_suffix)
    ax.set_yscale(yscale)
    # 取消背景虚线网格
    ax.grid(False)

    # 图例：仅保留方法曲线，不把 count 等辅助曲线塞进图例
    h1, l1 = ax.get_legend_handles_labels()
    keep_methods = METHOD_ORDER
    method_names = [METHOD_DISPLAY_NAME.get(m, m) for m in keep_methods]
    keep_labels = set()
    for name in method_names:
        if plot_mean:
            keep_labels.add(f"{name} (mean)")
        else:
            keep_labels.add(f"{name} (min–max)")

    handles, labels = [], []
    for hh, ll in zip(h1, l1):
        if ll in keep_labels:
            handles.append(hh)
            labels.append(ll)

    ax.legend(
        handles,
        labels,
        loc="lower left",
        bbox_to_anchor=(0.01, 0.01),
        borderaxespad=0.0,
        framealpha=0.9,
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
        _plot_range(
            df=df_loss,
            metric_suffix="loss",
            title=f"{os.path.basename(eq_dir)} | Loss range (min–max shaded)",
            out_path=os.path.join(eq_dir, f"loss_range_{os.path.basename(eq_dir)}.png"),
            yscale="log",
        )
        _plot_range(
            df=df_loss,
            metric_suffix="loss",
            title=f"{os.path.basename(eq_dir)} | Loss range (min–max shaded) (no mean)",
            out_path=os.path.join(eq_dir, f"loss_range_no_mean_{os.path.basename(eq_dir)}.png"),
            yscale="log",
            plot_mean=False,
        )
    else:
        print(f"[skip] not found: {loss_csv}")

    # relative error
    rel_csv = os.path.join(eq_dir, "merged_rel_l2_error.csv")
    if os.path.exists(rel_csv):
        df_rel = pd.read_csv(rel_csv)
        _plot_range(
            df=df_rel,
            metric_suffix="rel_l2_error",
            title=f"{os.path.basename(eq_dir)} | Relative L2 error range (min–max shaded)",
            out_path=os.path.join(eq_dir, f"rel_l2_error_range_{os.path.basename(eq_dir)}.png"),
            yscale="log",
        )
        _plot_range(
            df=df_rel,
            metric_suffix="rel_l2_error",
            title=f"{os.path.basename(eq_dir)} | Relative L2 error range (min–max shaded) (no mean)",
            out_path=os.path.join(eq_dir, f"rel_l2_error_range_no_mean_{os.path.basename(eq_dir)}.png"),
            yscale="log",
            plot_mean=False,
        )
    else:
        print(f"[skip] not found: {rel_csv}")


def main() -> None:
    here = os.path.dirname(os.path.abspath(__file__))
    equations = ["burgers_exact", "diffusion_exact", "heat3d_exact"]
    for eq in equations:
        eq_dir = os.path.join(here, eq)
        if not os.path.isdir(eq_dir):
            print(f"[skip] not a dir: {eq_dir}")
            continue
        _plot_for_equation_dir(eq_dir)


if __name__ == "__main__":
    main()


