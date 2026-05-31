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
    "DV": "QPINN",
    "PINN": "PINN",
    "pinn-rar-s": "PINN-RAR",
}

METHOD_COLOR = {
    "DV": "#1f77b4",
    "PINN": "#ff7f0e",
    "pinn-rar-s": "#2ca02c",
}


def _group_columns_by_method(columns: list[str]) -> dict[str, list[str]]:
    """
    Expected column format:
      {method}__seed{N}__loss
    e.g. DV__seed3__loss
    """
    pat = re.compile(r"^(?P<method>.+?)__seed\d+__loss$")
    grouped: dict[str, list[str]] = {}
    for c in columns:
        m = pat.match(c)
        if not m:
            continue
        method = m.group("method")
        grouped.setdefault(method, []).append(c)
    return grouped


def main() -> None:
    _setup_matplotlib_cn_font()
    here = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(here, "merged_loss.csv")
    if not os.path.exists(csv_path):
        raise FileNotFoundError(csv_path)

    df = pd.read_csv(csv_path)
    if "iteration" not in df.columns:
        raise ValueError("CSV 缺少 'iteration' 列")

    grouped = _group_columns_by_method([c for c in df.columns if c != "iteration"])
    if not grouped:
        raise ValueError("没有找到形如 {method}__seed{N}__loss 的列")

    
    target_methods = ["DV", "PINN", "pinn-rar-s"]
    existing_methods = [m for m in target_methods if m in grouped]
    if not existing_methods:
        raise ValueError(f"在 CSV 中未找到目标方法列：{target_methods}")

    x = pd.to_numeric(df["iteration"], errors="coerce").to_numpy()

    fig, ax = plt.subplots(figsize=(10, 5.5), dpi=160)
    for method in existing_methods:
        cols = grouped[method]
        y = df[cols].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)

        # 安全归约（避免全 NaN 行触发 RuntimeWarning）
        finite = np.isfinite(y)
        has_any = finite.any(axis=1)

        y_min = np.min(np.where(finite, y, np.inf), axis=1)
        y_min = np.where(has_any, y_min, np.nan)

        y_max = np.max(np.where(finite, y, -np.inf), axis=1)
        y_max = np.where(has_any, y_max, np.nan)

        y_sum = np.sum(np.where(finite, y, 0.0), axis=1)
        y_cnt = finite.sum(axis=1)
        y_mean = np.divide(y_sum, y_cnt, out=np.full_like(y_sum, np.nan, dtype=float), where=y_cnt > 0)

        color = METHOD_COLOR.get(method, None)
        label = METHOD_DISPLAY_NAME.get(method, method)

        ax.fill_between(x, y_min, y_max, color=color, alpha=0.18, linewidth=0)
        ax.plot(x, y_mean, color=color, linewidth=2.0, label=f"{label} (mean)")

    ax.set_title("Loss range (min–max shaded)")
    ax.set_xlabel("iteration")
    ax.set_ylabel("loss")
    ax.set_yscale("log")
    ax.grid(True, which="both", linestyle="--", linewidth=0.6, alpha=0.35)
    ax.legend()

    out_path = os.path.join(here, "loss_range_QPINN_PINN_PINN-RAR.png")
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()


