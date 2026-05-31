"""
统计每个方程、每种方法、每次实验（run）的训练迭代次数。

优先从对应的 `output.log` 中读取类似：
  L-BFGS training completed after 1365 iterations
的行来获取迭代次数；
若没有找到该信息，则回退为使用 `training_curve.csv` 中的最大 iteration。

输入目录结构（默认）与 `merge_10runs_training_curves.py` 一致：
  checkpoints/<equation>_exact/<run_id>/training_curve.csv
  checkpoints/<equation>_exact/<run_id>/output.log   (用于解析 seed / solver / use_rar)

输出：
  <out_dir>/iterations_summary.csv                  # 每个 run 的迭代次数
  <out_dir>/iterations_summary_stats.csv            # 每个方程、每种方法的迭代次数均值

表头示例：
  equation,run_id,method,seed,max_iteration

用法（在仓库根目录）：
  python scripts/collect_iterations_10runs.py --checkpoints_dir checkpoints --out_dir checkpoints
"""

from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path
from typing import Dict, List, Optional

# 复用已有脚本中的解析逻辑
from merge_10runs_training_curves import (  # type: ignore
    RunInfo,
    iter_training_runs,
    method_sort_key,
    read_run_training_curve,
)


_ITER_RE = re.compile(
    r"training completed after\s+(\d+)\s+iterations", re.IGNORECASE
)


def _parse_iterations_from_log(output_log: Path) -> Optional[int]:
    """
    从 output.log 中解析最终的迭代次数。
    例如：L-BFGS training completed after 1365 iterations
    """
    if not output_log.exists():
        return None

    last_match: Optional[int] = None
    with output_log.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            m = _ITER_RE.search(line)
            if m:
                try:
                    last_match = int(m.group(1))
                except Exception:
                    continue
    return last_match


def compute_max_iteration_per_run(
    checkpoints_dir: Path,
) -> List[Dict[str, str]]:
    """
    遍历所有 run，统计每个 run 的迭代次数。
    优先使用 output.log 中的 “training completed after X iterations”，
    若不存在则回退为 training_curve.csv 或 output.log 曲线记录的最大 iteration。
    返回的列表中每个元素为一行记录（字典），用于写入 CSV。
    """
    rows: List[Dict[str, str]] = []

    for run in iter_training_runs(checkpoints_dir):
        # 1) 优先从 output.log 解析
        it: Optional[int] = None
        if run.output_log is not None:
            it = _parse_iterations_from_log(run.output_log)

        # 2) 若日志中没有，则从曲线记录的最大 iteration 获取
        if it is None:
            _metric_fields, by_it = read_run_training_curve(run)
            if not by_it:
                # 没有任何 iteration 记录，跳过
                continue
            it = max(by_it.keys())

        row: Dict[str, str] = {
            "equation": run.equation,
            "run_id": run.run_id,
            "method": run.method_label,
            "seed": "" if run.seed is None else str(run.seed),
            "max_iteration": str(it),
        }
        rows.append(row)

    return rows


def compute_mean_iterations_per_equation_method(
    rows: List[Dict[str, str]],
) -> List[Dict[str, str]]:
    """
    在每个 run 的迭代次数基础上，按 (equation, method) 聚合，计算平均迭代次数和标准差。
    """
    agg: Dict[tuple, List[float]] = {}

    for r in rows:
        key = (r["equation"], r["method"])
        it = float(r["max_iteration"])
        if key not in agg:
            agg[key] = []
        agg[key].append(it)

    stats_rows: List[Dict[str, str]] = []
    for (equation, method), values in agg.items():
        n = len(values)
        if n == 0:
            mean_it = 0.0
            std_it = 0.0
        else:
            mean_it = sum(values) / n
            # 无偏/有偏在这里差别不大，直接用总体标准差
            var = sum((v - mean_it) ** 2 for v in values) / n
            std_it = var ** 0.5
        stats_rows.append(
            {
                "equation": equation,
                "method": method,
                "num_runs": str(n),
                "mean_iteration": f"{mean_it:.6f}",
                "std_iteration": f"{std_it:.6f}",
            }
        )

    # 为了可读性排序
    stats_rows.sort(key=lambda r: (r["equation"], method_sort_key(r["method"])))
    return stats_rows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--checkpoints_dir",
        type=str,
        default="checkpoints",
        help="checkpoints 根目录",
    )
    ap.add_argument(
        "--out_dir",
        type=str,
        default="checkpoints",
        help="输出目录根（结果写为 iterations_summary.csv）",
    )
    args = ap.parse_args()

    checkpoints_dir = Path(args.checkpoints_dir)
    out_dir = Path(args.out_dir)

    if not checkpoints_dir.exists():
        raise FileNotFoundError(f"checkpoints_dir 不存在: {checkpoints_dir}")

    rows = compute_max_iteration_per_run(checkpoints_dir)
    if not rows:
        print(f"未找到任何 training_curve.csv，目录：{checkpoints_dir}")
        return 1

    # 为了易于筛选，按 equation、method、seed 排序
    rows.sort(
        key=lambda r: (
            r["equation"],
            method_sort_key(r["method"]),
            r["seed"],
            r["run_id"],
        )
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    # 1) 明细：每个 run 的迭代次数
    out_csv = out_dir / "iterations_summary.csv"
    fieldnames = ["equation", "run_id", "method", "seed", "max_iteration"]
    with out_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    print(f"[OK] 共统计 {len(rows)} 个 run 的最大迭代次数 -> {out_csv}")

    # 2) 统计：每个方程、每种方法的平均迭代次数
    stats_rows = compute_mean_iterations_per_equation_method(rows)
    stats_csv = out_dir / "iterations_summary_stats.csv"
    stats_fieldnames = [
        "equation",
        "method",
        "num_runs",
        "mean_iteration",
        "std_iteration",
    ]
    with stats_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=stats_fieldnames)
        writer.writeheader()
        for row in stats_rows:
            writer.writerow(row)

    print(
        f"[OK] 共统计 {len(stats_rows)} 组 (equation, method) 的平均迭代次数 -> {stats_csv}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


