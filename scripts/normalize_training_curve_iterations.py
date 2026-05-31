"""
将 checkpoints 下各个 run 的 training_curve.csv 的 iteration 归一化到 [0, 1]。

需求：
- 对每个 training_curve.csv 单独做归一化：iteration_normalized = iteration / max_iteration
  （max_iteration 为该文件中可解析到的最大 iteration；因此最大 iteration 对应 1.0）
- 保留原 training_curve.csv 不改动
- 在同目录生成一个新的 CSV 文件（默认：training_curve_iteration_normalized.csv）

用法（在仓库根目录）：
  python scripts/normalize_training_curve_iterations.py --checkpoints_dir checkpoints

仅处理 heat3d_exact：
  python scripts/normalize_training_curve_iterations.py --checkpoints_dir checkpoints --equation heat3d_exact

处理 checkpoints 下所有方程目录（默认行为）：
  python scripts/normalize_training_curve_iterations.py --checkpoints_dir checkpoints
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

from merge_10runs_training_curves import iter_training_runs, read_run_training_curve


@dataclass(frozen=True)
class NormalizeResult:
    src: Path
    dst: Path
    max_iteration: int
    rows_written: int


def iter_training_curve_csvs(checkpoints_dir: Path, *, equation: Optional[str] = None) -> Iterable[Path]:
    if equation:
        yield from (checkpoints_dir / equation).rglob("training_curve.csv")
    else:
        yield from checkpoints_dir.rglob("training_curve.csv")


def _parse_iteration(x: str) -> Optional[int]:
    s = (x or "").strip()
    if not s:
        return None
    try:
        # 兼容 "10001" / "10001.0" 之类
        return int(float(s))
    except Exception:
        return None


def _compute_max_iteration(rows: List[Dict[str, str]], it_col: str) -> int:
    max_it: Optional[int] = None
    for r in rows:
        it = _parse_iteration(r.get(it_col, ""))
        if it is None:
            continue
        if max_it is None or it > max_it:
            max_it = it
    if max_it is None or max_it <= 0:
        raise ValueError("无法从 CSV 中解析到有效的 iteration（或 max_iteration <= 0）")
    return max_it


def normalize_one_csv(src_csv: Path, *, out_name: str) -> NormalizeResult:
    with src_csv.open("r", encoding="utf-8", errors="ignore", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError(f"CSV 缺少表头: {src_csv}")
        fieldnames = [c.strip() for c in reader.fieldnames]

        # 尽量兼容其它命名
        it_col = "iteration" if "iteration" in fieldnames else None
        if it_col is None:
            for cand in ("iter", "iters", "step", "steps"):
                if cand in fieldnames:
                    it_col = cand
                    break
        if it_col is None:
            raise ValueError(f"CSV 缺少 iteration 列: {src_csv} (fieldnames={fieldnames})")

        rows: List[Dict[str, str]] = []
        for row in reader:
            # DictReader 可能会返回 None key（当行列数不齐），这里直接忽略
            rows.append({(k or "").strip(): (v or "").strip() for k, v in row.items() if k is not None})

    max_it = _compute_max_iteration(rows, it_col)
    dst_csv = src_csv.parent / out_name

    # 输出列：iteration_normalized + 原始列
    out_fieldnames = ["iteration_normalized"] + fieldnames

    rows_written = 0
    with dst_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=out_fieldnames, extrasaction="ignore")
        writer.writeheader()
        for r in rows:
            it = _parse_iteration(r.get(it_col, ""))
            if it is None:
                it_norm = ""
            else:
                it_norm = f"{it / max_it:.10f}".rstrip("0").rstrip(".")
            out_row = {"iteration_normalized": it_norm}
            out_row.update(r)
            writer.writerow(out_row)
            rows_written += 1

    return NormalizeResult(src=src_csv, dst=dst_csv, max_iteration=max_it, rows_written=rows_written)


def normalize_one_run(run, *, out_name: str) -> NormalizeResult:
    src = run.training_curve_csv if run.training_curve_csv is not None else run.output_log
    if src is None:
        raise ValueError("run 缺少 training_curve.csv 和 output.log")

    metric_fields, by_it = read_run_training_curve(run)
    if not by_it:
        raise ValueError("无法解析到曲线记录")

    fieldnames = ["iteration"] + metric_fields
    rows: List[Dict[str, str]] = []
    for it in sorted(by_it.keys()):
        row = {"iteration": str(it)}
        row.update(by_it[it])
        rows.append(row)

    max_it = _compute_max_iteration(rows, "iteration")
    dst_csv = src.parent / out_name
    out_fieldnames = ["iteration_normalized"] + fieldnames

    rows_written = 0
    with dst_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=out_fieldnames, extrasaction="ignore")
        writer.writeheader()
        for r in rows:
            it = _parse_iteration(r.get("iteration", ""))
            it_norm = "" if it is None else f"{it / max_it:.10f}".rstrip("0").rstrip(".")
            out_row = {"iteration_normalized": it_norm}
            out_row.update(r)
            writer.writerow(out_row)
            rows_written += 1

    return NormalizeResult(src=src, dst=dst_csv, max_iteration=max_it, rows_written=rows_written)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoints_dir", type=str, default="checkpoints", help="checkpoints 根目录")
    ap.add_argument(
        "--equation",
        type=str,
        default=None,
        help="只处理指定方程目录（例如 heat3d_exact / burgers_exact / diffusion_exact）；不传则处理全部",
    )
    ap.add_argument(
        "--out_name",
        type=str,
        default="training_curve_iteration_normalized.csv",
        help="输出文件名（写入每个 run 目录中）",
    )
    args = ap.parse_args()

    checkpoints_dir = Path(args.checkpoints_dir)
    if not checkpoints_dir.exists():
        raise FileNotFoundError(f"checkpoints_dir 不存在: {checkpoints_dir}")

    runs = [
        run
        for run in iter_training_runs(checkpoints_dir)
        if args.equation is None or run.equation == args.equation
    ]
    if not runs:
        where = checkpoints_dir / args.equation if args.equation else checkpoints_dir
        print(f"未找到任何 run：{where}")
        return 1

    ok = 0
    failed = 0
    for run in sorted(runs, key=lambda x: (x.equation, x.run_id)):
        try:
            res = normalize_one_run(run, out_name=args.out_name)
            ok += 1
            print(f"[OK] {res.src} -> {res.dst} (max_iteration={res.max_iteration}, rows={res.rows_written})")
        except Exception as e:
            failed += 1
            print(f"[FAIL] {run.equation}/{run.run_id}: {e}")

    if failed:
        print(f"完成：成功 {ok} 个，失败 {failed} 个。")
        return 2
    print(f"完成：成功 {ok} 个。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

