"""
将每种方程的多次实验（通常 10 次）产生的 `training_curve.csv` 合并为一张表，
并在表头中标注方法与随机种子，且按 PINN / PINN-RAR / QPINN / QPINN-RAR 的顺序排列。

输入目录结构（默认）：
  checkpoints/<equation>_exact/<run_id>/training_curve.csv
  checkpoints/<equation>_exact/<run_id>/output.log   (用于解析 seed / solver / use_rar)

输出（默认写入到 out_dir 下，每个 equation 两份）：
  <out_dir>/<equation>/merged_loss.csv
  <out_dir>/<equation>/merged_rel_l2_error.csv

用法（在仓库根目录）：
  python scripts/merge_10runs_training_curves.py --checkpoints_dir checkpoints --out_dir checkpoints

说明：
- 合并方式为“按 iteration 对齐”的宽表（wide table）。
- 每张表只合并一个指标（loss 或 rel_l2_error）。
- 每个 run 的非 iteration 列会被重命名为：<method>__seed<seed>__<metric>
  例如：QPINN-RAR__seed672__rel_l2_error
"""

from __future__ import annotations

import argparse
import csv
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from extract_checkpoints_metrics import parse_output_log


_SEED_RE = re.compile(r"Random seed set to:\s*(\d+)|^\s*seed\s*:\s*(\d+)\s*$", re.IGNORECASE)
_SOLVER_RE = re.compile(r"^\s*solver\s*:\s*([A-Za-z0-9_]+)\s*$")
_USE_RAR_RE = re.compile(r"^\s*use_rar\s*:\s*(True|False)\s*$", re.IGNORECASE)
METHOD_ORDER: Tuple[str, ...] = ("PINN", "PINN-RAR", "QPINN", "QPINN-RAR")


def method_sort_key(method: str) -> Tuple[int, str]:
    try:
        return METHOD_ORDER.index(method), method
    except ValueError:
        return len(METHOD_ORDER), method


@dataclass(frozen=True)
class RunInfo:
    equation: str
    run_id: str
    seed: Optional[int]
    solver: Optional[str]
    use_rar: Optional[bool]
    method_label: str
    training_curve_csv: Optional[Path]
    output_log: Optional[Path]


def _infer_equation_and_run_id(training_curve_csv: Path, checkpoints_dir: Path) -> Tuple[str, str]:
    # checkpoints/<equation>/<run_id>/training_curve.csv
    try:
        rel = training_curve_csv.resolve().relative_to(checkpoints_dir.resolve())
        parts = rel.parts
        if len(parts) >= 3:
            return parts[0], parts[1]
    except Exception:
        pass
    return training_curve_csv.parent.parent.name, training_curve_csv.parent.name


def _parse_output_log_for_run_info(output_log: Path) -> Tuple[Optional[int], Optional[str], Optional[bool]]:
    seed: Optional[int] = None
    solver: Optional[str] = None
    use_rar: Optional[bool] = None

    with output_log.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            if seed is None:
                m = _SEED_RE.search(line)
                if m:
                    # 两个 group 任选其一
                    g1 = m.group(1)
                    g2 = m.group(2)
                    s = g1 or g2
                    if s is not None:
                        try:
                            seed = int(s)
                        except Exception:
                            seed = None

            if solver is None:
                m = _SOLVER_RE.match(line)
                if m:
                    solver = m.group(1).strip()

            if use_rar is None:
                m = _USE_RAR_RE.match(line)
                if m:
                    use_rar = m.group(1).strip().lower() == "true"

            if seed is not None and solver is not None and use_rar is not None:
                break

    return seed, solver, use_rar


def _method_label(solver: Optional[str], use_rar: Optional[bool]) -> str:
    if solver is None:
        return "UNKNOWN"
    if solver.upper() == "DV":
        if use_rar is True:
            return "QPINN-RAR"
        return "QPINN"
    # 仓库日志里 PINN 对应 Classical2（见 output.log）
    if solver.lower() in {"classical2", "classical"}:
        if use_rar is True:
            return "PINN-RAR"
        return "PINN"
    return solver


def iter_training_curve_csvs(checkpoints_dir: Path) -> Iterable[Path]:
    yield from checkpoints_dir.rglob("training_curve.csv")


def iter_training_runs(checkpoints_dir: Path) -> Iterable["RunInfo"]:
    run_dirs = {
        p.parent
        for p in checkpoints_dir.rglob("training_curve.csv")
    }
    run_dirs.update(
        p.parent
        for p in checkpoints_dir.rglob("output.log")
    )

    for run_dir in sorted(run_dirs):
        yield load_run_dir(run_dir, checkpoints_dir)


def _infer_equation_and_run_id_from_dir(run_dir: Path, checkpoints_dir: Path) -> Tuple[str, str]:
    try:
        rel = run_dir.resolve().relative_to(checkpoints_dir.resolve())
        parts = rel.parts
        if len(parts) >= 2:
            return parts[0], parts[1]
    except Exception:
        pass
    return run_dir.parent.name, run_dir.name


def load_run_dir(run_dir: Path, checkpoints_dir: Path) -> RunInfo:
    equation, run_id = _infer_equation_and_run_id_from_dir(run_dir, checkpoints_dir)
    output_log = run_dir / "output.log"
    training_curve_csv = run_dir / "training_curve.csv"

    if output_log.exists():
        seed, solver, use_rar = _parse_output_log_for_run_info(output_log)
    else:
        seed, solver, use_rar = None, None, None
        output_log = None

    return RunInfo(
        equation=equation,
        run_id=run_id,
        seed=seed,
        solver=solver,
        use_rar=use_rar,
        method_label=_method_label(solver, use_rar),
        training_curve_csv=training_curve_csv if training_curve_csv.exists() else None,
        output_log=output_log,
    )


def load_run(training_curve_csv: Path, checkpoints_dir: Path) -> RunInfo:
    equation, run_id = _infer_equation_and_run_id(training_curve_csv, checkpoints_dir)
    output_log = training_curve_csv.parent / "output.log"
    if output_log.exists():
        seed, solver, use_rar = _parse_output_log_for_run_info(output_log)
    else:
        seed, solver, use_rar = None, None, None
        output_log = None

    return RunInfo(
        equation=equation,
        run_id=run_id,
        seed=seed,
        solver=solver,
        use_rar=use_rar,
        method_label=_method_label(solver, use_rar),
        training_curve_csv=training_curve_csv,
        output_log=output_log,
    )


def read_training_curve(training_curve_csv: Path) -> Tuple[List[str], Dict[int, Dict[str, str]]]:
    # 返回 (metric_fields, iteration->row_metrics)
    by_it: Dict[int, Dict[str, str]] = {}
    with training_curve_csv.open("r", encoding="utf-8", errors="ignore", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError(f"CSV 缺少表头: {training_curve_csv}")
        fieldnames = [x.strip() for x in reader.fieldnames]
        if "iteration" not in fieldnames:
            raise ValueError(f"CSV 缺少 iteration 列: {training_curve_csv}")
        metric_fields = [c for c in fieldnames if c != "iteration"]

        for row in reader:
            it_raw = (row.get("iteration") or "").strip()
            if not it_raw:
                continue
            try:
                it = int(float(it_raw))
            except Exception:
                continue
            metrics: Dict[str, str] = {}
            for c in metric_fields:
                metrics[c] = (row.get(c) or "").strip()
            by_it[it] = metrics

    return metric_fields, by_it


def read_run_training_curve(run: RunInfo) -> Tuple[List[str], Dict[int, Dict[str, str]]]:
    if run.training_curve_csv is not None and run.training_curve_csv.exists():
        return read_training_curve(run.training_curve_csv)

    if run.output_log is None:
        return [], {}

    iter_rows, _summary = parse_output_log(run.output_log, run.output_log.parent.parent)
    by_it: Dict[int, Dict[str, str]] = {}
    for row in iter_rows:
        loss = row.loss_r if row.loss_r is not None else row.loss_total
        by_it[row.iteration] = {
            "loss": "" if loss is None else f"{loss:.12g}",
            "rel_l2_error": "" if row.rel_l2_error is None else f"{row.rel_l2_error:.12g}",
        }

    return ["loss", "rel_l2_error"], by_it


def merge_equation_runs(
    runs: Sequence[RunInfo],
    out_csv: Path,
    metric: str,
    method_order: Sequence[str] = METHOD_ORDER,
    *,
    sample_every: Optional[int] = None,
) -> None:
    # 读取所有 run，并按 iteration union 合并
    # data[method][seed][it][metric] = value
    data: Dict[str, Dict[int, Dict[int, Dict[str, str]]]] = {}
    all_iterations: set[int] = set()
    global_last_iteration: Optional[int] = None

    for r in runs:
        metric_fields, by_it = read_run_training_curve(r)
        if metric not in metric_fields:
            # 某些 run 可能没有该指标列，允许空白补齐
            pass

        all_iterations.update(by_it.keys())
        if by_it:
            run_last = max(by_it.keys())
            if global_last_iteration is None or run_last > global_last_iteration:
                global_last_iteration = run_last
        method = r.method_label
        seed = r.seed if r.seed is not None else -1
        data.setdefault(method, {}).setdefault(seed, {})
        data[method][seed].update(by_it)

    if not all_iterations:
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        with out_csv.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["iteration"])
            writer.writeheader()
        return

    # 迭代采样策略：
    # - 常规：每 sample_every 次输出一个点（例如 100）
    # - 额外：强制保留合并后“整体最大 iteration”对应的最后一行
    if sample_every is not None and sample_every > 0:
        sampled_iterations = {
            it
            for it in all_iterations
            if (
                # 仅保留 100 的倍数，但删除 10000
                ((it % sample_every == 0) and (it != 10000))
                # 无论如何保留最后一行（全局最大 iteration）
                or (global_last_iteration is not None and it == global_last_iteration)
                # 需求：保留 10001（不受“100 的倍数”限制）
                or (it == 10001)
            )
        }
        # 需求优先级：若“最后一行”恰好是 10000，则仍然保留最后一行
        if global_last_iteration == 10000:
            sampled_iterations.add(10000)
    else:
        sampled_iterations = set(all_iterations)

    # 生成表头：iteration + (method->seed->metric)
    header: List[str] = ["iteration"]

    def _sorted_methods() -> List[str]:
        ms = list(data.keys())
        # 先按指定顺序，再补其它
        ordered = [m for m in method_order if m in ms]
        rest = sorted([m for m in ms if m not in ordered], key=method_sort_key)
        return ordered + rest

    for method in _sorted_methods():
        seeds = sorted(data[method].keys())
        for seed in seeds:
            seed_str = "unknown" if seed == -1 else str(seed)
            header.append(f"{method}__seed{seed_str}__{metric}")

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=header)
        writer.writeheader()
        for it in sorted(sampled_iterations):
            out_row: Dict[str, str] = {"iteration": str(it)}
            for method in _sorted_methods():
                for seed, by_it in sorted(data[method].items(), key=lambda x: x[0]):
                    metrics = by_it.get(it) or {}
                    seed_str = "unknown" if seed == -1 else str(seed)
                    col = f"{method}__seed{seed_str}__{metric}"
                    out_row[col] = metrics.get(metric, "")
            # 需求：10000 处 rel_l2_error 没有数据时，不要生成“整行空”的相对误差记录。
            # 更通用地：合并 rel_l2_error 时，如果该 iteration 在所有 run 的该指标都为空，则跳过该行。
            if metric == "rel_l2_error":
                has_any_value = any((v.strip() != "") for k, v in out_row.items() if k != "iteration")
                if not has_any_value:
                    continue
            writer.writerow(out_row)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoints_dir", type=str, default="checkpoints", help="checkpoints 根目录")
    ap.add_argument("--out_dir", type=str, default="checkpoints", help="输出目录根（每个 equation 会创建子目录）")
    ap.add_argument(
        "--sample_every",
        type=int,
        default=100,
        help="合并后的 CSV 按 iteration 采样步长（默认 100）。<=0 表示不采样，输出全部 iteration。",
    )
    ap.add_argument(
        "--no_full",
        action="store_true",
        help="不输出 merged_*_full.csv（默认会同时输出采样版 merged_*.csv + 全量版 merged_*_full.csv）。",
    )
    args = ap.parse_args()

    checkpoints_dir = Path(args.checkpoints_dir)
    out_dir = Path(args.out_dir)
    if not checkpoints_dir.exists():
        raise FileNotFoundError(f"checkpoints_dir 不存在: {checkpoints_dir}")

    # 收集 runs：按 equation 分组
    by_equation: Dict[str, List[RunInfo]] = {}
    for run in iter_training_runs(checkpoints_dir):
        by_equation.setdefault(run.equation, []).append(run)

    if not by_equation:
        print(f"未找到任何 training_curve.csv，目录：{checkpoints_dir}")
        return 1

    targets = [
        ("loss", "merged_loss.csv"),
        ("rel_l2_error", "merged_rel_l2_error.csv"),
    ]

    for equation, runs in sorted(by_equation.items(), key=lambda x: x[0]):
        for metric, filename in targets:
            out_csv = out_dir / equation / filename
            try:
                merge_equation_runs(runs, out_csv=out_csv, metric=metric, sample_every=args.sample_every)
                print(f"[OK] {equation}: 合并 {len(runs)} 个 run ({metric}) -> {out_csv}")
            except PermissionError:
                print(f"[SKIP] 文件被占用，无法写入：{out_csv}")

            if not args.no_full:
                # 全量版本：不对 iteration 做 sample_every 采样
                full_name = filename.replace(".csv", "_full.csv")
                out_full_csv = out_dir / equation / full_name
                try:
                    merge_equation_runs(runs, out_csv=out_full_csv, metric=metric, sample_every=None)
                    print(f"[OK] {equation}: 合并 {len(runs)} 个 run ({metric}, full) -> {out_full_csv}")
                except PermissionError:
                    print(f"[SKIP] 文件被占用，无法写入：{out_full_csv}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
