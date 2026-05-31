"""
从 checkpoints/**/output.log 中提取训练曲线指标。

- 默认只为缺少 `training_curve.csv` 的 run 生成曲线文件（仅包含 iteration / loss / rel_l2_error）。
- 如需覆盖已有文件，使用 `--overwrite`。
- 不再在仓库根目录下生成/更新聚合的 `checkpoints_iterations.csv` 和 `checkpoints_runs_summary.csv`，
  以避免频繁改动这两个大文件。

用法（在仓库根目录）：
  python scripts/extract_checkpoints_metrics.py --checkpoints_dir checkpoints
"""

from __future__ import annotations

import argparse
import csv
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional, Tuple


# 支持普通浮点 + 科学计数法，同时兼容训练过程中可能出现的 nan/inf（例如评估异常时）
# 说明：float("nan") / float("inf") 在 Python 中可正常解析，但需要 regex 先匹配到字符串。
_FLOAT_RE = r"(?:[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?|[-+]?(?:nan|inf))"

# 兼容如下行：
# Iteration: 100, loss_r = 4.9e-03 ,  loss_bc = 1.1e-02,  loss_ic = 6.9e-02,  ... rel_l2_error = 7.5e-01
_ITERATION_RE = re.compile(
    # 兼容两种前缀：
    # 1) "Iteration: 100, loss_r = ..."
    # 2) "L-BFGS Iteration: 5, Loss: ..., Relative L2 Error: ..."
    rf"^\s*(?P<lbfgs>L-BFGS\s+)?Iteration:\s*(?P<it>\d+)\s*,\s*(?P<rest>.*)$"
)
# key 里可能包含数字（例如 "rel_l2_error"），因此需要允许 0-9
_KV_EQ_RE = re.compile(rf"(?P<key>[A-Za-z0-9_]+)\s*=\s*(?P<val>{_FLOAT_RE})")
_KV_COLON_RE = re.compile(
    # 兼容 "Loss: 1.23e-4, Relative L2 Error: 5.67e-3" 这种格式
    rf"(?P<key>[A-Za-z0-9_ ]+):\s*(?P<val>{_FLOAT_RE})"
)

# 兼容日志尾部汇总行：
# - "L-BFGS training completed after 6484 iterations"
_LBFGS_COMPLETED_RE = re.compile(r"^\s*L-BFGS training completed after\s+(?P<it>\d+)\s+iterations\s*$")
# - "Relative L2 error (eval data): 3.70e-02"
_FINAL_EVAL_REL_RE = re.compile(rf"^\s*Relative L2 error\s*\(eval data\)\s*:\s*(?P<val>{_FLOAT_RE})\s*$")
# - "RelL2_U%  : 3.530e-02"
_FINAL_EVAL_REL_PERCENT_RE = re.compile(rf"^\s*RelL2_U%\s*:\s*(?P<val>{_FLOAT_RE})\s*$")


@dataclass(frozen=True)
class IterRow:
    problem: str
    run_id: str
    output_log: str
    iteration: int
    loss_total: Optional[float]
    loss_r: Optional[float]
    loss_bc: Optional[float]
    loss_ic: Optional[float]
    rel_l2_error: Optional[float]
    lr: Optional[float]
    time_taken: Optional[float]


@dataclass(frozen=True)
class RunSummaryRow:
    problem: str
    run_id: str
    output_log: str
    final_iteration: Optional[int]
    final_rel_l2_error: Optional[float]
    best_rel_l2_error: Optional[float]
    best_rel_l2_iteration: Optional[int]
    # 随机种子（如果日志中能解析到）
    seed: Optional[int] = None


def _safe_float(x: Optional[str]) -> Optional[float]:
    if x is None:
        return None
    try:
        return float(x)
    except Exception:
        return None


def _parse_iteration_line(line: str) -> Optional[Tuple[int, Dict[str, float]]]:
    m = _ITERATION_RE.match(line)
    if not m:
        return None
    it = int(m.group("it"))
    is_lbfgs = m.group("lbfgs") is not None
    # 按需求：L-BFGS 阶段的迭代序号从 10000 开始计数
    if is_lbfgs:
        it = 10000 + it
    rest = m.group("rest")
    kv: Dict[str, float] = {}
    # 先解析 key = val 形式（Adam 阶段）
    for m2 in _KV_EQ_RE.finditer(rest):
        key = m2.group("key").strip()
        val = _safe_float(m2.group("val"))
        if val is None:
            continue
        kv[key] = val

    # 再解析 "Key: val" 形式（L-BFGS 阶段）
    for m2 in _KV_COLON_RE.finditer(rest):
        raw_key = m2.group("key").strip()
        # 例如 "Relative L2 Error" -> "relative_l2_error"
        key = raw_key.replace(" ", "_").lower()
        val = _safe_float(m2.group("val"))
        if val is None:
            continue
        # 避免覆盖前面的同名 key = val
        kv.setdefault(key, val)
    return it, kv


def _infer_problem_and_run_id(output_log: Path, checkpoints_dir: Path) -> Tuple[str, str]:
    # checkpoints/<problem>/<run_id>/output.log
    try:
        rel = output_log.resolve().relative_to(checkpoints_dir.resolve())
        parts = rel.parts
        if len(parts) >= 3:
            return parts[0], parts[1]
    except Exception:
        pass
    # 兜底：用父目录名
    return output_log.parent.parent.name, output_log.parent.name


def iter_output_logs(checkpoints_dir: Path) -> Iterator[Path]:
    yield from checkpoints_dir.rglob("output.log")


def parse_output_log(output_log: Path, checkpoints_dir: Path) -> Tuple[List[IterRow], RunSummaryRow]:
    problem, run_id = _infer_problem_and_run_id(output_log, checkpoints_dir)

    rows: List[IterRow] = []
    final_it: Optional[int] = None
    final_rel: Optional[float] = None
    best_rel: Optional[float] = None
    best_rel_it: Optional[int] = None
    seed_val: Optional[int] = None
    completed_it: Optional[int] = None
    eval_rel: Optional[float] = None

    # 某些日志可能包含非 UTF-8 字符；errors=ignore 保证尽量解析
    with output_log.open("r", encoding="utf-8", errors="ignore") as f:
        for raw_line in f:
            line = raw_line.rstrip("\n")

            # 解析训练完成的总迭代次数（原始计数，不做 10000 偏移）
            m_completed = _LBFGS_COMPLETED_RE.match(line)
            if m_completed:
                try:
                    completed_it = int(m_completed.group("it"))
                except Exception:
                    pass

            # 解析最终 eval 数据上的相对 L2 误差
            m_eval = _FINAL_EVAL_REL_RE.match(line)
            if m_eval:
                eval_rel = _safe_float(m_eval.group("val"))
            m_eval_percent = _FINAL_EVAL_REL_PERCENT_RE.match(line)
            if m_eval_percent:
                raw_eval_rel_percent = _safe_float(m_eval_percent.group("val"))
                if raw_eval_rel_percent is not None:
                    eval_rel = raw_eval_rel_percent / 100.0

            # 先尝试解析随机种子（只要解析到一次就够了）
            if seed_val is None:
                # 两种常见形式：
                # 1) "Random seed set to: 3"
                # 2) "seed : 3"
                if "Random seed set to:" in line:
                    try:
                        seed_str = line.split("Random seed set to:")[-1].strip()
                        seed_val = int(seed_str)
                    except Exception:
                        pass
                elif "seed" in line and ":" in line:
                    # 例如 "seed : 3"
                    try:
                        # 只在行中第一个冒号后取值，兼容 "seed: 3" / "seed : 3"
                        _, after = line.split(":", 1)
                        seed_str = after.strip()
                        # seed 可能后面跟其它内容，这里只取第一个空格前
                        seed_str = seed_str.split()[0]
                        seed_val = int(seed_str)
                    except Exception:
                        pass

            parsed = _parse_iteration_line(line)
            if parsed is None:
                continue
            it, kv = parsed

            # Adam 阶段：日志里是 loss_r = ...
            # SciPy L-BFGS 阶段：日志里是 "Loss: ..."，这里也统一归到 loss_r
            loss_r = kv.get("loss_r") or kv.get("loss")
            loss_bc = kv.get("loss_bc")
            loss_ic = kv.get("loss_ic")
            loss_total = kv.get("loss")
            if loss_total is None:
                # 如果没有显式 loss，则用现有项相加
                if loss_r is not None or loss_bc is not None or loss_ic is not None:
                    loss_total = (loss_r or 0.0) + (loss_bc or 0.0) + (loss_ic or 0.0)

            # 相对 L2 误差：
            # - Adam 阶段 key: "rel_l2_error"
            # - L-BFGS 阶段 key: "relative_l2_error"（由上面的 colon 解析统一为小写）
            rel = kv.get("rel_l2_error") or kv.get("relative_l2_error") or kv.get("Relative")  # 兜底（防止不同 key）
            lr = kv.get("lr")
            time_taken = kv.get("time_taken")

            rows.append(
                IterRow(
                    problem=problem,
                    run_id=run_id,
                    output_log=str(output_log),
                    iteration=it,
                    loss_total=loss_total,
                    loss_r=loss_r,
                    loss_bc=loss_bc,
                    loss_ic=loss_ic,
                    rel_l2_error=rel,
                    lr=lr,
                    time_taken=time_taken,
                )
            )

            final_it = it
            final_rel = rel
            if rel is not None and (best_rel is None or rel < best_rel):
                best_rel = rel
                best_rel_it = it

    # 如果日志尾部提供了“训练完成迭代次数”和“最终 eval rel_l2”，优先使用它们作为 summary 的 final_*。
    if completed_it is not None:
        # 注意：L-BFGS 阶段的迭代在上游统一加了 10000 偏移，
        # 因此这里也需要在原始 completed_it 的基础上加 10000，保证与曲线中的迭代编号一致。
        final_it = 10000 + completed_it
    if eval_rel is not None:
        final_rel = eval_rel

    summary = RunSummaryRow(
        problem=problem,
        run_id=run_id,
        output_log=str(output_log),
        final_iteration=final_it,
        final_rel_l2_error=final_rel,
        best_rel_l2_error=best_rel,
        best_rel_l2_iteration=best_rel_it,
        seed=seed_val,
    )
    return rows, summary


def write_iterations_csv(rows: Iterable[IterRow], out_csv: Path) -> None:
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "problem",
        "run_id",
        "output_log",
        "iteration",
        "loss_total",
        "loss_r",
        "loss_bc",
        "loss_ic",
        "rel_l2_error",
        "lr",
        "time_taken",
    ]
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(
                {
                    "problem": r.problem,
                    "run_id": r.run_id,
                    "output_log": r.output_log,
                    "iteration": r.iteration,
                    "loss_total": r.loss_total,
                    "loss_r": r.loss_r,
                    "loss_bc": r.loss_bc,
                    "loss_ic": r.loss_ic,
                    "rel_l2_error": r.rel_l2_error,
                    "lr": r.lr,
                    "time_taken": r.time_taken,
                }
            )


def write_summary_csv(rows: Iterable[RunSummaryRow], out_csv: Path) -> None:
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "problem",
        "run_id",
        "output_log",
        "final_iteration",
        "final_rel_l2_error",
        "best_rel_l2_error",
        "best_rel_l2_iteration",
        "seed",
    ]
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(
                {
                    "problem": r.problem,
                    "run_id": r.run_id,
                    "output_log": r.output_log,
                    "final_iteration": r.final_iteration,
                    "final_rel_l2_error": r.final_rel_l2_error,
                    "best_rel_l2_error": r.best_rel_l2_error,
                    "best_rel_l2_iteration": r.best_rel_l2_iteration,
                    "seed": r.seed,
                }
            )


def write_per_log_curve_csv(
    rows: Iterable[IterRow],
    summary: RunSummaryRow,
    out_csv: Path,
) -> None:
    """
    为单个 output.log 写一张仅包含 iteration / loss / rel_l2_error 的表。

    - 按需求约定：loss 列使用 loss_r（即 PDE 残差项）
      * Adam 阶段：来自 "loss_r = ..."
      * L-BFGS 阶段：来自 "Loss: ..."，在上游也归入 loss_r
    """
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    # 列：iteration, loss, rel_l2_error
    fieldnames = [
        "iteration",
        "loss",
        "rel_l2_error",
    ]
    # 先把输出行缓存在内存里，便于“最后一行覆盖”而不是重复追加。
    out_rows: List[Dict[str, Optional[float]]] = []
    final_it = summary.final_iteration
    final_rel = summary.final_rel_l2_error

    # 过滤规则（按你的需求）：
    # - 只保留 iteration 为 100 的倍数的行
    # - 额外保留 iteration == 10001
    # - 删除 iteration == 10000
    # - 始终保留最后一行（final_it / final_rel），即使不是 100 的倍数
    kept_last_it: Optional[int] = None

    # 为了给“最终评估点”补齐 loss：建立 iteration -> loss 的映射，同时保留最后一个可用 loss。
    loss_by_it: Dict[int, float] = {}
    last_loss: Optional[float] = None
    for r in rows:
        it = r.iteration
        loss_val = r.loss_r if r.loss_r is not None else r.loss_total
        if loss_val is not None:
            loss_by_it[it] = loss_val
            last_loss = loss_val

        if it == 10000:
            continue
        if (it % 100 != 0) and (it != 10001):
            continue
        kept_last_it = it
        out_rows.append(
            {
                "iteration": float(it),
                "loss": loss_val,
                "rel_l2_error": r.rel_l2_error,
            }
        )

    # 追加/覆盖最终评估点：同时写入 loss（优先用 final_it 对应的 loss，否则用 last_loss 兜底）。
    if (final_it is not None or final_rel is not None) and final_it != 10000:
        final_loss: Optional[float] = None
        if final_it is not None:
            final_loss = loss_by_it.get(final_it, last_loss)
        else:
            final_loss = last_loss

        final_row = {
            "iteration": float(final_it) if final_it is not None else None,
            "loss": final_loss,
            "rel_l2_error": final_rel,
        }

        if out_rows and final_it is not None and kept_last_it == final_it:
            # 同一 iteration：用 summary 的 final_rel/最终 loss 覆盖最后一行，避免重复点。
            out_rows[-1] = final_row
        else:
            out_rows.append(final_row)

    with out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in out_rows:
            w.writerow(r)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--checkpoints_dir",
        type=str,
        default="checkpoints",
        help="checkpoints 目录（默认 checkpoints）",
    )
    ap.add_argument(
        "--overwrite",
        action="store_true",
        help="覆盖已经存在的 training_curve.csv（默认只补缺失文件）。",
    )
    args = ap.parse_args()

    checkpoints_dir = Path(args.checkpoints_dir)
    created = 0
    skipped = 0
    failed = 0

    for output_log in iter_output_logs(checkpoints_dir):
        iter_rows, summary = parse_output_log(output_log, checkpoints_dir)
        if iter_rows:
            # 为当前 output.log 写一张只包含 iteration / loss / rel_l2_error 的表
            per_log_csv = output_log.parent / "training_curve.csv"
            if per_log_csv.exists() and not args.overwrite:
                skipped += 1
                continue
            try:
                write_per_log_curve_csv(iter_rows, summary, per_log_csv)
                created += 1
            except PermissionError:
                failed += 1

    print(
        f"training_curve.csv: created/updated={created}, skipped_existing={skipped}, failed_permission={failed}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
