from __future__ import annotations

import argparse
import csv
import math
import re
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


METHOD_ORDER = ["PINN", "PINN-RAR", "QPINN", "QPINN-RAR"]
METRICS = ["iteration", "loss", "rel_l2_error"]
DETAIL_COLUMNS = ["run_seeds", "run_ids"] + [f"{metric}_values" for metric in METRICS]


@dataclass(frozen=True)
class RunFinalMetrics:
    equation: str
    method: str
    seed: int | None
    run_id: str
    iteration: float
    loss: float
    rel_l2_error: float


def _read_text_head(path: Path, max_bytes: int = 20_000) -> str:
    with path.open("rb") as f:
        return f.read(max_bytes).decode("utf-8", errors="ignore")


def _parse_seed(text: str) -> int | None:
    m = re.search(r"Random seed set to:\s*(\d+)", text)
    if m:
        return int(m.group(1))
    m = re.search(r"^\s*seed\s*:\s*(\d+)\s*$", text, flags=re.MULTILINE)
    if m:
        return int(m.group(1))
    return None


def _parse_method(text: str) -> str | None:
    use_rar = bool(re.search(r"^\s*use_rar\s*:\s*True\s*$", text, flags=re.MULTILINE))

    if re.search(r"\bUsing\s+DV\s+Solver\b", text) or re.search(
        r"^\s*solver\s*:\s*DV\s*$", text, flags=re.MULTILINE
    ):
        return "QPINN-RAR" if use_rar else "QPINN"

    if re.search(r"\bUsing\s+Classical\s+Solver2\b", text) or re.search(
        r"^\s*solver\s*:\s*Classical2\s*$", text, flags=re.MULTILINE
    ):
        return "PINN-RAR" if use_rar else "PINN"

    return None


def _to_float(value: str | None) -> float:
    if value is None:
        return math.nan
    try:
        return float(value)
    except ValueError:
        return math.nan


def _read_last_metrics(csv_path: Path) -> tuple[float, float, float] | None:
    last: dict[str, str] | None = None
    with csv_path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            values = [_to_float(row.get(metric)) for metric in METRICS]
            if all(math.isfinite(v) for v in values):
                last = row

    if last is None:
        return None
    return tuple(_to_float(last.get(metric)) for metric in METRICS)  # type: ignore[return-value]


def iter_run_finals(checkpoints_dir: Path, csv_name: str) -> Iterable[RunFinalMetrics]:
    for equation_dir in sorted(p for p in checkpoints_dir.iterdir() if p.is_dir()):
        for run_dir in sorted(p for p in equation_dir.iterdir() if p.is_dir()):
            log_path = run_dir / "output.log"
            csv_path = run_dir / csv_name
            if not log_path.exists() or not csv_path.exists():
                continue

            head = _read_text_head(log_path)
            method = _parse_method(head)
            if method is None:
                continue

            metrics = _read_last_metrics(csv_path)
            if metrics is None:
                continue

            yield RunFinalMetrics(
                equation=equation_dir.name,
                method=method,
                seed=_parse_seed(head),
                run_id=run_dir.name,
                iteration=metrics[0],
                loss=metrics[1],
                rel_l2_error=metrics[2],
            )


def _pick_runs(runs: list[RunFinalMetrics], max_runs: int) -> list[RunFinalMetrics]:
    def key(run: RunFinalMetrics) -> tuple[int, str]:
        seed = run.seed if run.seed is not None else 10**18
        return (seed, run.run_id)

    return sorted(runs, key=key)[:max_runs]


def _mean(values: list[float]) -> float:
    return statistics.fmean(values) if values else math.nan


def _std(values: list[float]) -> float:
    return statistics.stdev(values) if len(values) > 1 else math.nan


def _fmt(value: float) -> str:
    if not math.isfinite(value):
        return ""
    return f"{value:.10g}"


def _join_float_values(values: list[float]) -> str:
    return ";".join(_fmt(value) for value in values)


def _write_csv(rows: list[dict[str, object]], out_path: Path) -> None:
    fieldnames = [
        "equation",
        "method",
        "n",
        "iteration_mean",
        "iteration_std",
        "loss_mean",
        "loss_std",
        "rel_l2_error_mean",
        "rel_l2_error_std",
    ] + DETAIL_COLUMNS
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_markdown(rows: list[dict[str, object]], out_path: Path) -> None:
    columns = [
        "equation",
        "method",
        "n",
        "iteration_mean",
        "iteration_std",
        "loss_mean",
        "loss_std",
        "rel_l2_error_mean",
        "rel_l2_error_std",
    ] + DETAIL_COLUMNS
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |",
    ]
    for row in rows:
        values = []
        for col in columns:
            value = row[col]
            values.append(_fmt(value) if isinstance(value, float) else str(value))
        lines.append("| " + " | ".join(values) + " |")
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_best_rel_l2_markdown(rows: list[dict[str, object]], out_path: Path) -> None:
    by_equation: dict[str, list[dict[str, object]]] = {}
    for row in rows:
        by_equation.setdefault(str(row["equation"]), []).append(row)

    lines = [
        "| equation | best_method | rel_l2_error_mean | rel_l2_error_std |",
        "| --- | --- | --- | --- |",
    ]
    for equation in sorted(by_equation):
        best = min(by_equation[equation], key=lambda r: float(r["rel_l2_error_mean"]))
        lines.append(
            "| "
            + " | ".join(
                [
                    equation,
                    str(best["method"]),
                    _fmt(float(best["rel_l2_error_mean"])),
                    _fmt(float(best["rel_l2_error_std"])),
                ]
            )
            + " |"
        )
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_summary(checkpoints_dir: Path, csv_name: str, max_runs: int) -> list[dict[str, object]]:
    groups: dict[tuple[str, str], list[RunFinalMetrics]] = {}
    for run in iter_run_finals(checkpoints_dir, csv_name):
        groups.setdefault((run.equation, run.method), []).append(run)

    equations = sorted({equation for equation, _ in groups})
    rows: list[dict[str, object]] = []
    for equation in equations:
        for method in METHOD_ORDER:
            selected = _pick_runs(groups.get((equation, method), []), max_runs)
            if not selected:
                continue
            row: dict[str, object] = {
                "equation": equation,
                "method": method,
                "n": len(selected),
                "run_seeds": ";".join("" if run.seed is None else str(run.seed) for run in selected),
                "run_ids": ";".join(run.run_id for run in selected),
            }
            for metric in METRICS:
                values = [getattr(run, metric) for run in selected]
                row[f"{metric}_mean"] = _mean(values)
                row[f"{metric}_std"] = _std(values)
                row[f"{metric}_values"] = _join_float_values(values)
            rows.append(row)
    return rows


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoints_dir", type=Path, default=Path("checkpoints"))
    parser.add_argument("--csv_name", default="training_curve_iteration_normalized.csv")
    parser.add_argument("--max_runs", type=int, default=10)
    parser.add_argument("--out_csv", type=Path, default=Path("final_metrics_summary.csv"))
    parser.add_argument("--out_md", type=Path, default=None)
    parser.add_argument("--out_best_rel_l2_md", type=Path, default=None)
    args = parser.parse_args()

    rows = build_summary(args.checkpoints_dir, args.csv_name, args.max_runs)
    if not rows:
        raise SystemExit("No valid runs found.")

    _write_csv(rows, args.out_csv)
    print(f"Saved CSV: {args.out_csv}")
    if args.out_md is not None:
        _write_markdown(rows, args.out_md)
        print(f"Saved Markdown: {args.out_md}")
    if args.out_best_rel_l2_md is not None:
        _write_best_rel_l2_markdown(rows, args.out_best_rel_l2_md)
        print(f"Saved best rel_l2_error Markdown: {args.out_best_rel_l2_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
