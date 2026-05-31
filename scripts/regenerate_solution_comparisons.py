import argparse
import os
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


class NullLogger:
    def __init__(self, output_dir: Path):
        self.output_dir = str(output_dir)

    def get_output_dir(self):
        return self.output_dir

    def print(self, *args):
        pass


PROBLEM_TITLES = {
    "burgers_exact": "Burgers",
    "diffusion_exact": "Diffusion",
    "heat3d_exact": "Heat3D slice",
}

RELATIVE_EPS = 1e-12


def finite_minmax(arrays: Iterable[np.ndarray]) -> Tuple[float, float]:
    values = []
    for arr in arrays:
        arr = np.asarray(arr)
        finite = arr[np.isfinite(arr)]
        if finite.size:
            values.append(finite)
    if not values:
        return 0.0, 1.0

    all_values = np.concatenate(values)
    vmin = float(np.min(all_values))
    vmax = float(np.max(all_values))
    if vmin == vmax:
        eps = 1e-12 if vmin == 0.0 else abs(vmin) * 1e-12
        return vmin - eps, vmax + eps
    return vmin, vmax


def relative_l2_error(u_pred: np.ndarray, u_true: np.ndarray) -> float:
    numerator = np.linalg.norm((u_pred - u_true).reshape(-1), ord=2)
    denominator = np.linalg.norm(u_true.reshape(-1), ord=2)
    if denominator <= RELATIVE_EPS:
        return float("nan")
    return float(numerator / denominator)


def relative_error_map(u_pred: np.ndarray, u_true: np.ndarray) -> np.ndarray:
    scale = float(np.linalg.norm(u_true.reshape(-1), ord=2))
    if scale <= RELATIVE_EPS:
        scale = 1.0
    return np.abs(u_pred - u_true) / scale


def save_relative_error_heatmap(
    *,
    x: np.ndarray,
    y: np.ndarray,
    error: np.ndarray,
    save_path: Path,
    x_label: str,
    y_label: str,
    title: str,
    vmin_vmax: Tuple[float, float],
    cmap: str = "magma",
    dpi: int = 300,
) -> None:
    import matplotlib.pyplot as plt
    from mpl_toolkits.axes_grid1 import make_axes_locatable

    save_path = Path(save_path)
    x = np.asarray(x).reshape(-1)
    y = np.asarray(y).reshape(-1)
    error = np.asarray(error)
    if error.shape != (y.size, x.size):
        raise ValueError(f"Expected error shape (ny,nx)=({y.size},{x.size}), got {error.shape}")

    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(1, 1, figsize=(5.2, 4.8))
    im = ax.imshow(
        error,
        origin="lower",
        aspect="auto",
        extent=[float(x.min()), float(x.max()), float(y.min()), float(y.max())],
        vmin=vmin_vmax[0],
        vmax=vmin_vmax[1],
        cmap=cmap,
    )
    ax.set_title(title)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="4.5%", pad=0.06)
    fig.colorbar(im, cax=cax, format="%.1e")
    plt.tight_layout()
    fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def load_checkpoint(path: Path, device) -> Dict:
    import torch

    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


def build_model(model_path: Path, device):
    from src.nn.CVPDESolver import CVPDESolver
    from src.nn.ClassicalSolver import ClassicalSolver
    from src.nn.ClassicalSolver2 import ClassicalSolver2
    from src.nn.DVPDESolver import DVPDESolver

    state = load_checkpoint(model_path, device)
    args = dict(state["args"])
    args["device"] = device
    args["log_path"] = str(model_path.parent)

    solver = args.get("solver", "DV")
    logger = NullLogger(model_path.parent)
    if solver == "CV":
        model = CVPDESolver(args, logger, data=None, device=device)
    elif solver == "Classical":
        model = ClassicalSolver(args, logger, data=None, device=device)
    elif solver == "Classical2":
        model = ClassicalSolver2(args, logger, data=None, device=device)
    else:
        model = DVPDESolver(args, logger, data=None, device=device)

    model.preprocessor.load_state_dict(state["preprocessor"])
    if "hidden_network" in state and hasattr(model, "hidden"):
        model.hidden.load_state_dict(state["hidden_network"])
    if "quantum_layer" in state and hasattr(model, "quantum_layer"):
        model.quantum_layer.load_state_dict(state["quantum_layer"])
    model.postprocessor.load_state_dict(state["postprocessor"])
    if hasattr(model, "draw_quantum_circuit_flag"):
        model.draw_quantum_circuit_flag = False

    model.to(device)
    model.eval()
    return model, args


def predict_in_chunks(model, inputs, chunk_size: int) -> np.ndarray:
    outputs = []
    import torch

    with torch.no_grad():
        for start in range(0, inputs.shape[0], chunk_size):
            chunk = inputs[start : start + chunk_size]
            try:
                outputs.append(model.forward(chunk).detach().cpu().numpy())
            except Exception as exc:
                if chunk_size <= 100 or "value too large" not in str(exc):
                    raise
                outputs.append(predict_in_chunks(model, chunk, 100))
    return np.concatenate(outputs, axis=0)


def evaluate_checkpoint(model_path: Path, device, chunk_size: int) -> Dict:
    import torch

    model, args = build_model(model_path, device)
    problem = args.get("problem", model_path.parent.parent.name)
    nx = int(args.get("nx", 200))
    nt = int(args.get("nt", 200))

    if problem == "burgers_exact":
        from src.data.burgers_exact_dataset import load_burgers_exact_data

        t, x, u_xt = load_burgers_exact_data(nx=nx, nt=nt)
        t_grid, x_grid = np.meshgrid(t, x)
        inputs = np.concatenate([t_grid.reshape(-1, 1), x_grid.reshape(-1, 1)], axis=1)
        u_true = u_xt.T
        output_name = "solution_comparison_burgers_exact_unified.png"
        relative_output_name = "relative_l2_error_burgers_exact_unified.png"
        x_label, y_label = "x", "t"
    elif problem == "diffusion_exact":
        from src.data.diffusion_exact_dataset import load_diffusion_exact_data

        t, x, u_xt = load_diffusion_exact_data(nx=nx, nt=nt)
        t_grid, x_grid = np.meshgrid(t, x)
        inputs = np.concatenate([t_grid.reshape(-1, 1), x_grid.reshape(-1, 1)], axis=1)
        u_true = u_xt.T
        output_name = "solution_comparison_diffusion_exact_unified.png"
        relative_output_name = "relative_l2_error_diffusion_exact_unified.png"
        x_label, y_label = "x", "t"
    elif problem == "heat3d_exact":
        from src.data.heat3d_exact_dataset import load_heat3d_exact_slice_data

        y0 = float(args.get("y0", 0.5))
        z0 = float(args.get("z0", 0.5))
        alpha = float(args.get("alpha", 0.1))
        t, x, u_xt = load_heat3d_exact_slice_data(nx=nx, nt=nt, y0=y0, z0=z0, alpha=alpha)
        t_grid, x_grid = np.meshgrid(t, x)
        inputs = np.concatenate(
            [
                t_grid.reshape(-1, 1),
                x_grid.reshape(-1, 1),
                np.full((t_grid.size, 1), y0),
                np.full((t_grid.size, 1), z0),
            ],
            axis=1,
        )
        u_true = u_xt.T
        output_name = "solution_comparison_heat3d_unified.png"
        relative_output_name = "relative_l2_error_heat3d_unified.png"
        x_label, y_label = "x", "t"
    else:
        raise ValueError(f"Unsupported problem {problem!r} in {model_path}")

    inputs_t = torch.tensor(inputs, dtype=torch.float32, device=device)
    u_pred = predict_in_chunks(model, inputs_t, chunk_size).reshape(nx, nt).T
    return {
        "problem": problem,
        "path": model_path.parent,
        "x": x,
        "y": t,
        "u_pred": u_pred,
        "u_true": u_true,
        "output_name": output_name,
        "relative_output_name": relative_output_name,
        "x_label": x_label,
        "y_label": y_label,
        "relative_l2": relative_l2_error(u_pred, u_true),
        "relative_error_map": relative_error_map(u_pred, u_true),
    }


def find_model_paths(checkpoints_dir: Path, problems: List[str]) -> List[Path]:
    paths = []
    for problem in problems:
        problem_dir = checkpoints_dir / problem
        if not problem_dir.exists():
            continue
        paths.extend(sorted(problem_dir.glob("*/model.pth")))
    return paths


def parse_args():
    parser = argparse.ArgumentParser(
        description="Regenerate solution_comparison figures from model.pth with shared axes and color scales per problem."
    )
    parser.add_argument("--checkpoints", default="checkpoints", help="Root checkpoints directory.")
    parser.add_argument(
        "--problem",
        choices=sorted(PROBLEM_TITLES),
        action="append",
        help="Problem to process. Can be passed more than once. Default: all supported problems.",
    )
    parser.add_argument("--chunk-size", type=int, default=500, help="Forward-pass chunk size.")
    parser.add_argument("--dpi", type=int, default=300, help="Output image DPI.")
    parser.add_argument("--device", default=None, help="cpu, cuda, or cuda:0. Default: cuda if available, else cpu.")
    parser.add_argument(
        "--only-relative-l2",
        action="store_true",
        help="Only write the new relative-L2-style figures, not solution_comparison_*_unified.png.",
    )
    parser.add_argument(
        "--relative-vmax",
        type=float,
        default=None,
        help="Optional shared upper colorbar limit for relative error maps.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Only list checkpoints that would be processed.")
    return parser.parse_args()


def main():
    args = parse_args()
    root = Path(args.checkpoints)
    problems = args.problem or sorted(PROBLEM_TITLES)

    model_paths = find_model_paths(root, problems)
    if not model_paths:
        print("No model.pth files found.")
        return

    print(f"Found {len(model_paths)} checkpoints:")
    for path in model_paths:
        print(f"  {path}")
    if args.dry_run:
        return

    import torch
    from src.utils.plot_solution_comparison import save_solution_comparison_heatmaps

    device_name = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(device_name)

    results_by_problem: Dict[str, List[Dict]] = {}
    for index, model_path in enumerate(model_paths, start=1):
        print(f"[{index}/{len(model_paths)}] Evaluating {model_path}")
        result = evaluate_checkpoint(model_path, device, args.chunk_size)
        results_by_problem.setdefault(result["problem"], []).append(result)

    for problem, results in results_by_problem.items():
        solution_range = finite_minmax(
            value for item in results for value in (item["u_true"], item["u_pred"])
        )
        error_range = (0.0, finite_minmax(np.abs(item["u_pred"] - item["u_true"]) for item in results)[1])
        relative_error_range = (
            0.0,
            finite_minmax(item["relative_error_map"] for item in results)[1],
        )
        if args.relative_vmax is not None:
            relative_error_range = (0.0, float(args.relative_vmax))
        print(f"{problem}: solution range={solution_range}, error range={error_range}")
        print(f"{problem}: relative error map range={relative_error_range}")

        for item in results:
            if not args.only_relative_l2:
                save_path = item["path"] / item["output_name"]
                save_solution_comparison_heatmaps(
                    x=item["x"],
                    y=item["y"],
                    u_pred=item["u_pred"],
                    u_true=item["u_true"],
                    save_path=str(save_path),
                    x_label=item["x_label"],
                    y_label=item["y_label"],
                    title_prefix=PROBLEM_TITLES.get(problem, problem),
                    solution_vmin_vmax=solution_range,
                    error_vmin_vmax=error_range,
                    dpi=args.dpi,
                )
                print(f"Saved {save_path}")

            relative_save_path = item["path"] / item["relative_output_name"]
            save_relative_error_heatmap(
                x=item["x"],
                y=item["y"],
                error=item["relative_error_map"],
                save_path=str(relative_save_path),
                x_label=item["x_label"],
                y_label=item["y_label"],
                title=f"{PROBLEM_TITLES.get(problem, problem)} Relative Error, RelL2={item['relative_l2']:.2e}",
                vmin_vmax=relative_error_range,
                dpi=args.dpi,
            )
            print(f"Saved {relative_save_path}")


if __name__ == "__main__":
    main()
