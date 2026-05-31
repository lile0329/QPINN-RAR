import os
import torch
import matplotlib.pyplot as plt
import numpy as np

from src.utils.logger import Logging
from src.utils.error_metrics import lp_error
from src.utils.plot_solution_comparison import save_solution_comparison_heatmaps
from src.data.heat3d_exact_dataset import Heat3DExactDataset, load_heat3d_exact_slice_data
import src.trainer.heat3d_exact_train as heat3d_exact_train
from src.nn.DVPDESolver import DVPDESolver
from src.nn.CVPDESolver import CVPDESolver
from src.nn.ClassicalSolver import ClassicalSolver
from src.nn.ClassicalSolver2 import ClassicalSolver2

from src.utils.seed import seed_everything


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- network / quantum settings
mode = "hybrid"
num_qubits = 5
output_dim = 1
input_dim = 4  # (t, x, y, z)
hidden_dim = 20
num_quantum_layers = 1
cutoff_dim = 20
classic_network = [input_dim, hidden_dim, output_dim]

# --- training arguments (tweak as needed)
args = {
    "batch_size": 64,
    "epochs": 20000,
    "lr": 0.001,
    "seed": 10,
    "print_every": 100,
    "log_path": "./checkpoints/heat3d_exact",
    "input_dim": input_dim,
    "output_dim": output_dim,
    "num_qubits": num_qubits,
    "hidden_dim": hidden_dim,
    "num_quantum_layers": num_quantum_layers,
    "classic_network": classic_network,
    "q_ansatz": "cross_mesh",  # options: None , alternate, layered , cascade, cross_mesh ,farhi
    "mode": mode,
    "activation": "tanh",
    "shots": None,
    "problem": "heat3d_exact",
    "solver": "DV",  # options : "CV", "Classical", "DV", "Classical2"
    "device": DEVICE,
    "method": "None",
    "cutoff_dim": cutoff_dim,
    "class": "CVNeuralNetwork1",
    "encoding": "angle",
    # RAR / adaptivity
    "use_rar": False,
    "rar_eval_freq": 2000,
    "rar_threshold": 0.0005,
    "rar_num_candidates": 10000,
    "rar_num_add": 100,
    # hybrid optimizer settings
    "use_hybrid_optimizer": True,
    "adam_iterations": 10000,
    "use_scipy_lbfgs": True,
    "lbfgs_lr": 1,
    "lbfgs_line_search": "strong_wolfe",
    "lbfgs_tolerance_grad": 1e-05,
    "lbfgs_tolerance_change": 1e-07,
    "lbfgs_maxiter": 10000,
    "lbfgs_factr": 1e3,
    "lbfgs_pgtol": 1e-14,
    "lbfgs_maxls": 50,
    "lbfgs_m": 100,
    "lbfgs_eval_every": 100,
    # dataset parameters
    "domainN": 500,
    "boundaryN": 50,
    "initialN": 50,
    "eval_total_points": 10000,
    # slice evaluation parameters
    "nx": 200,
    "nt": 200,
    "y0": 0.5,
    "z0": 0.5,
    "alpha": 0.1,
}

# reproducibility
seed = args["seed"]
seed_everything(seed)

log_path = args["log_path"]
logger = Logging(log_path)
logger.print(f"Random seed set to: {seed}")
logger.print("Solving 3D heat equation u_t = alpha (u_{xx}+u_{yy}+u_{zz}) on [0,1]^3 with homogeneous Dirichlet BC")

# prepare dataset

dataset = Heat3DExactDataset(
    domainN=args["domainN"],
    boundaryN=args["boundaryN"],
    initialN=args["initialN"],
    alpha=args["alpha"],
    seed=args["seed"],
    device=DEVICE,
    eval_total_points=args["eval_total_points"],
)
train_dataloader = dataset.__getitem__()

# select solver
if args["solver"] == "CV":
    model = CVPDESolver(args, logger, train_dataloader, DEVICE)
    model.logger.print("Using CV Solver")
elif args["solver"] == "Classical2":
    model = ClassicalSolver2(args, logger, train_dataloader, DEVICE)
    model.logger.print("Using Classical Solver2")
elif args["solver"] == "Classical":
    model = ClassicalSolver(args, logger, train_dataloader, DEVICE)
    model.logger.print("Using Classical Solver")
else:
    model = DVPDESolver(args, logger, train_dataloader, DEVICE)
    model.logger.print("Using DV Solver")

# print configuration
model.logger.print("The settings used:")
for key, value in args.items():
    model.logger.print(f"{key} : {value}")

total_params = sum(p.numel() for p in model.parameters())
model.logger.print(f"Total number of parameters: {total_params}")

# evaluation data if available

eval_data = None
try:
    if hasattr(dataset, "get_eval_data"):
        eval_data = dataset.get_eval_data()
except Exception:
    pass

# run training

heat3d_exact_train.train(
    model,
    use_rar=args["use_rar"],
    rar_eval_freq=args["rar_eval_freq"],
    rar_threshold=args["rar_threshold"],
    rar_num_candidates=args["rar_num_candidates"],
    rar_num_add=args["rar_num_add"],
    use_hybrid_optimizer=args["use_hybrid_optimizer"],
    adam_iterations=args["adam_iterations"],
    eval_data=eval_data,
)

model.save_state()
model.logger.print("Training completed successfully!")

# compute relative L2 error on evaluation set
try:
    rel_err = heat3d_exact_train.compute_l2_relative_error(model, eval_data=eval_data)
    logger.print(f"Relative L2 error (eval data): {rel_err:.2e}")
except Exception as exc:
    logger.print(f"Skipped eval error computation: {exc}")

# slice visualization using precomputed exact solution
try:
    t_slice, x_slice, u_true_xt = load_heat3d_exact_slice_data(
        nx=args["nx"], nt=args["nt"], y0=args["y0"], z0=args["z0"], alpha=args["alpha"]
    )
    # build prediction grid
    T, X = np.meshgrid(t_slice, x_slice)  # shapes (nx, nt)
    t_flat = torch.tensor(T.flatten(), dtype=torch.float32).unsqueeze(1).to(DEVICE)
    x_flat = torch.tensor(X.flatten(), dtype=torch.float32).unsqueeze(1).to(DEVICE)
    y_flat = torch.full_like(t_flat, float(args["y0"]))
    z_flat = torch.full_like(t_flat, float(args["z0"]))
    inp = torch.cat([t_flat, x_flat, y_flat, z_flat], dim=1)

    eval_chunk_size = 500
    u_pred_chunks = []
    with torch.no_grad():
        for i in range(0, inp.shape[0], eval_chunk_size):
            chunk = inp[i : i + eval_chunk_size]
            u_pred_chunks.append(model.forward(chunk).cpu().numpy())
    u_pred_flat = np.concatenate(u_pred_chunks, axis=0)
    u_pred_grid = u_pred_flat.reshape(args["nx"], args["nt"]).T  # (nt, nx)

    comparison_path = os.path.join(model.log_path, "solution_comparison_heat3d.png")
    save_solution_comparison_heatmaps(
        x=x_slice,
        y=t_slice,
        u_pred=u_pred_grid,
        u_true=u_true_xt.T,
        save_path=comparison_path,
        x_label="x",
        y_label="t",
        title_prefix="Heat3D slice",
    )
    logger.print(f"Saved solution comparison heatmaps to: {comparison_path}")
except Exception as exc:
    logger.print(f"Skipped slice visualization: {exc}")

# loss history plot

plt.plot(range(len(model.loss_history)), model.loss_history)
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Training Loss Over Epochs")
plt.grid()

file_path = os.path.join(model.log_path, "loss_history_heat3d.pdf")
plt.savefig(file_path, bbox_inches="tight")
plt.close("all")
