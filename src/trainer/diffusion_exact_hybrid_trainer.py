import os
import random
import torch
import matplotlib.pyplot as plt
import numpy as np

from src.utils.logger import Logging
from src.utils.error_metrics import lp_error
from src.utils.plot_solution_comparison import save_solution_comparison_heatmaps
from src.data.diffusion_exact_dataset import DiffusionExactDataset, load_diffusion_exact_data
import src.trainer.diffusion_exact_train as diffusion_exact_train
from src.nn.DVPDESolver import DVPDESolver
from src.nn.CVPDESolver import CVPDESolver
from src.nn.ClassicalSolver import ClassicalSolver
from src.nn.ClassicalSolver2 import ClassicalSolver2


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

mode = "hybrid"
num_qubits = 5
output_dim = 1
input_dim = 2
hidden_dim = 20
num_quantum_layers = 1
cutoff_dim = 20
classic_network = [input_dim, hidden_dim, output_dim]


args = {
    "batch_size": 64,
    "epochs": 20000,

    "lr": 0.001,
    "seed": 578,
    "print_every": 100,
    "log_path": "./checkpoints/diffusion_exact",
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
    "problem": "diffusion_exact",
    "solver": "DV",  # options : "CV", "Classical", "DV"
    "device": DEVICE,
    "method": "None",
    "cutoff_dim": cutoff_dim,
    "class": "CVNeuralNetwork1",
    "encoding": "angle",
    "use_rar": False,
    "rar_eval_freq": 2000,
    "rar_threshold": 0.0005,
    "rar_num_candidates": 10000,
    "rar_num_add": 100,
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
    "nx": 200,
    "nt": 200,
    "domainN": 500,
    "leftN": 50,
    "rightN": 50,
    "initialN": 50,
    "eval_total_points": 10000,
}


from src.utils.seed import seed_everything

seed = args["seed"]
seed_everything(seed)

log_path = args["log_path"]
logger = Logging(log_path)
logger.print(f"Random seed set to: {seed}")
logger.print("Solving diffusion equation with exact solution u(x,t)=sin(pi x) e^{-t}")
logger.print(f"Domain: (x, t) = [-1, 1] x [0, 1]")

dataset = DiffusionExactDataset(
    nx=args["nx"],
    nt=args["nt"],
    domainN=args["domainN"],
    leftN=args["leftN"],
    rightN=args["rightN"],
    initialN=args["initialN"],
    dist="random",
    device=DEVICE,
    eval_total_points=args["eval_total_points"],
)
train_dataloader = dataset.__getitem__()

if args["solver"] == "CV":
    model = CVPDESolver(args, logger, train_dataloader, DEVICE)
    model.logger.print("Using CV Solver")
elif args["solver"] == "Classical2":
    model = ClassicalSolver2(args, logger, train_dataloader, DEVICE)
    model.logger.print("Using Classical Solver2")
else:
    model = DVPDESolver(args, logger, train_dataloader, DEVICE)
    model.logger.print("Using DV Solver")

model.logger.print(f"The settings used:")
for key, value in args.items():
    model.logger.print(f"{key} : {value}")

total_params = sum(p.numel() for p in model.parameters())
model.logger.print(f"Total number of parameters: {total_params}")

eval_data = None
try:
    if hasattr(dataset, 'get_eval_data'):
        eval_data = dataset.get_eval_data()
except:
    pass

diffusion_exact_train.train(
    model,
    use_rar=args["use_rar"],
    rar_eval_freq=args["rar_eval_freq"],
    rar_threshold=args["rar_threshold"],
    rar_num_candidates=args["rar_num_candidates"],
    rar_num_add=args["rar_num_add"],
    use_hybrid_optimizer=args["use_hybrid_optimizer"],
    adam_iterations=args["adam_iterations"],
    eval_data=eval_data
)

model.save_state()
model.logger.print("Training completed successfully!")

try:
    t, x, u_xt = load_diffusion_exact_data(nx=args["nx"], nt=args["nt"])
    T, X = np.meshgrid(t, x)
    tx = np.concatenate([T.reshape(-1, 1), X.reshape(-1, 1)], axis=1)
    tx_torch = torch.tensor(tx, dtype=torch.float32).to(model.device)

    eval_chunk_size = 500
    total_points = tx_torch.shape[0]
    logger.print(f"Evaluating on {total_points} points using chunk size {eval_chunk_size}")

    u_pred_chunks = []
    with torch.no_grad():
        for i in range(0, total_points, eval_chunk_size):
            chunk = tx_torch[i:i+eval_chunk_size]
            chunk_pred = model.forward(chunk).cpu().numpy()
            u_pred_chunks.append(chunk_pred)

    u_pred = np.concatenate(u_pred_chunks, axis=0)
    u_true = u_xt.reshape(-1, 1)
    lp_error(u_pred, u_true, "RelL2_U%", logger, 2)

    u_pred_grid = u_pred.reshape(u_xt.shape)  # (nx, nt)
    comparison_path = os.path.join(model.log_path, "solution_comparison_diffusion_exact.png")
    save_solution_comparison_heatmaps(
        x=x,
        y=t,
        u_pred=u_pred_grid.T,  # (nt, nx)
        u_true=u_xt.T,
        save_path=comparison_path,
        x_label="x",
        y_label="t",
        title_prefix="Diffusion",
    )
    logger.print(f"Saved solution comparison heatmaps to: {comparison_path}")
except Exception as exc:
    logger.print(f"Skipped evaluation due to error: {exc}")

plt.plot(range(len(model.loss_history)), model.loss_history)
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Training Loss Over Epochs")
plt.grid()

file_path = os.path.join(model.log_path, "loss_history_diffusion.pdf")
plt.savefig(file_path, bbox_inches="tight")
plt.close("all")
