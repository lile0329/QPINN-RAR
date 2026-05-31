import os
import random
import torch
import matplotlib.pyplot as plt
import numpy as np

from src.utils.logger import Logging
from src.utils.error_metrics import lp_error
from src.utils.plot_solution_comparison import save_solution_comparison_heatmaps
from src.data.burgers_exact_dataset import BurgersExactDataset, load_burgers_exact_data
import src.trainer.burgers_exact_train as burgers_exact_train
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
    "seed": 487,
    "print_every": 100,
    "log_path": "./checkpoints/burgers_exact",
    "input_dim": input_dim,
    "output_dim": output_dim,
    "num_qubits": num_qubits,
    "hidden_dim": hidden_dim,
    "num_quantum_layers": num_quantum_layers,
    "classic_network": classic_network,
    "q_ansatz": "cross_mesh",  # options: None , alternate, layered , cascade, cross_mesh ,farhi
    "mode": mode,
    "activation": "tanh",
    "shots": None,  # Analytical gradients enabled
    "problem": "burgers_exact",
    "solver": "DV",  # options : "CV", "Classical", "Classical2", "DV"
    "device": DEVICE,
    "method": "None",
    "cutoff_dim": cutoff_dim,  # num_qubits >= cutoff_dim
    "class": "CVNeuralNetwork1",  # options CVNeuralNetwork1, CVNeuralNetwork2, CVNeuralNetwork3
    "encoding": "angle",  # options : "ampiltude" , "angle" for DV , none for others
    # RAR (Residual-based Adaptive Refinement) parameters
    "use_rar": False,  # Enable RAR adaptive sampling
    "rar_eval_freq": 2000,  # Evaluate residual every N iterations
    "rar_threshold": 0.0005,  # Mean residual threshold for adding points (increased from 0.0001 to reduce overfitting)
    "rar_num_candidates": 10000,  # Number of candidate points to evaluate
    "rar_num_add": 100,  # Number of points to add per RAR iteration (reduced from 500 to prevent over-concentration)
    # Hybrid optimizer parameters
    "use_hybrid_optimizer": True,  # Enable hybrid optimizer (Adam for 10000 iterations, then LBFGS)
    "adam_iterations": 10000,  # Number of Adam iterations before switching to LBFGS
    "use_scipy_lbfgs": True,  # Use scipy L-BFGS-B optimizer (from bburgers_hybrid_trainer.py approach)
    # PyTorch LBFGS parameters (used if use_scipy_lbfgs=False)
    "lbfgs_lr": 1,  # Lower LR to avoid inf loss in LBFGS
    "lbfgs_line_search": "strong_wolfe",
    "lbfgs_tolerance_grad": 1e-05,      # 自定义梯度阈值
    "lbfgs_tolerance_change": 1e-07,     # 自定义变化阈值
    # Scipy LBFGS parameters (used if use_scipy_lbfgs=True, from bburgers_hybrid_trainer.py)
    "lbfgs_maxiter": 10000,   # Maximum number of iterations
    "lbfgs_factr": 1e3,       # Stricter stopping (smaller -> more iterations)
    "lbfgs_pgtol": 1e-14,     # Very small projected-gradient tolerance
    "lbfgs_maxls": 50,       # Maximum number of line search steps
    "lbfgs_m": 100,          # Larger history may aid convergence
    "lbfgs_eval_every": 100,  # Evaluate L2 error every N iterations
    # Dataset parameters
    "nx": 200,  # Number of spatial points
    "nt": 200,  # Number of time points
    # Sampling params (fixed number of points)
    "domainN": 500,  # Fixed number of domain points
    "leftN": 50,  # Fixed number of left boundary points
    "rightN": 50,  # Fixed number of right boundary points
    "initialN": 50,  # Fixed number of initial condition points
    "eval_total_points": 10000,  # Fixed test points number

}


def seed_everything(seed: int):
    # Python-level
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)

    # PyTorch
    torch.manual_seed(seed)
    try:
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except Exception:
        pass

    # Force deterministic algorithms where available
    try:
        torch.use_deterministic_algorithms(True)
    except Exception:
        # Older PyTorch versions may not have this API
        pass

    # Configure CUDA libs for determinism when possible
    if torch.cuda.is_available():
        os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# Set random seeds for reproducibility
seed = args["seed"]
seed_everything(seed)

log_path = args["log_path"]
logger = Logging(log_path)
logger.print(f"Random seed set to: {seed}")
logger.print(f"Solving Burgers equation: u_t + u*u_x = nu*u_xx with nu=0.01")
logger.print(f"Domain: (x, t) = [-1, 1] x [0, 1]")
logger.print(f"Using exact solution for boundary and initial conditions")

# Generate dataset from exact solution
dataset = BurgersExactDataset(
    nx=args["nx"],
    nt=args["nt"],
    domainN=args["domainN"],
    leftN=args["leftN"],
    rightN=args["rightN"],
    initialN=args["initialN"],
    dist="random",  # Use random sampling instead of Sobol sequence
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

# Get evaluation data if available
eval_data = None
try:
    if hasattr(dataset, 'get_eval_data'):
        eval_data = dataset.get_eval_data()
except:
    pass

burgers_exact_train.train(
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
    t, x, u_xt = load_burgers_exact_data(nx=args["nx"], nt=args["nt"])
    T, X = np.meshgrid(t, x)
    tx = np.concatenate([T.reshape(-1, 1), X.reshape(-1, 1)], axis=1)
    tx_torch = torch.tensor(tx, dtype=torch.float32).to(model.device)
    
    # 使用分块处理来避免"value too large"错误
    # 评估时使用较小的块大小以确保稳定性
    eval_chunk_size = 500
    total_points = tx_torch.shape[0]
    logger.print(f"Evaluating on {total_points} points using chunk size {eval_chunk_size}")
    
    u_pred_chunks = []
    with torch.no_grad():
        for i in range(0, total_points, eval_chunk_size):
            chunk = tx_torch[i:i+eval_chunk_size]
            try:
                chunk_pred = model.forward(chunk).cpu().numpy()
                u_pred_chunks.append(chunk_pred)
            except Exception as e:
                # 如果仍然失败，尝试更小的块
                if "value too large" in str(e):
                    logger.print(f"Chunk {i//eval_chunk_size + 1} failed, trying smaller chunks...")
                    smaller_chunk_size = 100
                    sub_chunks = []
                    for j in range(0, chunk.shape[0], smaller_chunk_size):
                        sub_chunk = chunk[j:j+smaller_chunk_size]
                        sub_pred = model.forward(sub_chunk).cpu().numpy()
                        sub_chunks.append(sub_pred)
                    chunk_pred = np.concatenate(sub_chunks, axis=0)
                    u_pred_chunks.append(chunk_pred)
                else:
                    raise
    
    u_pred = np.concatenate(u_pred_chunks, axis=0)
    u_true = u_xt.reshape(-1, 1)
    lp_error(u_pred, u_true, "RelL2_U%", logger, 2)

    u_pred_grid = u_pred.reshape(u_xt.shape)  # (nx, nt)
    comparison_path = os.path.join(model.log_path, "solution_comparison_burgers_exact.png")
    save_solution_comparison_heatmaps(
        x=x,
        y=t,
        u_pred=u_pred_grid.T,  # (nt, nx)
        u_true=u_xt.T,
        save_path=comparison_path,
        x_label="x",
        y_label="t",
        title_prefix="Burgers",
    )
    logger.print(f"Saved solution comparison heatmaps to: {comparison_path}")
except Exception as exc:
    logger.print(f"Skipped evaluation due to error: {exc}")

plt.plot(range(len(model.loss_history)), model.loss_history)
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Training Loss Over Epochs")
plt.grid()

file_path = os.path.join(model.log_path, "loss_history.pdf")
plt.savefig(file_path, bbox_inches="tight")
plt.close("all")

