import numpy as np
import torch

# Parameters for the diffusion equation
NU = 1.0 / (np.pi ** 2)  # choose nu so that solution decays as e^{-t}
X_MIN, X_MAX = -1.0, 1.0
T_MIN, T_MAX = 0.0, 1.0


def exact_solution(x, t):
    """
    Exact solution: u(x,t) = sin(pi x) * exp(-t)
    """
    x = np.asarray(x)
    t = np.asarray(t)
    return np.sin(np.pi * x) * np.exp(-t)


def generate_training_data(nx=200, nt=200, domainN=1000, leftN=50, rightN=50, initialN=50, dist="random"):
    x = np.linspace(X_MIN, X_MAX, nx)
    t = np.linspace(T_MIN, T_MAX, nt)
    T, X = np.meshgrid(t, x)

    u_xt = exact_solution(X, T)

    domain = np.concatenate([T.reshape(-1, 1), X.reshape(-1, 1), u_xt.reshape(-1, 1)], axis=1)

    u_left = exact_solution(X_MIN, t)
    left = np.concatenate([t.reshape(-1, 1), np.full((t.size, 1), X_MIN), u_left.reshape(-1, 1)], axis=1)

    u_right = exact_solution(X_MAX, t)
    right = np.concatenate([t.reshape(-1, 1), np.full((t.size, 1), X_MAX), u_right.reshape(-1, 1)], axis=1)

    u_initial = exact_solution(x, T_MIN)
    initial = np.concatenate([np.full((x.size, 1), T_MIN), x.reshape(-1, 1), u_initial.reshape(-1, 1)], axis=1)

    data = [domain, left, right, initial]
    training_dataset = []
    eval_dataset = []

    fixed_counts = [domainN, leftN, rightN, initialN]

    for d, count in zip(data, fixed_counts):
        actual_count = min(count, d.shape[0])
        if actual_count <= 0:
            training_dataset.append(np.empty((0, 3)))
            eval_dataset.append(d)
        else:
            if dist == "Sobol":
                # fallback to random if Sobol not used
                idx = np.random.choice(d.shape[0], actual_count, replace=False)
            else:
                idx = np.random.choice(d.shape[0], actual_count, replace=False)
            training_dataset.append(d[idx, :])
            eval_idx = np.setdiff1d(np.arange(d.shape[0]), idx)
            eval_dataset.append(d[eval_idx, :])

    domain, left, right, initial = training_dataset
    domain_eval, left_eval, right_eval, initial_eval = eval_dataset

    return [domain, left, right, initial], [domain_eval, left_eval, right_eval, initial_eval]


class DiffusionExactDataset(object):
    def __init__(self, nx=200, nt=200, domainN=1000, leftN=50, rightN=50, initialN=50, dist="random", device=None, eval_total_points=None):
        [domain, left, right, initial], [domain_eval, left_eval, right_eval, initial_eval] = generate_training_data(
            nx, nt, domainN, leftN, rightN, initialN, dist
        )

        # convert to torch tensors and move to device if provided
        if domain.shape[0] > 0:
            self.tx_domain = torch.tensor(domain[:, 0:2], dtype=torch.float32).to(device)
            self.u_domain = torch.tensor(domain[:, 2:3], dtype=torch.float32).to(device)
        else:
            self.tx_domain = torch.empty((0, 2), dtype=torch.float32).to(device)
            self.u_domain = torch.empty((0, 1), dtype=torch.float32).to(device)

        if left.shape[0] > 0:
            self.tx_left = torch.tensor(left[:, 0:2], dtype=torch.float32).to(device)
            self.u_left = torch.tensor(left[:, 2:3], dtype=torch.float32).to(device)
        else:
            self.tx_left = torch.empty((0, 2), dtype=torch.float32).to(device)
            self.u_left = torch.empty((0, 1), dtype=torch.float32).to(device)

        if right.shape[0] > 0:
            self.tx_right = torch.tensor(right[:, 0:2], dtype=torch.float32).to(device)
            self.u_right = torch.tensor(right[:, 2:3], dtype=torch.float32).to(device)
        else:
            self.tx_right = torch.empty((0, 2), dtype=torch.float32).to(device)
            self.u_right = torch.empty((0, 1), dtype=torch.float32).to(device)

        if initial.shape[0] > 0:
            self.tx_initial = torch.tensor(initial[:, 0:2], dtype=torch.float32).to(device)
            self.u_initial = torch.tensor(initial[:, 2:3], dtype=torch.float32).to(device)
        else:
            self.tx_initial = torch.empty((0, 2), dtype=torch.float32).to(device)
            self.u_initial = torch.empty((0, 1), dtype=torch.float32).to(device)

        if domain_eval.shape[0] > 0:
            self.tx_domain_eval = torch.tensor(domain_eval[:, 0:2], dtype=torch.float32).to(device)
            self.u_domain_eval = torch.tensor(domain_eval[:, 2:3], dtype=torch.float32).to(device)
        else:
            self.tx_domain_eval = torch.empty((0, 2), dtype=torch.float32).to(device)
            self.u_domain_eval = torch.empty((0, 1), dtype=torch.float32).to(device)

        if left_eval.shape[0] > 0:
            self.tx_left_eval = torch.tensor(left_eval[:, 0:2], dtype=torch.float32).to(device)
            self.u_left_eval = torch.tensor(left_eval[:, 2:3], dtype=torch.float32).to(device)
        else:
            self.tx_left_eval = torch.empty((0, 2), dtype=torch.float32).to(device)
            self.u_left_eval = torch.empty((0, 1), dtype=torch.float32).to(device)

        if right_eval.shape[0] > 0:
            self.tx_right_eval = torch.tensor(right_eval[:, 0:2], dtype=torch.float32).to(device)
            self.u_right_eval = torch.tensor(right_eval[:, 2:3], dtype=torch.float32).to(device)
        else:
            self.tx_right_eval = torch.empty((0, 2), dtype=torch.float32).to(device)
            self.u_right_eval = torch.empty((0, 1), dtype=torch.float32).to(device)

        if initial_eval.shape[0] > 0:
            self.tx_initial_eval = torch.tensor(initial_eval[:, 0:2], dtype=torch.float32).to(device)
            self.u_initial_eval = torch.tensor(initial_eval[:, 2:3], dtype=torch.float32).to(device)
        else:
            self.tx_initial_eval = torch.empty((0, 2), dtype=torch.float32).to(device)
            self.u_initial_eval = torch.empty((0, 1), dtype=torch.float32).to(device)

        self.size = max(domain.shape[0], left.shape[0], right.shape[0], initial.shape[0])

    def __getitem__(self):
        return (
            dict(
                {
                    "tx_domain": self.tx_domain,
                    "tx_left": self.tx_left,
                    "tx_right": self.tx_right,
                    "tx_initial": self.tx_initial,
                }
            ),
            dict(
                {
                    "u_domain": self.u_domain,
                    "u_left": self.u_left,
                    "u_right": self.u_right,
                    "u_initial": self.u_initial,
                }
            ),
        )

    def get_eval_data(self):
        return (
            dict(
                {
                    "tx_domain": self.tx_domain_eval,
                    "tx_left": self.tx_left_eval,
                    "tx_right": self.tx_right_eval,
                    "tx_initial": self.tx_initial_eval,
                }
            ),
            dict(
                {
                    "u_domain": self.u_domain_eval,
                    "u_left": self.u_left_eval,
                    "u_right": self.u_right_eval,
                    "u_initial": self.u_initial_eval,
                }
            ),
        )


def load_diffusion_exact_data(nx=200, nt=200):
    x = np.linspace(X_MIN, X_MAX, nx)
    t = np.linspace(T_MIN, T_MAX, nt)
    T, X = np.meshgrid(t, x)
    u_xt = exact_solution(X, T)
    return t, x, u_xt
