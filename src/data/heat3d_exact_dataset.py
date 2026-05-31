import numpy as np
import torch

# Domain for the 3D heat equation
X_MIN, X_MAX = 0.0, 1.0
Y_MIN, Y_MAX = 0.0, 1.0
Z_MIN, Z_MAX = 0.0, 1.0
T_MIN, T_MAX = 0.0, 2.0


def exact_solution(x, y, z, t, alpha=0.1):
    """
    Exact solution for the 3D heat equation on [0,1]^3 with homogeneous Dirichlet BC:
        u(x,y,z,t) = exp(-3*pi^2*alpha*t) * sin(pi x) sin(pi y) sin(pi z)
    """
    x = np.asarray(x)
    y = np.asarray(y)
    z = np.asarray(z)
    t = np.asarray(t)
    return (
        np.exp(-3.0 * (np.pi**2) * alpha * t)
        * np.sin(np.pi * x)
        * np.sin(np.pi * y)
        * np.sin(np.pi * z)
    )


def _rand_uniform(n, low, high, rng):
    return rng.uniform(low=low, high=high, size=(n, 1))


def generate_training_data(
    *,
    domainN=2000,
    boundaryN=200,
    initialN=500,
    alpha=0.1,
    seed=42,
):
    """
    Generate random training points for the 3D heat equation.

    Returns:
        (tx_train_dict, u_train_dict), (tx_eval_dict, u_eval_dict)
    """
    rng = np.random.default_rng(seed)

    # --- Domain points: (t,x,y,z)
    t_d = _rand_uniform(domainN, T_MIN, T_MAX, rng)
    x_d = _rand_uniform(domainN, X_MIN, X_MAX, rng)
    y_d = _rand_uniform(domainN, Y_MIN, Y_MAX, rng)
    z_d = _rand_uniform(domainN, Z_MIN, Z_MAX, rng)
    u_d = exact_solution(x_d, y_d, z_d, t_d, alpha=alpha)

    tx_domain = np.concatenate([t_d, x_d, y_d, z_d], axis=1)
    u_domain = u_d.reshape(-1, 1)

    # --- Boundary faces (6): x=0, x=1, y=0, y=1, z=0, z=1
    def _sample_face(*, fixed_dim: str, fixed_value: float, n: int):
        t = _rand_uniform(n, T_MIN, T_MAX, rng)
        x = _rand_uniform(n, X_MIN, X_MAX, rng)
        y = _rand_uniform(n, Y_MIN, Y_MAX, rng)
        z = _rand_uniform(n, Z_MIN, Z_MAX, rng)
        if fixed_dim == "x":
            x[:] = fixed_value
        elif fixed_dim == "y":
            y[:] = fixed_value
        elif fixed_dim == "z":
            z[:] = fixed_value
        else:
            raise ValueError(f"Unknown fixed_dim={fixed_dim}")
        tx = np.concatenate([t, x, y, z], axis=1)
        u = exact_solution(x, y, z, t, alpha=alpha).reshape(-1, 1)
        return tx, u

    tx_x0, u_x0 = _sample_face(fixed_dim="x", fixed_value=X_MIN, n=boundaryN)
    tx_x1, u_x1 = _sample_face(fixed_dim="x", fixed_value=X_MAX, n=boundaryN)
    tx_y0, u_y0 = _sample_face(fixed_dim="y", fixed_value=Y_MIN, n=boundaryN)
    tx_y1, u_y1 = _sample_face(fixed_dim="y", fixed_value=Y_MAX, n=boundaryN)
    tx_z0, u_z0 = _sample_face(fixed_dim="z", fixed_value=Z_MIN, n=boundaryN)
    tx_z1, u_z1 = _sample_face(fixed_dim="z", fixed_value=Z_MAX, n=boundaryN)

    # --- Initial condition: t=0
    t_i = np.full((initialN, 1), T_MIN, dtype=np.float64)
    x_i = _rand_uniform(initialN, X_MIN, X_MAX, rng)
    y_i = _rand_uniform(initialN, Y_MIN, Y_MAX, rng)
    z_i = _rand_uniform(initialN, Z_MIN, Z_MAX, rng)
    u_i = exact_solution(x_i, y_i, z_i, t_i, alpha=alpha).reshape(-1, 1)
    tx_initial = np.concatenate([t_i, x_i, y_i, z_i], axis=1)
    u_initial = u_i

    tx_train = {
        "tx_domain": tx_domain,
        "tx_x0": tx_x0,
        "tx_x1": tx_x1,
        "tx_y0": tx_y0,
        "tx_y1": tx_y1,
        "tx_z0": tx_z0,
        "tx_z1": tx_z1,
        "tx_initial": tx_initial,
    }
    u_train = {
        "u_domain": u_domain,
        "u_x0": u_x0,
        "u_x1": u_x1,
        "u_y0": u_y0,
        "u_y1": u_y1,
        "u_z0": u_z0,
        "u_z1": u_z1,
        "u_initial": u_initial,
    }

    # Evaluation: independent random sample (kept small, no giant 4D grid)
    eval_total = max(1, int(domainN))  # default: comparable scale to domain
    eval_domain = int(0.7 * eval_total)
    eval_each_face = int(0.2 * eval_total / 6)
    eval_initial = max(1, eval_total - eval_domain - 6 * eval_each_face)

    t_ed = _rand_uniform(eval_domain, T_MIN, T_MAX, rng)
    x_ed = _rand_uniform(eval_domain, X_MIN, X_MAX, rng)
    y_ed = _rand_uniform(eval_domain, Y_MIN, Y_MAX, rng)
    z_ed = _rand_uniform(eval_domain, Z_MIN, Z_MAX, rng)
    tx_domain_eval = np.concatenate([t_ed, x_ed, y_ed, z_ed], axis=1)
    u_domain_eval = exact_solution(x_ed, y_ed, z_ed, t_ed, alpha=alpha).reshape(-1, 1)

    def _eval_face(fixed_dim: str, fixed_value: float):
        return _sample_face(fixed_dim=fixed_dim, fixed_value=fixed_value, n=eval_each_face)

    tx_x0_eval, u_x0_eval = _eval_face("x", X_MIN)
    tx_x1_eval, u_x1_eval = _eval_face("x", X_MAX)
    tx_y0_eval, u_y0_eval = _eval_face("y", Y_MIN)
    tx_y1_eval, u_y1_eval = _eval_face("y", Y_MAX)
    tx_z0_eval, u_z0_eval = _eval_face("z", Z_MIN)
    tx_z1_eval, u_z1_eval = _eval_face("z", Z_MAX)

    t_ei = np.full((eval_initial, 1), T_MIN, dtype=np.float64)
    x_ei = _rand_uniform(eval_initial, X_MIN, X_MAX, rng)
    y_ei = _rand_uniform(eval_initial, Y_MIN, Y_MAX, rng)
    z_ei = _rand_uniform(eval_initial, Z_MIN, Z_MAX, rng)
    tx_initial_eval = np.concatenate([t_ei, x_ei, y_ei, z_ei], axis=1)
    u_initial_eval = exact_solution(x_ei, y_ei, z_ei, t_ei, alpha=alpha).reshape(-1, 1)

    tx_eval = {
        "tx_domain": tx_domain_eval,
        "tx_x0": tx_x0_eval,
        "tx_x1": tx_x1_eval,
        "tx_y0": tx_y0_eval,
        "tx_y1": tx_y1_eval,
        "tx_z0": tx_z0_eval,
        "tx_z1": tx_z1_eval,
        "tx_initial": tx_initial_eval,
    }
    u_eval = {
        "u_domain": u_domain_eval,
        "u_x0": u_x0_eval,
        "u_x1": u_x1_eval,
        "u_y0": u_y0_eval,
        "u_y1": u_y1_eval,
        "u_z0": u_z0_eval,
        "u_z1": u_z1_eval,
        "u_initial": u_initial_eval,
    }

    return (tx_train, u_train), (tx_eval, u_eval)


class Heat3DExactDataset(object):
    def __init__(
        self,
        *,
        domainN=2000,
        boundaryN=200,
        initialN=500,
        alpha=0.1,
        seed=42,
        device=None,
        eval_total_points=10000,
    ):
        (tx_train, u_train), (tx_eval, u_eval) = generate_training_data(
            domainN=domainN,
            boundaryN=boundaryN,
            initialN=initialN,
            alpha=alpha,
            seed=seed,
        )

        # override eval sizes if requested
        if eval_total_points is not None:
            # regenerate eval set with desired total size
            # (keep training fixed)
            rng = np.random.default_rng(seed + 1)
            eval_domain = int(0.7 * eval_total_points)
            eval_each_face = int(0.2 * eval_total_points / 6)
            eval_initial = max(1, eval_total_points - eval_domain - 6 * eval_each_face)

            def _rand(n, low, high):
                return rng.uniform(low=low, high=high, size=(n, 1))

            t_ed = _rand(eval_domain, T_MIN, T_MAX)
            x_ed = _rand(eval_domain, X_MIN, X_MAX)
            y_ed = _rand(eval_domain, Y_MIN, Y_MAX)
            z_ed = _rand(eval_domain, Z_MIN, Z_MAX)
            tx_eval["tx_domain"] = np.concatenate([t_ed, x_ed, y_ed, z_ed], axis=1)
            u_eval["u_domain"] = exact_solution(x_ed, y_ed, z_ed, t_ed, alpha=alpha).reshape(-1, 1)

            def _eval_face(fixed_dim: str, fixed_value: float):
                t = _rand(eval_each_face, T_MIN, T_MAX)
                x = _rand(eval_each_face, X_MIN, X_MAX)
                y = _rand(eval_each_face, Y_MIN, Y_MAX)
                z = _rand(eval_each_face, Z_MIN, Z_MAX)
                if fixed_dim == "x":
                    x[:] = fixed_value
                elif fixed_dim == "y":
                    y[:] = fixed_value
                elif fixed_dim == "z":
                    z[:] = fixed_value
                else:
                    raise ValueError(f"Unknown fixed_dim={fixed_dim}")
                tx = np.concatenate([t, x, y, z], axis=1)
                u = exact_solution(x, y, z, t, alpha=alpha).reshape(-1, 1)
                return tx, u

            tx_eval["tx_x0"], u_eval["u_x0"] = _eval_face("x", X_MIN)
            tx_eval["tx_x1"], u_eval["u_x1"] = _eval_face("x", X_MAX)
            tx_eval["tx_y0"], u_eval["u_y0"] = _eval_face("y", Y_MIN)
            tx_eval["tx_y1"], u_eval["u_y1"] = _eval_face("y", Y_MAX)
            tx_eval["tx_z0"], u_eval["u_z0"] = _eval_face("z", Z_MIN)
            tx_eval["tx_z1"], u_eval["u_z1"] = _eval_face("z", Z_MAX)

            t_ei = np.full((eval_initial, 1), T_MIN, dtype=np.float64)
            x_ei = _rand(eval_initial, X_MIN, X_MAX)
            y_ei = _rand(eval_initial, Y_MIN, Y_MAX)
            z_ei = _rand(eval_initial, Z_MIN, Z_MAX)
            tx_eval["tx_initial"] = np.concatenate([t_ei, x_ei, y_ei, z_ei], axis=1)
            u_eval["u_initial"] = exact_solution(x_ei, y_ei, z_ei, t_ei, alpha=alpha).reshape(-1, 1)

        def _to_torch(d):
            return {k: torch.tensor(v, dtype=torch.float32).to(device) for k, v in d.items()}

        self.tx_train = _to_torch(tx_train)
        self.u_train = _to_torch(u_train)
        self.tx_eval = _to_torch(tx_eval)
        self.u_eval = _to_torch(u_eval)

        # for RAR bounds etc.
        self.min_x = torch.min(self.tx_train["tx_domain"], dim=0)[0]
        self.max_x = torch.max(self.tx_train["tx_domain"], dim=0)[0]

        self.size = int(max(
            self.tx_train["tx_domain"].shape[0],
            self.tx_train["tx_x0"].shape[0],
            self.tx_train["tx_x1"].shape[0],
            self.tx_train["tx_y0"].shape[0],
            self.tx_train["tx_y1"].shape[0],
            self.tx_train["tx_z0"].shape[0],
            self.tx_train["tx_z1"].shape[0],
            self.tx_train["tx_initial"].shape[0],
        ))

    def __getitem__(self):
        return self.tx_train, self.u_train

    def get_eval_data(self):
        return self.tx_eval, self.u_eval

    def __len__(self):
        return self.size


def load_heat3d_exact_slice_data(*, nx=200, nt=200, y0=0.5, z0=0.5, alpha=0.1):
    """
    Convenience function for plotting: returns a (x,t) slice at fixed (y,z).
    Returns:
        t (nt,), x (nx,), u_xt (nx, nt)
    """
    x = np.linspace(X_MIN, X_MAX, nx)
    t = np.linspace(T_MIN, T_MAX, nt)
    T, X = np.meshgrid(t, x)  # shapes (nx, nt)
    Y = np.full_like(X, float(y0))
    Z = np.full_like(X, float(z0))
    u_xt = exact_solution(X, Y, Z, T, alpha=alpha)
    return t, x, u_xt


