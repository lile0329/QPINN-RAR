import random
import time

import numpy as np
import torch
import torch.optim as optim
from scipy.optimize import fmin_l_bfgs_b

from src.nn.pde import heat_3d_operator


def get_random_minibatch(dataset_length, batch_size):
    if dataset_length <= 0:
        return []
    if batch_size <= dataset_length:
        return random.sample(range(dataset_length), batch_size)
    return random.choices(range(dataset_length), k=batch_size)


def _tensors_are_finite(*tensors):
    for tensor in tensors:
        if tensor is None:
            continue
        if not torch.isfinite(tensor).all():
            return False
    return True


def _losses_are_finite(losses):
    return _tensors_are_finite(*losses.values())


def _reduce_lr(optimizer, factor=0.5, min_lr=1e-6):
    if optimizer is None:
        return
    for group in optimizer.param_groups:
        group["lr"] = max(group["lr"] * factor, min_lr)


def _concat_eval(tx_dict, u_dict):
    keys = [
        "tx_domain",
        "tx_x0",
        "tx_x1",
        "tx_y0",
        "tx_y1",
        "tx_z0",
        "tx_z1",
        "tx_initial",
    ]
    tx_all = torch.cat([tx_dict[k] for k in keys if k in tx_dict], dim=0)
    u_all = torch.cat(
        [
            u_dict["u_domain"],
            u_dict["u_x0"],
            u_dict["u_x1"],
            u_dict["u_y0"],
            u_dict["u_y1"],
            u_dict["u_z0"],
            u_dict["u_z1"],
            u_dict["u_initial"],
        ],
        dim=0,
    )
    return tx_all, u_all


def compute_l2_relative_error(model, eval_data=None, max_eval_points=5000):
    try:
        model.eval()
        with torch.no_grad():
            if eval_data is not None:
                tx_eval, u_eval = eval_data
                tx_all, u_true_all = _concat_eval(tx_eval, u_eval)
            else:
                tx_all, u_true_all = _concat_eval(model.data[0], model.data[1])

            if tx_all.shape[0] > max_eval_points:
                indices = random.sample(range(tx_all.shape[0]), max_eval_points)
                tx_all = tx_all[indices]
                u_true_all = u_true_all[indices]

            chunk_size = 500
            u_pred_chunks = []
            for i in range(0, tx_all.shape[0], chunk_size):
                chunk = tx_all[i : i + chunk_size]
                u_pred_chunk = model.forward(chunk)
                u_pred_chunks.append(u_pred_chunk.cpu().numpy())

            u_pred = np.concatenate(u_pred_chunks, axis=0)
            u_true = u_true_all.cpu().numpy()

            numerator = np.linalg.norm(u_pred - u_true, 2)
            denominator = np.linalg.norm(u_true, 2)
            if denominator == 0:
                return float("inf")
            rel_l2_error = numerator / denominator

        model.train()
        return float(rel_l2_error)
    except Exception:
        model.train()
        return float("nan")


def compute_losses(model, tx_adaptive_points=None):
    alpha = float(model.args.get("alpha", 0.1))

    # --- Domain sampling (optionally mix adaptive points up to 30%)
    if tx_adaptive_points is not None and tx_adaptive_points.shape[0] > 0:
        n_adaptive = min(int(0.3 * model.batch_size), tx_adaptive_points.shape[0])
        n_original = max(0, model.batch_size - n_adaptive)

        idx = get_random_minibatch(model.data[0]["tx_domain"].shape[0], n_original)
        tx_domain = model.data[0]["tx_domain"][idx, :] if n_original > 0 else None

        if n_adaptive > 0:
            a_idx = get_random_minibatch(tx_adaptive_points.shape[0], n_adaptive)
            tx_adapt = tx_adaptive_points[a_idx, :]
            tx_domain = tx_adapt if tx_domain is None else torch.cat([tx_domain, tx_adapt], dim=0)
    else:
        idx = get_random_minibatch(model.data[0]["tx_domain"].shape[0], model.batch_size)
        tx_domain = model.data[0]["tx_domain"][idx, :]

    def _sample(key_tx, key_u):
        idx = get_random_minibatch(model.data[0][key_tx].shape[0], model.batch_size)
        tx = model.data[0][key_tx][idx, :]
        u = model.data[1][key_u][idx, :]
        return tx, u

    tx_x0, u_x0 = _sample("tx_x0", "u_x0")
    tx_x1, u_x1 = _sample("tx_x1", "u_x1")
    tx_y0, u_y0 = _sample("tx_y0", "u_y0")
    tx_y1, u_y1 = _sample("tx_y1", "u_y1")
    tx_z0, u_z0 = _sample("tx_z0", "u_z0")
    tx_z1, u_z1 = _sample("tx_z1", "u_z1")
    tx_initial, u_initial = _sample("tx_initial", "u_initial")

    # --- PDE residual on domain
    t_r = tx_domain[:, 0:1]
    x_r = tx_domain[:, 1:2]
    y_r = tx_domain[:, 2:3]
    z_r = tx_domain[:, 3:4]
    _, residual = heat_3d_operator(model, t_r, x_r, y_r, z_r, alpha=alpha)
    residual = torch.nan_to_num(residual, nan=0.0, posinf=1e6, neginf=-1e6)
    lphy = torch.mean(residual**2)

    # --- Boundary condition losses (Dirichlet)
    def _bc_loss(tx, u):
        pred = torch.nan_to_num(model.forward(tx), nan=0.0, posinf=1e6, neginf=-1e6)
        return model.loss_fn(pred, u)

    lbc_x0 = _bc_loss(tx_x0, u_x0)
    lbc_x1 = _bc_loss(tx_x1, u_x1)
    lbc_y0 = _bc_loss(tx_y0, u_y0)
    lbc_y1 = _bc_loss(tx_y1, u_y1)
    lbc_z0 = _bc_loss(tx_z0, u_z0)
    lbc_z1 = _bc_loss(tx_z1, u_z1)
    lbc = lbc_x0 + lbc_x1 + lbc_y0 + lbc_y1 + lbc_z0 + lbc_z1

    # --- Initial condition loss
    pred_initial = torch.nan_to_num(
        model.forward(tx_initial), nan=0.0, posinf=1e6, neginf=-1e6
    )
    lic = model.loss_fn(pred_initial, u_initial)

    return {
        "lphy": lphy,
        "lbc": lbc,
        "lic": lic,
        "lbc_x0": lbc_x0,
        "lbc_x1": lbc_x1,
        "lbc_y0": lbc_y0,
        "lbc_y1": lbc_y1,
        "lbc_z0": lbc_z0,
        "lbc_z1": lbc_z1,
    }


class LBFGSOptimizer:
    """Scipy L-BFGS-B optimizer (full-batch), following existing exact trainers' style."""

    def __init__(
        self,
        model,
        factr=1e3,
        m=100,
        maxls=50,
        maxiter=10000,
        pgtol=1e-14,
        eval_data=None,
        eval_every=100,
    ):
        self.model = model
        self.logger = getattr(model, "logger", None)
        self.factr = factr
        self.m = m
        self.maxls = maxls
        self.maxiter = maxiter
        self.pgtol = pgtol
        self.loss_fn = torch.nn.MSELoss()

        self.eval_data = eval_data
        self.eval_every = max(1, int(eval_every))
        self.l2_history = []
        self.l2_steps = []

        self.current_step = 0
        self.loss_history = []
        self.time_100_iterations = None
        self.start_time_100 = None

    def evaluate_loss_and_grad(self):
        alpha = float(self.model.args.get("alpha", 0.1))
        self.model.train()

        tx_domain = self.model.data[0]["tx_domain"]
        tx_x0 = self.model.data[0]["tx_x0"]
        tx_x1 = self.model.data[0]["tx_x1"]
        tx_y0 = self.model.data[0]["tx_y0"]
        tx_y1 = self.model.data[0]["tx_y1"]
        tx_z0 = self.model.data[0]["tx_z0"]
        tx_z1 = self.model.data[0]["tx_z1"]
        tx_initial = self.model.data[0]["tx_initial"]

        u_x0 = self.model.data[1]["u_x0"]
        u_x1 = self.model.data[1]["u_x1"]
        u_y0 = self.model.data[1]["u_y0"]
        u_y1 = self.model.data[1]["u_y1"]
        u_z0 = self.model.data[1]["u_z0"]
        u_z1 = self.model.data[1]["u_z1"]
        u_initial = self.model.data[1]["u_initial"]

        t_res = tx_domain[:, 0:1]
        x_res = tx_domain[:, 1:2]
        y_res = tx_domain[:, 2:3]
        z_res = tx_domain[:, 3:4]
        _, r_pred = heat_3d_operator(self.model, t_res, x_res, y_res, z_res, alpha=alpha)
        r_pred = torch.nan_to_num(r_pred, nan=0.0, posinf=1e6, neginf=-1e6)
        loss_eqn = torch.mean(r_pred**2)

        def _pred(tx):
            return torch.nan_to_num(
                self.model.forward(tx), nan=0.0, posinf=1e6, neginf=-1e6
            )

        loss_bc = (
            self.loss_fn(_pred(tx_x0), u_x0)
            + self.loss_fn(_pred(tx_x1), u_x1)
            + self.loss_fn(_pred(tx_y0), u_y0)
            + self.loss_fn(_pred(tx_y1), u_y1)
            + self.loss_fn(_pred(tx_z0), u_z0)
            + self.loss_fn(_pred(tx_z1), u_z1)
        )
        loss_ic = self.loss_fn(_pred(tx_initial), u_initial)

        total_loss = loss_eqn + loss_bc + loss_ic
        total_loss.backward()
        return total_loss

    def set_weights(self, flat_weights):
        idx = 0
        with torch.no_grad():
            for param in self.model.parameters():
                if not param.requires_grad:
                    continue
                param_size = param.numel()
                param.data = torch.tensor(
                    flat_weights[idx : idx + param_size],
                    dtype=param.dtype,
                    device=param.device,
                ).reshape(param.shape)
                idx += param_size

    def get_weights(self):
        params_list = [p for p in self.model.parameters() if p.requires_grad]
        if len(params_list) == 0:
            return np.array([], dtype="float64")
        weights = torch.cat([p.detach().flatten().cpu() for p in params_list]).numpy()
        return weights.astype("float64")

    def evaluate(self, weights):
        self.set_weights(weights)
        self.model.zero_grad()
        loss = self.evaluate_loss_and_grad()

        grads = []
        for param in self.model.parameters():
            if not param.requires_grad:
                continue
            if param.grad is not None:
                grads.append(param.grad.flatten().cpu().numpy())
            else:
                grads.append(np.zeros(param.numel(), dtype=np.float64))
        grads = np.concatenate(grads).astype("float64")
        return loss.item(), grads

    def callback(self, weights):
        self.current_step += 1
        loss, _ = self.evaluate(weights)
        self.loss_history.append(loss)

        rel_l2 = None
        should_eval_l2 = self.eval_data is not None and (
            self.current_step % self.eval_every == 0 or self.current_step == 1
        )
        should_print = self.current_step % 100 == 0 or self.current_step == 1
        should_compute_l2_for_print = should_print and self.eval_data is not None

        if should_eval_l2 or should_compute_l2_for_print:
            rel_l2 = compute_l2_relative_error(self.model, self.eval_data)
            if should_eval_l2:
                self.l2_history.append(rel_l2)
                self.l2_steps.append(self.current_step)

        if should_print:
            if self.current_step == 100 and self.start_time_100 is not None:
                self.time_100_iterations = time.time() - self.start_time_100
                msg = f"L-BFGS: Time for 100 iterations = {self.time_100_iterations:.4f} s"
                (self.logger.print(msg) if self.logger else print(msg))

            msg = f"L-BFGS Iteration: {self.current_step}, Loss: {loss:.6e}"
            if rel_l2 is not None:
                msg += f", Relative L2 Error: {rel_l2:.6e}"
            (self.logger.print(msg) if self.logger else print(msg))

    def fit(self):
        msg = (
            f"Optimizer: L-BFGS-B (maxiter={self.maxiter}, factr={self.factr}, "
            f"m={self.m}, maxls={self.maxls}, pgtol={self.pgtol})"
        )
        (self.logger.print(msg) if self.logger else print(msg))

        self.current_step = 0
        self.loss_history = []
        self.time_100_iterations = None
        self.start_time_100 = None

        initial_weights = self.get_weights()
        initial_loss, _ = self.evaluate(initial_weights)
        self.loss_history.append(initial_loss)

        msg = f"L-BFGS Iteration: {self.current_step}, Loss: {initial_loss:.6e}"
        (self.logger.print(msg) if self.logger else print(msg))

        self.start_time_100 = time.time()

        result = fmin_l_bfgs_b(
            func=self.evaluate,
            x0=initial_weights,
            factr=self.factr,
            m=self.m,
            maxls=self.maxls,
            maxiter=self.maxiter,
            pgtol=self.pgtol,
            callback=self.callback,
        )

        self.set_weights(result[0])
        return result


def train(
    model,
    use_rar=False,
    rar_eval_freq=2000,
    rar_threshold=0.0005,
    rar_num_candidates=100000,
    rar_num_add=50,
    use_hybrid_optimizer=False,
    adam_iterations=10000,
    eval_data=None,
):
    use_scipy_lbfgs = model.args.get("use_scipy_lbfgs", True)
    alpha = float(model.args.get("alpha", 0.1))

    tx_domain_original = model.data[0]["tx_domain"]
    dom_coords = torch.stack(
        [torch.min(tx_domain_original, dim=0)[0], torch.max(tx_domain_original, dim=0)[0]]
    ).to(model.device)

    tx_adaptive_points = None
    switched_to_lbfgs = False

    for it in range(model.epochs + 1):
        if use_hybrid_optimizer and it == adam_iterations and not switched_to_lbfgs:
            model.logger.print(f"\n=== Switching to LBFGS optimizer at iteration {it} ===")

            if use_scipy_lbfgs:
                model.logger.print("Using scipy L-BFGS-B optimizer")
                lbfgs_optimizer = LBFGSOptimizer(
                    model=model,
                    factr=model.args.get("lbfgs_factr", 1e3),
                    m=model.args.get("lbfgs_m", 100),
                    maxls=model.args.get("lbfgs_maxls", 50),
                    maxiter=model.args.get("lbfgs_maxiter", 10000),
                    pgtol=model.args.get("lbfgs_pgtol", 1e-14),
                    eval_data=eval_data,
                    eval_every=model.args.get("lbfgs_eval_every", 100),
                )
                lbfgs_optimizer.fit()

                if hasattr(lbfgs_optimizer, "loss_history"):
                    model.loss_history.extend(lbfgs_optimizer.loss_history)
                if hasattr(lbfgs_optimizer, "l2_history") and len(lbfgs_optimizer.l2_history) > 0:
                    if not hasattr(model, "l2_history"):
                        model.l2_history = []
                    if not hasattr(model, "l2_iters"):
                        model.l2_iters = []
                    model.l2_history.extend(lbfgs_optimizer.l2_history)
                    model.l2_iters.extend([it + step for step in lbfgs_optimizer.l2_steps])

                if hasattr(lbfgs_optimizer, "time_100_iterations"):
                    model.time_100_iterations = lbfgs_optimizer.time_100_iterations

                model.logger.print(f"L-BFGS training completed after {lbfgs_optimizer.current_step} iterations")
                switched_to_lbfgs = True
                break

            lbfgs_lr = model.args.get("lbfgs_lr", 1)
            lbfgs_line_search = model.args.get("lbfgs_line_search", "strong_wolfe")
            lbfgs_tolerance_grad = model.args.get("lbfgs_tolerance_grad", 1e-5)
            lbfgs_tolerance_change = model.args.get("lbfgs_tolerance_change", 1e-7)
            model.optimizer = optim.LBFGS(
                filter(lambda p: p.requires_grad, model.parameters()),
                lr=lbfgs_lr,
                max_iter=20,
                max_eval=None,
                tolerance_grad=lbfgs_tolerance_grad,
                tolerance_change=lbfgs_tolerance_change,
                history_size=100,
                line_search_fn=lbfgs_line_search,
            )
            model.scheduler = None
            switched_to_lbfgs = True
            model.logger.print("PyTorch LBFGS optimizer initialized\n")

        time_start = time.time()

        if use_hybrid_optimizer and switched_to_lbfgs and (not use_scipy_lbfgs):
            closure_nonfinite = False

            def closure():
                nonlocal closure_nonfinite
                if model.optimizer is not None:
                    model.optimizer.zero_grad()
                losses_closure = compute_losses(model, tx_adaptive_points)
                if not _losses_are_finite(losses_closure):
                    closure_nonfinite = True
                    loss_closure = torch.tensor(0.0, device=model.device, requires_grad=True)
                    loss_closure.backward()
                    return loss_closure

                loss_closure = losses_closure["lphy"] + losses_closure["lbc"] + losses_closure["lic"]
                loss_closure.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
                return loss_closure

            loss = model.optimizer.step(closure) if model.optimizer is not None else None

            with torch.enable_grad():
                losses_log = compute_losses(model, tx_adaptive_points)
                if not _losses_are_finite(losses_log) or closure_nonfinite:
                    model.logger.print(f"WARNING: Non-finite loss detected with LBFGS at iteration {it}. Continuing.")
                    loss_res = torch.tensor(float("nan"), device=model.device)
                    loss_bc = torch.tensor(float("nan"), device=model.device)
                    loss_ic = torch.tensor(float("nan"), device=model.device)
                    loss = torch.tensor(float("nan"), device=model.device)
                else:
                    loss_res = losses_log["lphy"]
                    loss_bc = losses_log["lbc"]
                    loss_ic = losses_log["lic"]
                    if loss is None or torch.isnan(loss) or torch.isinf(loss):
                        loss = loss_res + loss_bc + loss_ic
        else:
            if model.optimizer is not None:
                model.optimizer.zero_grad()
            losses = compute_losses(model, tx_adaptive_points)
            if not _losses_are_finite(losses):
                model.logger.print(
                    f"WARNING: Non-finite loss detected at iteration {it}. Skipping optimizer step and reducing LR."
                )
                _reduce_lr(model.optimizer, factor=0.5, min_lr=1e-6)
                model.loss_history.append(float("nan"))
                continue

            loss_res = losses["lphy"]
            loss_bc = losses["lbc"]
            loss_ic = losses["lic"]
            loss = loss_res + loss_bc + loss_ic

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            if model.optimizer is not None:
                model.optimizer.step()
            if model.scheduler is not None:
                model.scheduler.step(loss)

        time_taken = time.time() - time_start

        if it % model.args["print_every"] == 0:
            rel_l2_error = compute_l2_relative_error(model, eval_data)
            model.logger.print(
                "Iteration: %d, loss_r = %.1e ,  loss_bc = %.1e,  loss_ic = %.1e,  lr = %0.1e, time_taken = %.1e, rel_l2_error = %.1e"
                % (
                    it,
                    loss_res.item(),
                    loss_bc.item(),
                    loss_ic.item(),
                    model.optimizer.param_groups[0]["lr"] if model.optimizer else 0.0,
                    time_taken,
                    rel_l2_error,
                )
            )
            model.save_state()

        model.loss_history.append(loss.item() if hasattr(loss, "item") else float(loss))

        # --- RAR: residual-based adaptive refinement
        if use_rar and it > 0 and it % rar_eval_freq == 0:
            model.logger.print(f"\n=== RAR Evaluation at iteration {it} ===")

            rand_vals = torch.rand(rar_num_candidates, 4, device=model.device)
            X_candidates = dom_coords[0:1, :] + (dom_coords[1:2, :] - dom_coords[0:1, :]) * rand_vals

            model.eval()
            with torch.enable_grad():
                t_c = X_candidates[:, 0:1]
                x_c = X_candidates[:, 1:2]
                y_c = X_candidates[:, 2:3]
                z_c = X_candidates[:, 3:4]
                _, r_pred = heat_3d_operator(model, t_c, x_c, y_c, z_c, alpha=alpha)
                err_eq = torch.abs(r_pred).squeeze()
                mean_residual = torch.mean(err_eq).item()
                model.logger.print(f"Mean residual: {mean_residual:.3e}")

            model.train()

            if mean_residual > rar_threshold:
                _, top_indices = torch.topk(err_eq, rar_num_add)
                X_new = X_candidates[top_indices]
                tx_adaptive_points = X_new if tx_adaptive_points is None else torch.cat([tx_adaptive_points, X_new], dim=0)

                max_adaptive_points = 10 * model.data[0]["tx_domain"].shape[0]
                if tx_adaptive_points.shape[0] > max_adaptive_points:
                    tx_adaptive_points = tx_adaptive_points[-max_adaptive_points:, :]
                    model.logger.print(f"Adaptive points limited to {max_adaptive_points} (keeping most recent)\n")

                model.logger.print(f"Added {rar_num_add} point(s). Total adaptive points: {tx_adaptive_points.shape[0]}\n")
            else:
                model.logger.print(
                    f"Mean residual ({mean_residual:.3e}) below threshold ({rar_threshold:.3e}). No points added.\n"
                )


