import random
import time
import torch
import torch.optim as optim
import numpy as np
from scipy.optimize import fmin_l_bfgs_b
from src.nn.pde import diffusion_1d_operator





def get_random_minibatch(dataset_length, batch_size):
    if batch_size <= dataset_length:
        batch_indices = random.sample(range(dataset_length), batch_size)
    else:
        batch_indices = random.choices(range(dataset_length), k=batch_size)
    return batch_indices


def _tensors_are_finite(*tensors):
    for tensor in tensors:
        if tensor is None:
            continue
        if not torch.isfinite(tensor).all():
            return False
    return True


def _losses_are_finite(losses):
    return _tensors_are_finite(*losses.values())


def _params_are_finite(model):
    for param in model.parameters():
        if param is None:
            continue
        if not torch.isfinite(param).all():
            return False
    return True


def _reduce_lr(optimizer, factor=0.5, min_lr=1e-6):
    if optimizer is None:
        return
    for group in optimizer.param_groups:
        group["lr"] = max(group["lr"] * factor, min_lr)


def compute_l2_relative_error(model, eval_data=None, max_eval_points=5000):
    try:
        model.eval()
        with torch.no_grad():
            if eval_data is not None:
                tx_eval, u_eval = eval_data
                tx_all = torch.cat([
                    tx_eval["tx_domain"],
                    tx_eval["tx_left"],
                    tx_eval["tx_right"],
                    tx_eval["tx_initial"]
                ], dim=0)
                u_true_all = torch.cat([
                    u_eval["u_domain"],
                    u_eval["u_left"],
                    u_eval["u_right"],
                    u_eval["u_initial"]
                ], dim=0)
            else:
                tx_all = torch.cat([
                    model.data[0]["tx_domain"],
                    model.data[0]["tx_left"],
                    model.data[0]["tx_right"],
                    model.data[0]["tx_initial"]
                ], dim=0)
                u_true_all = torch.cat([
                    model.data[1]["u_domain"],
                    model.data[1]["u_left"],
                    model.data[1]["u_right"],
                    model.data[1]["u_initial"]
                ], dim=0)

            if tx_all.shape[0] > max_eval_points:
                indices = random.sample(range(tx_all.shape[0]), max_eval_points)
                tx_all = tx_all[indices]
                u_true_all = u_true_all[indices]

            chunk_size = 500
            u_pred_chunks = []
            for i in range(0, tx_all.shape[0], chunk_size):
                chunk = tx_all[i:i+chunk_size]
                u_pred_chunk = model.forward(chunk)
                u_pred_chunks.append(u_pred_chunk.cpu().numpy())

            u_pred = np.concatenate(u_pred_chunks, axis=0)
            u_true = u_true_all.cpu().numpy()

            numerator = np.linalg.norm(u_pred - u_true, 2)
            denominator = np.linalg.norm(u_true, 2)
            if denominator == 0:
                return float('inf')
            rel_l2_error = (numerator / denominator)

        model.train()
        return rel_l2_error
    except Exception:
        model.train()
        return float('nan')


def compute_losses(model, tx_adaptive_points=None):
    if tx_adaptive_points is not None and tx_adaptive_points.shape[0] > 0:
        n_adaptive = min(int(0.3 * model.batch_size), tx_adaptive_points.shape[0])
        n_original = max(0, model.batch_size - n_adaptive)

        batch_indices = get_random_minibatch(
            model.data[0]["tx_domain"].shape[0], n_original
        )
        tx_domain = model.data[0]["tx_domain"][batch_indices, :]

        if n_adaptive > 0:
            n_adaptive_available = tx_adaptive_points.shape[0]
            if n_adaptive <= n_adaptive_available:
                adaptive_indices = random.sample(range(n_adaptive_available), n_adaptive)
            else:
                adaptive_indices = random.choices(range(n_adaptive_available), k=n_adaptive)
            tx_adaptive_batch = tx_adaptive_points[adaptive_indices, :]

            if n_original > 0:
                tx_domain = torch.cat([tx_domain, tx_adaptive_batch], dim=0)
            else:
                tx_domain = tx_adaptive_batch
    else:
        batch_indices = get_random_minibatch(
            model.data[0]["tx_domain"].shape[0], model.batch_size
        )
        tx_domain = model.data[0]["tx_domain"][batch_indices, :]

    batch_indices = get_random_minibatch(
        model.data[0]["tx_left"].shape[0], model.batch_size
    )
    tx_left = model.data[0]["tx_left"][batch_indices, :]
    u_left = model.data[1]["u_left"][batch_indices, :]

    batch_indices = get_random_minibatch(
        model.data[0]["tx_right"].shape[0], model.batch_size
    )
    tx_right = model.data[0]["tx_right"][batch_indices, :]
    u_right = model.data[1]["u_right"][batch_indices, :]

    batch_indices = get_random_minibatch(
        model.data[0]["tx_initial"].shape[0], model.batch_size
    )
    tx_initial = model.data[0]["tx_initial"][batch_indices, :]
    u_initial = model.data[1]["u_initial"][batch_indices, :]

    t_r, x_r = tx_domain[:, 0:1], tx_domain[:, 1:2]
    _, residual = diffusion_1d_operator(model, t_r, x_r, nu=1.0 / (torch.pi ** 2))
    residual = torch.nan_to_num(residual, nan=0.0, posinf=1e6, neginf=-1e6)
    lphy = torch.mean(residual**2)

    pred_left = torch.nan_to_num(
        model.forward(tx_left), nan=0.0, posinf=1e6, neginf=-1e6
    )
    lleft = model.loss_fn(pred_left, u_left)

    pred_right = torch.nan_to_num(
        model.forward(tx_right), nan=0.0, posinf=1e6, neginf=-1e6
    )
    lright = model.loss_fn(pred_right, u_right)

    pred_initial = torch.nan_to_num(
        model.forward(tx_initial), nan=0.0, posinf=1e6, neginf=-1e6
    )
    linitial = model.loss_fn(pred_initial, u_initial)

    return {
        "lleft": lleft,
        "lright": lright,
        "linitial": linitial,
        "lphy": lphy,
    }


class LBFGSOptimizer:
    def __init__(
        self,
        model,
        factr=1e3,
        m=1000,
        maxls=200,
        maxiter=5000,
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
        self.eval_every = max(1, eval_every)
        self.l2_history = []
        self.l2_steps = []

        self.current_step = 0
        self.loss_history = []
        self.time_100_iterations = None
        self.start_time_100 = None

    def evaluate_loss_and_grad(self):
        self.model.train()

        tx_domain = self.model.data[0]["tx_domain"]
        tx_left = self.model.data[0]["tx_left"]
        tx_right = self.model.data[0]["tx_right"]
        tx_initial = self.model.data[0]["tx_initial"]

        u_left = self.model.data[1]["u_left"]
        u_right = self.model.data[1]["u_right"]
        u_initial = self.model.data[1]["u_initial"]

        t_res = tx_domain[:, 0:1]
        x_res = tx_domain[:, 1:2]
        _, r_pred = diffusion_1d_operator(self.model, t_res, x_res, nu=1.0 / (torch.pi ** 2))
        r_pred = torch.nan_to_num(r_pred, nan=0.0, posinf=1e6, neginf=-1e6)
        loss_eqn = torch.mean(r_pred**2)

        pred_left = torch.nan_to_num(
            self.model.forward(tx_left), nan=0.0, posinf=1e6, neginf=-1e6
        )
        loss_left = self.loss_fn(pred_left, u_left)

        pred_right = torch.nan_to_num(
            self.model.forward(tx_right), nan=0.0, posinf=1e6, neginf=-1e6
        )
        loss_right = self.loss_fn(pred_right, u_right)

        pred_initial = torch.nan_to_num(
            self.model.forward(tx_initial), nan=0.0, posinf=1e6, neginf=-1e6
        )
        loss_initial = self.loss_fn(pred_initial, u_initial)

        total_loss = loss_eqn + loss_left + loss_right + loss_initial

        total_loss.backward()

        return total_loss

    def set_weights(self, flat_weights):
        idx = 0
        with torch.no_grad():
            for param in self.model.parameters():
                if not param.requires_grad:
                    continue
                param_shape = param.shape
                param_size = param.numel()
                param.data = torch.tensor(
                    flat_weights[idx:idx + param_size],
                    dtype=param.dtype,
                    device=param.device
                ).reshape(param_shape)
                idx += param_size

    def get_weights(self):
        params_list = [p for p in self.model.parameters() if p.requires_grad]
        if len(params_list) == 0:
            return np.array([], dtype='float64')

        target_device = params_list[0].device
        if torch.cuda.is_available() and target_device.type == 'cuda':
            weights = torch.cat([param.flatten().to(target_device) for param in params_list]).detach().cpu().numpy()
        else:
            weights = torch.cat([param.flatten().cpu() for param in params_list]).detach().numpy()
        return weights.astype('float64')

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

        grads = np.concatenate(grads).astype('float64')

        return loss.item(), grads

    def callback(self, weights):
        self.current_step += 1
        loss, _ = self.evaluate(weights)

        self.loss_history.append(loss)

        rel_l2 = None
        should_eval_l2 = (
            self.eval_data is not None
            and (self.current_step % self.eval_every == 0 or self.current_step == 1)
        )

        should_print = (self.current_step % 100 == 0 or self.current_step == 1)
        should_compute_l2_for_print = should_print and self.eval_data is not None

        if should_eval_l2 or should_compute_l2_for_print:
            rel_l2 = compute_l2_relative_error(self.model, self.eval_data)

            if should_eval_l2:
                self.l2_history.append(rel_l2)
                self.l2_steps.append(self.current_step)

        if should_print:
            if self.current_step == 100 and self.start_time_100 is not None:
                self.time_100_iterations = time.time() - self.start_time_100
                time_msg = f"L-BFGS: Time for 100 iterations = {self.time_100_iterations:.4f} s"
                if self.logger:
                    self.logger.print(time_msg)
                else:
                    print(time_msg)

            if rel_l2 is not None:
                msg = f"L-BFGS Iteration: {self.current_step}, Loss: {loss:.6e}, Relative L2 Error: {rel_l2:.6e}"
            else:
                msg = f"L-BFGS Iteration: {self.current_step}, Loss: {loss:.6e}"

            if self.logger:
                self.logger.print(msg)
            else:
                print(msg)

    def fit(self):
        msg = f'L-BFGS-B optimizer (maxiter={self.maxiter}, factr={self.factr})'
        if self.logger:
            self.logger.print(msg)
        else:
            print(msg)

        self.current_step = 0
        self.loss_history = []
        self.time_100_iterations = None
        self.start_time_100 = None

        initial_weights = self.get_weights()

        initial_loss, _ = self.evaluate(initial_weights)
        self.loss_history.append(initial_loss)
        init_msg = f"L-BFGS Iteration: {self.current_step}, Loss: {initial_loss:.6e}"
        if self.logger:
            self.logger.print(init_msg)
        else:
            print(init_msg)

        self.start_time_100 = time.time()

        result = fmin_l_bfgs_b(
            func=self.evaluate,
            x0=initial_weights,
            factr=self.factr,
            m=self.m,
            maxls=self.maxls,
            maxiter=self.maxiter,
            pgtol=self.pgtol,
            callback=self.callback
        )

        self.set_weights(result[0])

        final_loss, _ = self.evaluate(result[0])
        if len(self.loss_history) == 0 or self.loss_history[-1] != final_loss:
            self.loss_history.append(final_loss)

        finish_msg = f"L-BFGS optimization finished after {self.current_step} iterations."
        if self.logger:
            self.logger.print(finish_msg)
        else:
            print(finish_msg)

        return result


def train(model, use_rar=False, rar_eval_freq=2000, rar_threshold=0.0005, rar_num_candidates=100000, rar_num_add=50, 
          use_hybrid_optimizer=False, adam_iterations=10000, eval_data=None):
    use_scipy_lbfgs = model.args.get("use_scipy_lbfgs", True)

    tx_domain_original = model.data[0]["tx_domain"]
    dom_coords = torch.stack([
        torch.min(tx_domain_original, dim=0)[0],
        torch.max(tx_domain_original, dim=0)[0]
    ]).to(model.device)

    tx_adaptive_points = None
    switched_to_lbfgs = False

    for it in range(model.epochs + 1):
        if use_hybrid_optimizer and it == adam_iterations and not switched_to_lbfgs:
            model.logger.print(f"\n=== Switching to LBFGS optimizer at iteration {it} ===")

            if use_scipy_lbfgs:
                model.logger.print("Using scipy L-BFGS-B optimizer")
                lbfgs_maxiter = model.args.get("lbfgs_maxiter", 5000)
                lbfgs_factr = model.args.get("lbfgs_factr", 1e3)
                lbfgs_maxls = model.args.get("lbfgs_maxls", 200)
                lbfgs_m = model.args.get("lbfgs_m", 1000)
                lbfgs_pgtol = model.args.get("lbfgs_pgtol", 1e-14)
                lbfgs_eval_every = model.args.get("lbfgs_eval_every", 100)

                lbfgs_optimizer = LBFGSOptimizer(
                    model=model,
                    factr=lbfgs_factr,
                    m=lbfgs_m,
                    maxls=lbfgs_maxls,
                    maxiter=lbfgs_maxiter,
                    pgtol=lbfgs_pgtol,
                    eval_data=eval_data,
                    eval_every=lbfgs_eval_every,
                )

                lbfgs_optimizer.fit()

                if hasattr(lbfgs_optimizer, 'loss_history'):
                    model.loss_history.extend(lbfgs_optimizer.loss_history)

                if hasattr(lbfgs_optimizer, 'l2_history') and len(lbfgs_optimizer.l2_history) > 0:
                    if not hasattr(model, 'l2_history'):
                        model.l2_history = []
                    if not hasattr(model, 'l2_iters'):
                        model.l2_iters = []
                    model.l2_history.extend(lbfgs_optimizer.l2_history)
                    model.l2_iters.extend([it + step for step in lbfgs_optimizer.l2_steps])

                if hasattr(lbfgs_optimizer, 'time_100_iterations'):
                    model.time_100_iterations = lbfgs_optimizer.time_100_iterations

                model.logger.print(f"L-BFGS training completed after {lbfgs_optimizer.current_step} iterations")
                switched_to_lbfgs = True
                break
            else:
                lbfgs_lr = model.args.get("lbfgs_lr", 1)
                lbfgs_line_search = model.args.get("lbfgs_line_search", "strong_wolfe")
                lbfgs_tolerance_grad = model.args.get("lbfgs_tolerance_grad", 1e-07)
                lbfgs_tolerance_change = model.args.get("lbfgs_tolerance_change", 1e-09)
                model.optimizer = optim.LBFGS(
                    filter(lambda p: p.requires_grad, model.parameters()),
                    lr=lbfgs_lr,
                    max_iter=20,
                    max_eval=None,
                    tolerance_grad=lbfgs_tolerance_grad,
                    tolerance_change=lbfgs_tolerance_change,
                    history_size=100,
                    line_search_fn=lbfgs_line_search
                )
                model.scheduler = None
                switched_to_lbfgs = True
                model.logger.print("PyTorch LBFGS optimizer initialized\n")

        time_start = time.time()

        if use_hybrid_optimizer and switched_to_lbfgs:
            closure_nonfinite = False

            def closure():
                nonlocal closure_nonfinite
                if model.optimizer is not None:
                    model.optimizer.zero_grad()
                losses_closure = compute_losses(model, tx_adaptive_points)
                if not _losses_are_finite(losses_closure):
                    closure_nonfinite = True
                    loss_closure = torch.tensor(
                        0.0, device=model.device, requires_grad=True
                    )
                    loss_closure.backward()
                    return loss_closure

                loss_factors = {
                    "lleft": 1.0,
                    "lright": 1.0,
                    "linitial": 1.0,
                    "lphy": 1,
                }
                loss_closure = sum(factor * losses_closure[key] for key, factor in loss_factors.items())

                loss_closure.backward()

                if model.args["solver"] == "CV":
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.1)
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)

                return loss_closure

            if model.optimizer is not None:
                loss = model.optimizer.step(closure)

            with torch.enable_grad():
                losses_log = compute_losses(model, tx_adaptive_points)
                if not _losses_are_finite(losses_log) or closure_nonfinite:
                    model.logger.print(
                        f"WARNING: Non-finite loss detected with LBFGS at iteration {it}. Continuing."
                    )
                else:
                    loss_bc = losses_log["lleft"] + losses_log["lright"]
                    loss_ic = losses_log["linitial"]
                    loss_res = losses_log["lphy"]

                    if loss is None or torch.isnan(loss) or torch.isinf(loss):
                        loss = sum(factor * losses_log[key] for key, factor in {
                            "lleft": 1.0,
                            "lright": 1.0,
                            "linitial": 1.0,
                            "lphy": 1,
                        }.items())
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

            loss_bc = losses["lleft"] + losses["lright"]
            loss_ic = losses["linitial"]
            loss_res = losses["lphy"]

            loss_factors = {
                "lleft": 1.0,
                "lright": 1.0,
                "linitial": 1.0,
                "lphy": 1,
            }
            loss = sum(factor * losses[key] for key, factor in loss_factors.items())

            loss.backward()
            if model.args["solver"] == "CV":
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.1)
            else:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1)

            if model.optimizer is not None:
                model.optimizer.step()
            if model.scheduler is not None:
                model.scheduler.step(loss)

        time_end = time.time()
        time_taken = time_end - time_start
        if it % model.args["print_every"] == 0:
            rel_l2_error = compute_l2_relative_error(model, eval_data)
            model.logger.print(
                "Iteration: %d, loss_r = %.1e ,  loss_bc = %.1e,  loss_ic = %.1e,  lr = %0.1e, time_taken = %.1e, rel_l2_error = %.1e"
                % (
                    it,
                    loss_res.item() if 'loss_res' in locals() else torch.tensor(float('nan')),
                    loss_bc.item() if 'loss_bc' in locals() else torch.tensor(float('nan')),
                    loss_ic.item() if 'loss_ic' in locals() else torch.tensor(float('nan')),
                    model.optimizer.param_groups[0]["lr"] if model.optimizer else 0.0,
                    time_taken,
                    rel_l2_error,
                )
            )
            model.save_state()

        model.loss_history.append(loss.item())

        if use_rar and it > 0 and it % rar_eval_freq == 0:
            model.logger.print(f"\n=== RAR Evaluation at iteration {it} ===")

            rand_vals = torch.rand(rar_num_candidates, 2, device=model.device)
            X_candidates = (
                dom_coords[0:1, :]
                + (dom_coords[1:2, :] - dom_coords[0:1, :]) * rand_vals
            )

            model.eval()
            with torch.enable_grad():
                t_cand, x_cand = X_candidates[:, 0:1], X_candidates[:, 1:2]
                _, r_pred = diffusion_1d_operator(model, t_cand, x_cand, nu=1.0 / (torch.pi ** 2))
                err_eq = torch.abs(r_pred).squeeze()
                mean_residual = torch.mean(err_eq).item()
                model.logger.print(f"Mean residual: {mean_residual:.3e}")

            model.train()

            if mean_residual > rar_threshold:
                _, top_indices = torch.topk(err_eq, rar_num_add)
                X_new = X_candidates[top_indices]

                model.logger.print(f"Adding {rar_num_add} new point(s) with highest residuals:")
                if tx_adaptive_points is None:
                    tx_adaptive_points = X_new
                else:
                    tx_adaptive_points = torch.cat([tx_adaptive_points, X_new], dim=0)

                max_adaptive_points = 10 * model.data[0]["tx_domain"].shape[0]
                if tx_adaptive_points.shape[0] > max_adaptive_points:
                    tx_adaptive_points = tx_adaptive_points[-max_adaptive_points:, :]
                    model.logger.print(f"Adaptive points limited to {max_adaptive_points} (keeping most recent)\n")

                model.logger.print(f"Total adaptive points: {tx_adaptive_points.shape[0]}\n")
            else:
                model.logger.print(f"Mean residual ({mean_residual:.3e}) below threshold ({rar_threshold:.3e}). No points added.\n")
