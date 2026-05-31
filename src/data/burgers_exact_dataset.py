import numpy as np
import torch

# Parameters from the equation
NU = 0.01  # viscosity
X_MIN, X_MAX = -1.0, 1.0  # spatial domain
T_MIN, T_MAX = 0.0, 1.0  # time domain
T0 = np.exp(1.0 / (8.0 * NU))  # t_0 = e^{1/(8\nu)}


def exact_solution(x, t):
    """
    Exact solution for Burgers equation:
    u(x, t) = (x/(t+1)) / (1 + sqrt((t+1)/t_0) * exp(x^2/(4*nu*(t+1))))
    
    Args:
        x: spatial coordinate (can be array)
        t: time coordinate (can be array)
    
    Returns:
        u: solution value(s)
    """
    # Handle scalar and array inputs
    x = np.asarray(x)
    t = np.asarray(t)
    
    # Avoid division by zero
    t_safe = t + 1.0
    
    numerator = x / t_safe
    denominator = 1.0 + np.sqrt(t_safe / T0) * np.exp(x**2 / (4.0 * NU * t_safe))
    
    u = numerator / denominator
    
    return u


def generate_sobol_sequence(low, high, n):
    soboleng = torch.quasirandom.SobolEngine(dimension=1)
    bounds = [low, high]
    input_tb = soboleng.draw(n)
    result = np.floor((bounds[0] + (bounds[1] - bounds[0]) * input_tb))
    result = [int(i) for i in result]
    return result


def generate_training_data(nx=200, nt=200, domainN=1000, leftN=50, rightN=50, initialN=50, dist="random"):
    """
    Generate training data for Burgers equation with exact solution.
    
    This version always uses random sampling for the training points.  The
    ``dist`` argument is accepted for compatibility but is ignored – the code
    will never use a Sobol sequence even if ``dist="Sobol"`` is passed.
    
    Args:
        nx: number of spatial points
        nt: number of time points
        domainN: fixed number of domain points to use
        leftN: fixed number of left boundary points to use
        rightN: fixed number of right boundary points to use
        initialN: fixed number of initial condition points to use
        dist: distribution type (ignored, always random)
    
    Returns:
        ([domain, left, right, initial], [domain_eval, left_eval, right_eval, initial_eval]): 
        training data arrays and evaluation data arrays
    """
    np.random.seed(42)
    fixed_counts = [domainN, leftN, rightN, initialN]
    
    # Generate grid
    x = np.linspace(X_MIN, X_MAX, nx)
    t = np.linspace(T_MIN, T_MAX, nt)
    T, X = np.meshgrid(t, x)
    
    # Compute exact solution on grid
    u_xt = exact_solution(X, T)
    
    # Domain points: (t, x, u)
    domain = np.concatenate(
        [T.reshape(-1, 1), X.reshape(-1, 1), u_xt.reshape(-1, 1)], axis=1
    )
    
    # Left boundary: x = x_min for all t
    u_left = exact_solution(X_MIN, t)
    left = np.concatenate(
        [
            t.reshape(-1, 1),
            np.full((t.size, 1), X_MIN),
            u_left.reshape(-1, 1),
        ],
        axis=1,
    )
    
    # Right boundary: x = x_max for all t
    u_right = exact_solution(X_MAX, t)
    right = np.concatenate(
        [
            t.reshape(-1, 1),
            np.full((t.size, 1), X_MAX),
            u_right.reshape(-1, 1),
        ],
        axis=1,
    )
    
    # Initial condition: t = t_min for all x
    u_initial = exact_solution(x, T_MIN)
    initial = np.concatenate(
        [
            np.full((x.size, 1), T_MIN),
            x.reshape(-1, 1),
            u_initial.reshape(-1, 1),
        ],
        axis=1,
    )
    
    # Sample points according to fixed counts
    data = [domain, left, right, initial]
    training_dataset = []
    eval_dataset = []
    
    # 计算并输出采样点数
    domain_total = domain.shape[0]
    left_total = left.shape[0]
    right_total = right.shape[0]
    initial_total = initial.shape[0]
    
    # 确保采样点数不超过总点数
    domain_samples = min(domainN, domain_total)
    left_samples = min(leftN, left_total)
    right_samples = min(rightN, right_total)
    initial_samples = min(initialN, initial_total)
    
    domain_eval_samples = domain_total - domain_samples
    left_eval_samples = left_total - left_samples
    right_eval_samples = right_total - right_samples
    initial_eval_samples = initial_total - initial_samples
    
    print(f"\n=== 采样点数统计 (Exact Dataset) ===")
    print(f"原始数据维度: nx={nx}, nt={nt}")
    print(f"原始点数:")
    print(f"  domain: {domain_total} (nx * nt)")
    print(f"  left: {left_total} (nt)")
    print(f"  right: {right_total} (nt)")
    print(f"  initial: {initial_total} (nx)")
    print(f"\n固定采样点数:")
    print(f"  domainN = {domainN}")
    print(f"  leftN = {leftN}")
    print(f"  rightN = {rightN}")
    print(f"  initialN = {initialN}")
    print(f"\n实际训练采样点数:")
    print(f"  domain: {domain_samples}")
    print(f"  left: {left_samples}")
    print(f"  right: {right_samples}")
    print(f"  initial: {initial_samples}")
    print(f"总训练采样点数: {domain_samples + left_samples + right_samples + initial_samples}")
    print(f"\n剩余评估点数:")
    print(f"  domain: {domain_eval_samples}")
    print(f"  left: {left_eval_samples}")
    print(f"  right: {right_eval_samples}")
    print(f"  initial: {initial_eval_samples}")
    print(f"总剩余评估点数: {domain_eval_samples + left_eval_samples + right_eval_samples + initial_eval_samples}")
    print(f"====================================\n")
    
    for d, count in zip(data, fixed_counts):
        # 确保采样点数不超过总点数
        actual_count = min(count, d.shape[0])
        if actual_count <= 0:
            # 如果没有点可采样，创建空数组
            training_dataset.append(np.empty((0, 3)))
            eval_dataset.append(d)
        else:
            # always perform random sampling; ignore the ``dist`` parameter
            idxi = np.random.choice(
                d.shape[0], actual_count, replace=False
            )
            training_dataset.append(d[idxi, :])
            # 获取剩下的点用于评估
            eval_idxi = np.setdiff1d(np.arange(d.shape[0]), idxi)
            eval_dataset.append(d[eval_idxi, :])
    
    domain = training_dataset[0]
    left = training_dataset[1]
    right = training_dataset[2]
    initial = training_dataset[3]
    
    # Sort by time (first column), handle empty arrays
    if domain.shape[0] > 0:
        domain = domain[np.argsort(domain[:, 0])]
    if left.shape[0] > 0:
        left = left[np.argsort(left[:, 0])]
    if right.shape[0] > 0:
        right = right[np.argsort(right[:, 0])]
    if initial.shape[0] > 0:
        initial = initial[np.argsort(initial[:, 0])]
    
    domain_eval = eval_dataset[0]
    left_eval = eval_dataset[1]
    right_eval = eval_dataset[2]
    initial_eval = eval_dataset[3]
    
    if domain_eval.shape[0] > 0:
        domain_eval = domain_eval[np.argsort(domain_eval[:, 0])]
    if left_eval.shape[0] > 0:
        left_eval = left_eval[np.argsort(left_eval[:, 0])]
    if right_eval.shape[0] > 0:
        right_eval = right_eval[np.argsort(right_eval[:, 0])]
    if initial_eval.shape[0] > 0:
        initial_eval = initial_eval[np.argsort(initial_eval[:, 0])]
    
    return [domain, left, right, initial], [domain_eval, left_eval, right_eval, initial_eval]


class BurgersExactDataset(object):
    def __init__(
        self,
        nx=256,
        nt=100,
        domainN=1000,
        leftN=50,
        rightN=50,
        initialN=50,
        dist="random",
        device=None,
        eval_total_points=None,
    ):
        [domain, left, right, initial], [domain_eval, left_eval, right_eval, initial_eval] = generate_training_data(
            nx, nt, domainN, leftN, rightN, initialN, dist
        )
        
        # 如果指定了评估点总数，则对"剩余的点"进行按比例下采样或从整个网格采样
        # eval_total_points 为 None 时，默认仍然使用全部剩余点做评估
        if eval_total_points is not None:
            eval_arrays = [domain_eval, left_eval, right_eval, initial_eval]
            total_eval_points = sum(arr.shape[0] for arr in eval_arrays)
            
            if eval_total_points <= 0:
                raise ValueError("eval_total_points 必须为正数（或设置为 None 表示不下采样）")
            
            # 记录所有训练点的坐标（用于排除）
            train_tx = np.concatenate([
                domain[:, 0:2],
                left[:, 0:2],
                right[:, 0:2],
                initial[:, 0:2]
            ], axis=0)
            
            # 如果剩余点少于需要的测试点数，从整个网格中采样，但排除训练点
            if eval_total_points > total_eval_points:
                # 生成整个网格
                x = np.linspace(X_MIN, X_MAX, nx)
                t = np.linspace(T_MIN, T_MAX, nt)
                T, X = np.meshgrid(t, x)
                u_xt = exact_solution(X, T)
                
                # 所有网格点
                all_domain = np.concatenate(
                    [T.reshape(-1, 1), X.reshape(-1, 1), u_xt.reshape(-1, 1)], axis=1
                )
                all_left = np.concatenate([
                    t.reshape(-1, 1),
                    np.full((t.size, 1), X_MIN),
                    exact_solution(X_MIN, t).reshape(-1, 1),
                ], axis=1)
                all_right = np.concatenate([
                    t.reshape(-1, 1),
                    np.full((t.size, 1), X_MAX),
                    exact_solution(X_MAX, t).reshape(-1, 1),
                ], axis=1)
                all_initial = np.concatenate([
                    np.full((x.size, 1), T_MIN),
                    x.reshape(-1, 1),
                    exact_solution(x, T_MIN).reshape(-1, 1),
                ], axis=1)
                
                # 合并所有点
                all_points = np.concatenate([all_domain, all_left, all_right, all_initial], axis=0)
                all_tx = all_points[:, 0:2]
                
                # 找到不在训练集中的点（使用坐标匹配）
                # 将坐标转换为元组集合以便快速查找
                train_tx_set = set(tuple(row) for row in train_tx)
                candidate_indices = []
                for i, tx_row in enumerate(all_tx):
                    if tuple(tx_row) not in train_tx_set:
                        candidate_indices.append(i)
                
                candidate_indices = np.array(candidate_indices)
                
                if len(candidate_indices) < eval_total_points:
                    print(f"警告: 可用测试点({len(candidate_indices)})少于请求的测试点数({eval_total_points})，使用所有可用点")
                    eval_total_points = len(candidate_indices)
                
                # 从候选点中随机采样
                selected_indices = np.random.choice(candidate_indices, eval_total_points, replace=False)
                selected_points = all_points[selected_indices]
                
                # 按类别分配（根据坐标判断）
                new_domain_eval = []
                new_left_eval = []
                new_right_eval = []
                new_initial_eval = []
                
                for point in selected_points:
                    tx = point[0:2]
                    u = point[2:3]
                    if abs(tx[1] - X_MIN) < 1e-10:  # left boundary
                        new_left_eval.append(point)
                    elif abs(tx[1] - X_MAX) < 1e-10:  # right boundary
                        new_right_eval.append(point)
                    elif abs(tx[0] - T_MIN) < 1e-10:  # initial condition
                        new_initial_eval.append(point)
                    else:  # domain
                        new_domain_eval.append(point)
                
                domain_eval = np.array(new_domain_eval) if new_domain_eval else np.empty((0, 3))
                left_eval = np.array(new_left_eval) if new_left_eval else np.empty((0, 3))
                right_eval = np.array(new_right_eval) if new_right_eval else np.empty((0, 3))
                initial_eval = np.array(new_initial_eval) if new_initial_eval else np.empty((0, 3))
                
                # 排序
                if domain_eval.shape[0] > 0:
                    domain_eval = domain_eval[np.argsort(domain_eval[:, 0])]
                if left_eval.shape[0] > 0:
                    left_eval = left_eval[np.argsort(left_eval[:, 0])]
                if right_eval.shape[0] > 0:
                    right_eval = right_eval[np.argsort(right_eval[:, 0])]
                if initial_eval.shape[0] > 0:
                    initial_eval = initial_eval[np.argsort(initial_eval[:, 0])]
                
                print(f"从整个网格中采样了 {eval_total_points} 个测试点（排除训练点）")
            elif eval_total_points < total_eval_points:
                # 从剩余点中固定采样指定数量的测试点
                # 合并所有剩余点
                all_remaining_points = np.concatenate(eval_arrays, axis=0)
                all_remaining_tx = all_remaining_points[:, 0:2]
                
                # 随机采样固定数量的点
                if eval_total_points > all_remaining_points.shape[0]:
                    print(f"警告: 请求的测试点数({eval_total_points})超过剩余点数({all_remaining_points.shape[0]})，使用所有剩余点")
                    eval_total_points = all_remaining_points.shape[0]
                
                selected_indices = np.random.choice(
                    all_remaining_points.shape[0], eval_total_points, replace=False
                )
                selected_points = all_remaining_points[selected_indices]
                
                # 按类别分配（根据坐标判断）
                new_domain_eval = []
                new_left_eval = []
                new_right_eval = []
                new_initial_eval = []
                
                for point in selected_points:
                    tx = point[0:2]
                    if abs(tx[1] - X_MIN) < 1e-10:  # left boundary
                        new_left_eval.append(point)
                    elif abs(tx[1] - X_MAX) < 1e-10:  # right boundary
                        new_right_eval.append(point)
                    elif abs(tx[0] - T_MIN) < 1e-10:  # initial condition
                        new_initial_eval.append(point)
                    else:  # domain
                        new_domain_eval.append(point)
                
                domain_eval = np.array(new_domain_eval) if new_domain_eval else np.empty((0, 3))
                left_eval = np.array(new_left_eval) if new_left_eval else np.empty((0, 3))
                right_eval = np.array(new_right_eval) if new_right_eval else np.empty((0, 3))
                initial_eval = np.array(new_initial_eval) if new_initial_eval else np.empty((0, 3))
                
                # 排序
                if domain_eval.shape[0] > 0:
                    domain_eval = domain_eval[np.argsort(domain_eval[:, 0])]
                if left_eval.shape[0] > 0:
                    left_eval = left_eval[np.argsort(left_eval[:, 0])]
                if right_eval.shape[0] > 0:
                    right_eval = right_eval[np.argsort(right_eval[:, 0])]
                if initial_eval.shape[0] > 0:
                    initial_eval = initial_eval[np.argsort(initial_eval[:, 0])]
                
                print(f"从剩余点中采样了 {eval_total_points} 个测试点")
        
        # 计算最大尺寸（处理空数组）
        sizes = [domain.shape[0], left.shape[0], right.shape[0], initial.shape[0]]
        max_size = max(sizes) if any(s > 0 for s in sizes) else 0
        
        # 计算min_x和max_x（从所有训练数据中）
        all_train_tx = []
        if domain.shape[0] > 0:
            all_train_tx.append(domain[:, 0:2])
        if left.shape[0] > 0:
            all_train_tx.append(left[:, 0:2])
        if right.shape[0] > 0:
            all_train_tx.append(right[:, 0:2])
        if initial.shape[0] > 0:
            all_train_tx.append(initial[:, 0:2])
        
        if len(all_train_tx) > 0:
            all_train_tx = np.concatenate(all_train_tx, axis=0)
            self.min_x = torch.tensor(
                np.min(all_train_tx, axis=0), dtype=torch.float32
            ).to(device)
            self.max_x = torch.tensor(
                np.max(all_train_tx, axis=0), dtype=torch.float32
            ).to(device)
        else:
            # 如果没有训练数据，使用默认值
            self.min_x = torch.tensor([T_MIN, X_MIN], dtype=torch.float32).to(device)
            self.max_x = torch.tensor([T_MAX, X_MAX], dtype=torch.float32).to(device)
        
        # 训练数据
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
        
        # 评估数据（剩下的点）
        # 处理空数组的情况
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
        
        self.size = max_size
    
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
        """返回评估数据（剩下的点）"""
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
    
    def __len__(self):
        return self.size


def load_burgers_exact_data(nx=256, nt=100):
    """
    Generate exact solution data for evaluation.
    
    Returns:
        t, x, u_xt: time array, space array, and solution matrix
    """
    x = np.linspace(X_MIN, X_MAX, nx)
    t = np.linspace(T_MIN, T_MAX, nt)
    T, X = np.meshgrid(t, x)
    u_xt = exact_solution(X, T)
    return t, x, u_xt

