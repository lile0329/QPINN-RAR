import os
import random
import numpy as np
import torch


def seed_everything(seed: int):
    """Set seeds for Python, NumPy and PyTorch and enable deterministic behavior where possible."""
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)
    try:
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except Exception:
        pass

    try:
        torch.use_deterministic_algorithms(True)
    except Exception:
        pass

    if torch.cuda.is_available():
        os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


__all__ = ["seed_everything"]
