# pyright: reportUnboundVariable=false
import numpy as np
def softmax_regression_epoch_cpp(
    X: np.ndarray,          # shape (m, n), dtype=float32
    y: np.ndarray,          # shape (m,), dtype=uint8
    theta: np.ndarray,      # shape (n, k), dtype=float32
    lr: float,              # learning rate
    batch: int,             # batch size
) -> None: ...
