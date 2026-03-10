import numpy as np

def objective(X, y, w, lambda_param):
    """
    Compute the LASSO objective function value.
    """
    residual = X @ w - y
    return 0.5 * np.sum(residual**2) + lambda_param * np.sum(np.abs(w))

def sparsity(w):
    """
    Compute the sparsity of the weight vector w.
    """
    return np.sum(np.abs(w) > 1e-6)

def residual_norm(X, y, w):
    """
    Compute the L2 norm of the residual (Xw - y).
    """
    return np.linalg.norm(X @ w - y)
