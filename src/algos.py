import numpy as np
import time

class Recorder:
    """
    A simple class to record the history of optimization.
    """
    def __init__(self):
        self.history = []
        self.t0 = time.perf_counter()

    def record(self, iteration, X, y, lambda_param, w):
        objective_value = objective(X, y, w, lambda_param)
        sparsity_value = sparsity(w)
        residual_value = residual_norm(X, y, w)
        timestamp = time.perf_counter() - self.t0

        self.history.append((iteration, objective_value, sparsity_value, residual_value, timestamp))


def soft_thresholding(x, threshold):
    """
    The soft-thresholding operator.
    """
    return np.sign(x) * np.maximum(np.abs(x) - threshold, 0.)

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
    return np.sum(w == 0) / len(w)

def residual_norm(X, y, w):
    """
    Compute the L2 norm of the residual (Xw - y).
    """
    return np.linalg.norm(X @ w - y)

def ista_lasso(X, y, lambda_param, w0, iters, learning_rate, callback=None):
    """
    Iterative Shrinkage-Thresholding Algorithm (ISTA) for LASSO regression.

    Args:
        X (np.ndarray): Covariate matrix (n_samples, n_features).
        y (np.ndarray): Response vector (n_samples,).
        lambda_param (float): The L1 penalty parameter (λ).
        w0 (np.ndarray): Initial weight vector (n_features,).
        iters (int): Number of iterations.
        learning_rate (float): The step size (t0 in some literature).
        callback (callable, optional): A function to be called after each iteration.
    Returns:
        np.ndarray: The final weight vector w.
    References:
        - Beck and Teboulle, "A Fast Iterative Shrinkage-Thresholding Algorithm for Linear Inverse Problems", SIAM Journal on Imaging Sciences, 2009.
        - Boyd et al., "Distributed Optimization and Statistical Learning via the Alternating Direction Method of Multipliers", Foundations and Trends in Machine Learning, 2011.
    """
    w = w0.copy()
    if w.shape[0] != X.shape[1]:
        raise ValueError("Initial weight vector w0 must have the same number of features as X")
    if w.ndim != 1:
        raise ValueError("Initial weight vector w0 must be a 1D array")
    
    # Pre-calculate X.T @ X and X.T @ y for efficiency
    X_T_X = X.T @ X
    X_T_y = X.T @ y

    L = np.linalg.eigvalsh(X_T_X)[-1]  # or use power iteration
    if learning_rate > 1/L:
        print("Warning: step size may exceed Lipschitz constant")

    for i in range(iters):
        # Calculate the gradient of the smooth part (least squares)
        # Gradient = X.T @ (X @ w - y) = X_T_X @ w - X_T_y
        gradient = X_T_X @ w - X_T_y
        
        # Proximal gradient update step (ISTA update rule)
        # z = w - learning_rate * gradient
        z = w - learning_rate * gradient
        
        # Apply the soft-thresholding operator
        w = soft_thresholding(z, learning_rate * lambda_param)

        # Call the callback function if provided
        if callback is not None:
            callback(i, X, y, lambda_param, w)

    return w

def fista_lasso(X, y, lambda_param, w0, iters, learning_rate, callback=None):
    """
    Fast Iterative Shrinkage-Thresholding Algorithm (FISTA) for LASSO regression.

    Args:
        X (np.ndarray): Covariate matrix (n_samples, n_features).
        y (np.ndarray): Response vector (n_samples,).
        lambda_param (float): The L1 penalty parameter (λ).
        w0 (np.ndarray): Initial weight vector (n_features,).
        iters (int): Number of iterations.
        learning_rate (float): The step size (t0 in some literature).
        callback (callable, optional): A function to be called after each iteration.
    Returns:
        np.ndarray: The final weight vector w.
    References:
        - Beck and Teboulle, "A Fast Iterative Shrinkage-Thresholding Algorithm for Linear Inverse Problems", SIAM Journal on Imaging Sciences, 2009.
        - Boyd et al., "Distributed Optimization and Statistical Learning via the Alternating Direction Method of Multipliers", Foundations and Trends in Machine Learning, 2011.
    """
    w = w0.copy()
    if w.shape[0] != X.shape[1]:
        raise ValueError("Initial weight vector w0 must have the same number of features as X")
    if w.ndim != 1:
        raise ValueError("Initial weight vector w0 must be a 1D array")
    
    z = w0.copy()  # Auxiliary variable for momentum
    t = 1  # Momentum parameter

    # Pre-calculate X.T @ X and X.T @ y for efficiency
    X_T_X = X.T @ X
    X_T_y = X.T @ y

    L = np.linalg.eigvalsh(X_T_X)[-1]  # or use power iteration
    if learning_rate > 1/L:
        print("Warning: step size may exceed Lipschitz constant")

    for i in range(iters):
        # Calculate the gradient of the smooth part (least squares)
        gradient = X_T_X @ z - X_T_y
        
        # Proximal gradient update step (FISTA update rule)
        z_new = z - learning_rate * gradient
        w_new = soft_thresholding(z_new, learning_rate * lambda_param)

        # Update momentum parameter
        t_new = (1 + np.sqrt(1 + 4 * t**2)) / 2
        
        # Update auxiliary variable with momentum
        z = w_new + ((t - 1) / t_new) * (w_new - w)

        # Update variables for next iteration
        w = w_new
        t = t_new

        # Call the callback function if provided
        if callback is not None:
            callback(i, X, y, lambda_param, w)

    return w


def admm_lasso(X, y, lambda_param, w0, iters, rho, callback=None):
    """
    Alternating Direction Method of Multipliers (ADMM) for LASSO regression.

    Args:
        X (np.ndarray): Covariate matrix (n_samples, n_features).
        y (np.ndarray): Response vector (n_samples,).
        lambda_param (float): The L1 penalty parameter (λ).
        w0 (np.ndarray): Initial weight vector (n_features,).
        iters (int): Number of iterations.
        rho (float): Augmented Lagrangian parameter.
        callback (callable, optional): A function to be called after each iteration.
    Returns:
        np.ndarray: The final weight vector w.
    References:
        - Boyd et al., "Distributed Optimization and Statistical Learning via the Alternating Direction Method of Multipliers", Foundations and Trends in Machine Learning, 2011.
    """
    n, d = X.shape
    w = w0.copy()
    if w.shape[0] != X.shape[1]:
        raise ValueError("Initial weight vector w0 must have the same number of features as X")
    if w.ndim != 1:
        raise ValueError("Initial weight vector w0 must be a 1D array")
    
    z = np.zeros(d)  # Auxiliary variable for the L1 penalty
    u = np.zeros(d)  # Dual variable

    # Pre-calculate X.T @ X and X.T @ y for efficiency
    X_T_X = X.T @ X
    X_T_y = X.T @ y

    # Pre-calculate the matrix to invert for the w-update
    A = X_T_X + rho * np.eye(d)

    L = np.linalg.cholesky(A)

    for i in range(iters):
        rhs = X_T_y + rho * (z - u)
        w = np.linalg.solve(L.T, np.linalg.solve(L, rhs))
        # w-update: minimize (1/2)||Xw - y||^2 + (rho/2)||w - z + u||^2
        #w = A_inv @ (X_T_y + rho * (z - u))

        # z-update: minimize lambda_param * ||z||_1 + (rho/2)||w - z + u||^2
        z = soft_thresholding(w + u, lambda_param / rho)

        # u-update: dual variable update
        u += w - z

        # Call the callback function if provided
        if callback is not None:
            callback(i, X, y, lambda_param, w)

    return w