import numpy as np
from scipy.linalg import toeplitz


def generate_data(n, d, cond_number, sparsity, noise_level, struct='iid'):
    """
    Generate synthetic data for sparse linear regression.
    
    Parameters:
    -----------
    n : int
        Number of samples
    d : int
        Number of features
    cond_number : float
        Condition number of the design matrix A
    sparsity : int or float
        If int: number of non-zero entries in x*
        If float in (0,1): fraction of non-zero entries in x*
    noise_level : float
        Standard deviation of Gaussian noise
    struct : str, 'iid' or 'toeplitz'
        Structure of the design matrix
        - 'iid': Sample i.i.d. Gaussian and normalize columns
        - 'toeplitz': Create Toeplitz covariance matrix
        
    Returns:
    --------
    A : ndarray of shape (n, d)
        Design matrix
    b : ndarray of shape (n,)
        Observation vector b = Ax* + ε
    x_true : ndarray of shape (d,)
        Ground truth sparse vector x*
    """
    
    if struct == 'iid':
        # Sample i.i.d. Gaussian
        A = np.random.randn(n, d)
        
        # Normalize columns
        A = A / np.linalg.norm(A, axis=0, keepdims=True)
        
        # Adjust condition number by scaling singular values
        U, s, Vt = np.linalg.svd(A, full_matrices=False)
        s_max = s[0]
        s_min = s_max / cond_number
        # Create scaled singular values from max to min
        s_scaled = np.linspace(s_max, s_min, len(s))
        A = U @ np.diag(s_scaled) @ Vt
        
    elif struct == 'toeplitz':
        # Create Toeplitz covariance matrix with Σ_ij = ρ^|i-j|
        # For Toeplitz(ρ^|i-j|), condition number ≈ (1+ρ)/(1-ρ)
        # Solve for ρ: cond_number = (1+ρ)/(1-ρ)
        rho = (cond_number - 1) / (cond_number + 1)
        rho = np.clip(rho, 0.0, 0.9999)
        # Create first row of Toeplitz matrix
        row = rho ** np.arange(d)
        cov_matrix = toeplitz(row)
        
        # Generate data from multivariate Gaussian N(0, Σ)
        # Each row of A ~ N(0, Σ)
        L = np.linalg.cholesky(cov_matrix)
        A = np.random.randn(n, d) @ L.T
        
    else:
        raise ValueError(f"struct must be 'iid' or 'toeplitz', got {struct}")
    
    # Generate sparse ground truth x*
    # Convert sparsity to number of non-zero entries if given as fraction
    if isinstance(sparsity, float) and 0 < sparsity < 1:
        k = int(np.round(sparsity * d))
    else:
        k = int(sparsity)
    
    if k < 0 or k > d:
        raise ValueError(f"sparsity leads to k={k} nonzeros, which is outside [0, d={d}]")

    x_true = np.zeros(d)
    # Randomly choose support
    support = np.random.choice(d, size=k, replace=False)
    # Set non-zero values to ±1
    x_true[support] = np.random.choice([-1, 1], size=k)
    
    # Compute b = Ax* + ε with Gaussian noise
    b = A @ x_true + noise_level * np.random.randn(n)
    
    return A, b, x_true

