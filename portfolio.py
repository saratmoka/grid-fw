"""
Author: Sarat Moka
Email: s.moka@unsw.edu.au
Website: saratmoka.com
Affiliation: The University of New South Wales
Date: 15 May 2025
"""


"""
Packages required
"""
import numpy as np
from numpy.linalg import norm, inv
from scipy.sparse.linalg import cg

#%%
'''
Helper functions for Grid-FW
'''
def sci(x, dig=2):
    return f"{x:.{dig}e}".replace("e-0", "e-").replace("e+0", "e+")

def print_in_box(text):
    # Split into lines in case the text contains newline characters
    lines = text.split("\n") 

    # Find the maximum width among all lines
    width = max(len(line) for line in lines)

    # Print the top border
    print("+" + "-" * (width + 2) + "+")

    # Print each line, padded to the same width
    for line in lines:
        print("| " + line + " " * (width - len(line)) + " |")

    # Print the bottom border
    print("+" + "-" * (width + 2) + "+")

def power_method(matrix, max_iter=1000, tol=1e-8):
    """
    Compute the dominant eigenvalue/eigenvector of a positive definite matrix using the Power Method.

    Parameters:
        matrix (np.ndarray): Positive definite square matrix
        max_iter (int): Maximum iterations (default: 1000)
        tol (float): Convergence tolerance (default: 1e-8)

    Returns:
        (float, np.ndarray): Dominant eigenvalue and eigenvector
    """
    n = matrix.shape[0]
    if n != matrix.shape[1]:
        raise ValueError("Matrix must be square")

    # Initialize with random unit vector
    b_k = np.random.randn(n)
    b_k /= norm(b_k) + 1e-16  # Avoid zero division

    # Pre-allocate memory
    Ab = np.empty_like(b_k)
    prev_rayleigh = None  # Initialize as None to handle first iteration
    epsilon = 1e-16

    for _ in range(max_iter):
        np.dot(matrix, b_k, out=Ab)
        rayleigh = b_k.dot(Ab)

        # Skip convergence check in the first iteration
        if prev_rayleigh is not None:
            # Handle near-zero denominator
            denom = abs(prev_rayleigh) + epsilon
            rel_error = abs(rayleigh - prev_rayleigh) / denom
            if rel_error < tol:
                break

        prev_rayleigh = rayleigh

        # Safe normalization
        norm_Ab = norm(Ab)
        if norm_Ab < epsilon:
            break  # Avoid division by zero
        np.divide(Ab, norm_Ab, out=b_k)

    return rayleigh, b_k

def geometric_sequence(a, b, n):
    r = (b / a) ** (1 / (n - 1))
    return [a * r**i for i in range(n)]

def discrete_f(Sigma, subset):
    Sigma_s = Sigma[subset, :][:, subset]
    Sigma_inv = inv(Sigma_s)
    return - np.sum(Sigma_inv)

def variance(Sigma, subset):
    Sigma_s = Sigma[subset, :][:, subset]
    Sigma_inv = inv(Sigma_s)
    return 1/np.sum(Sigma_inv)

def covariance_to_correlation(cov_matrix):
    std_dev = np.sqrt(np.diag(cov_matrix))  # Standard deviations (square root of diagonal elements)
    corr_matrix = cov_matrix / np.outer(std_dev, std_dev)  # Normalize covariance matrix
    return corr_matrix

def f1(t, Sigma, v, delta):
    t_sqr = t*t
    dia = delta*(1 - t_sqr)  

    Sigmat = np.diag(t)@Sigma@np.diag(t) + np.diag(v * dia)
    vec = inv(Sigmat)@t
    value = -t.T@vec
         
    return value

def f2(t, Sigma, v, delta, x0, cg_maxiter=None, cg_tol=1e-5):
    p = t.shape[0]
    
    if cg_maxiter == None:
        cg_maxiter = p
   
    t_sqr = t*t
    t_sqr[t_sqr < 1e-6] = 1e-6  #For numerical stability, small values are mapped to 1e-8. 
    dia = delta*(1 - t_sqr)/t_sqr    

    M = Sigma.copy()
    np.fill_diagonal(M, M.diagonal() + dia*v)
    
    
    ## Obtaining gamma estimate
    #vec, _ = cg(M, np.ones(p), x0=x0, maxiter=cg_maxiter, rtol=cg_tol)
    vec, _ = cg(M, np.ones(p), x0=x0, maxiter=cg_maxiter, tol=cg_tol)
    value = -np.sum(vec)
    return value

def f_grad(t, Sigma, v, delta, gamma, 
            cg_maxiter=None, 
            cg_tol=1e-5):
    
    p = t.shape[0]
    
    if cg_maxiter == None:
        cg_maxiter = p
   
    t_sqr = t*t
    t_sqr[t_sqr < 1e-6] = 1e-6  #For numerical stability, small values are mapped to 1e-8. 
    dia = delta*(1 - t_sqr)/t_sqr    

    ## Construct Mt
    M = Sigma.copy()
    np.fill_diagonal(M, M.diagonal() + dia*v)
    
    
    ## Obtaining gamma estimate
    gamma, _ = cg(M, np.ones(p), x0=gamma, maxiter=cg_maxiter, rtol=cg_tol)
    
    ## Constructing gradient
    grad = -2*delta*(gamma*gamma*v)/(t**3)

    return grad, gamma

        
#%%
def fw(Sigma, k, t_init, v, delta, alpha, gamma,
        fw_maxiter=100,     # Maximum number of iterations allowed by FW
        cg_maxiter=None,     # Maximum number of iterations allowed by CG
        cg_tol=1e-3):        # Tolerance of CG

    p = Sigma.shape[0]
    
    # Setting default values
    if cg_maxiter == None:
        cg_maxiter = p
    
    t = np.array(t_init)
    if t.shape[0] == 0:
        t = np.ones(p)*(k/p)
        
    model_best = np.array(range(k))
    var_best = np.inf
    
    t_list = [t.copy()]

    converge = True
    if np.any((t >= 0.01) & (t <= 0.99)):
        converge = False
        
    l = 0
    stop = False
    while not stop:
        l += 1
            
        grad, gamma = f_grad(t, Sigma, v, delta, gamma, cg_maxiter=cg_maxiter, cg_tol=cg_tol)
        
        # Create the binary vector s
        model = np.argsort(grad)[:k]  # Indices of k smallest elements of the gradient
        s = np.zeros(p, dtype=int)
        s[model] = 1
        
        # Update t
        t = (1 - alpha) * t + alpha * s
        t[t < 0.001] = 0.001 # truncation for numerical stability
        t[t > 0.999] = 0.999 # truncation for numerical stability
        t_list.append(t.copy())


        if l >= fw_maxiter:
            stop = True
            
        model = np.sort(model)
        var = variance(Sigma, model)
        if var < var_best:
            model_best = model.copy()
            var_best = var

    if converge:
        map_curr = np.ones(p, dtype=int)*(-1)
        map_curr[t_list[0] > 0.99] = 1
        map_curr[t_list[0] < 0.01] = 0
        for i in range(len(t_list)-1):
            
            map_next = np.ones(p, dtype=int)*(-1)
            map_next[t_list[i+1] > 0.99] = 1
            map_next[t_list[i+1] < 0.01] = 0
            
            if np.any(map_next == -1) or not np.array_equal(map_next, map_curr):
                converge = False
                break
                
            map_curr = map_next.copy()
        
    
    # Collect the results
    result = {
        't' : t,
        's' : s,
        't_list' : t_list,
        'converge' : converge, 
        'gamma' : gamma,
		'model_final' : model,
		'model_best' : model_best,
		'var_final' : var,
        'var_best' : var_best,
		'niter' : l,
		}
    return  result


def grid_fw(Sigma, k, 
        scale = True,
        grid_type = 'geometric', # grid type
        alpha=0.05,           # line search is applied if true 
        maxepoch=1000,       # Maximum number of epochs 
        maxiter=100,         # Maximum number of iterations in an epoch
        patience = 10,        # patience parameter to terminate the algorithm
        cg_maxiter=None,     # Maximum number of iterations allowed by CG
        cg_tol=0.001):        # Tolerance of CG

    p = Sigma.shape[0]
    v = np.diag(Sigma)
    
    # If k is 1, just find the smallest diagonal element
    if k == 1:
        ind = np.argmin(v)
        model = np.array([ind + 1])
        t = np.zeros(p)
        t[ind] = 1
        t_path = [t]
        result = {
            "t_path" : t_path,
    		"model_final" : model,
    		"model_best" : model,
    		"var_final" : v[ind],
            "var_best" : v[ind],
            "delta_grid" : None,
            "nepoch" : 0,
            "alpha" : alpha,
            "scale" : scale,
            "grid_type" : grid_type,
            "cg_tol": cg_tol
    		}
        return result
    

    if not scale:
        v = np.ones(p)
        
    gamma = np.ones(p)*(k/p)
        
    eta_1, _ = power_method(Sigma)
    eta_p, _ = power_method(eta_1*np.identity(p) - Sigma) 
    eta_p = -eta_p + eta_1
    
    delta_max = eta_1/np.min(v)
    
    if grid_type == 'geometric':
        epsilon = min((k/p)*0.1, 0.001)
        epsilon_sqr = (epsilon)**2
        delta_min = (3*eta_p*epsilon_sqr)/(np.max(v)*(1 + 3*epsilon_sqr))
        delta_grid = geometric_sequence(delta_min, delta_max, maxepoch)
    
    if grid_type == 'linear':
        delta_grid = np.linspace(0, delta_max, maxepoch)
        delta_grid = delta_grid + delta_grid[1]

    if grid_type == 'midpoint':
        delta_grid = [delta_max/2]
            
    # Setting default values
    if cg_maxiter == None:
        cg_maxiter = p
        
    t_path = []
    
    model_best = np.array(range(k))
    var_best = np.inf
    

    t = np.ones(p)*(k/p)
    s = np.zeros(p, dtype=int)
    s[range(k)] = 1
    
    l = 0

    stop = False
    count = 0
    while not stop:
        delta = delta_grid[l]
        result_fw = fw(Sigma, k, t, v, delta, alpha, gamma=gamma, fw_maxiter=maxiter, cg_maxiter=cg_maxiter, cg_tol=cg_tol) 
        gamma = result_fw['gamma']
        l += 1
        
        if var_best > result_fw['var_best']:
            model_best = np.copy(result_fw['model_best'])
            var_best = result_fw['var_best']
        
        t = result_fw['t'].copy()
        t_path.append(t)
        
        if result_fw['converge']:
            count += 1
        else:
            count = 0
        if count >= patience:
            stop = True
    if count < patience:
        print("WARNING: Algorithm terminated after maximum epochs without convergence! Consider increasing the number of epochs.")
    
    # Collect the results
    result = {
        "t_path" : t_path,
		"model_final" : result_fw['model_final']+1,
		"model_best" : model_best+1,
		"var_final" : result_fw['var_final'],
        "var_best" : var_best,
        "delta_grid" : delta_grid,
        "nepoch" : l,
        "alpha" : alpha,
        "scale" : scale,
        "grid_type" : grid_type,
        "cg_tol": cg_tol
		}
        
    return  result


