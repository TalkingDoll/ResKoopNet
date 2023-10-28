import numpy as np

def ou_process_1d(theta, mu, sigma, X0, step_num, h):
    """
    Simulates the Ornstein-Uhlenbeck process.
    
    Parameters:
    - theta: rate of mean reversion
    - mu: long-term mean of the process
    - sigma: volatility
    - X0: initial value
    - step_num: number of steps in the simulation
    - h: Integration step size
    
    Returns:
    - A numpy array containing the simulated values
    """
    
    T = step_num * h    
    X = np.zeros(step_num)
    X[0] = X0
    
    for t in range(1, step_num):
        dW = np.sqrt(h) * np.random.randn()
        X[t] = X[t-1] + theta * (mu - X[t-1]) * h + sigma * dW
        
    return X

def ou_process_2d(theta, mu, sigma, X0, Y0, step_num, h=1e-5):
    """
    Simulates the 2D Ornstein-Uhlenbeck process.
    """
    
    X = np.zeros(step_num)
    Y = np.zeros(step_num)
    X[0] = X0
    Y[0] = Y0
    
    for t in range(1, step_num):
        dWx = np.sqrt(h) * np.random.randn()
        dWy = np.sqrt(h) * np.random.randn()
        
        X[t] = X[t-1] + theta[0] * (mu[0] - X[t-1]) * h + sigma[0] * dWx
        Y[t] = Y[t-1] + theta[1] * (mu[1] - Y[t-1]) * h + sigma[1] * dWy
        
    return X, Y



