import numpy as np

# Define the system of ODEs
def duffing_oscillator(t, Y, delta, alpha, beta):
    # If Y is just a single value, we assume the second value (y) is 0
    if np.isscalar(Y):
        x, y = Y, 0
    else:
        x, y = Y
    dxdt = y
    dydt = -delta*y - alpha*x - beta*x**3
    return dxdt, dydt