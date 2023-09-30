import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from KoopPseudoSpec import KoopPseudoSpec

# Set parameters
np.random.seed(1)
M = 10**5
delta_t = 0.05
SIGMA = 10
BETA = 8/3
RHO = 28

def ODEFUN(t, y):
    return [SIGMA*(y[1]-y[0]), y[0]*(RHO-y[2])-y[1], y[0]*y[1]-BETA*y[2]]

N=100
# Produce the data
Y0 = (np.random.rand(3) - 0.5) * 4
sol = solve_ivp(ODEFUN, [0.000001, (10000+M+2+N)*delta_t], Y0, t_eval=np.arange(0.000001, (10000+M+2+N)*delta_t, delta_t))
Y = sol.y[:, 10000:].T

# Reference scalar-valued spectral measure
PX1 = np.zeros((M, N+1))
PX1[:, 0] = Y[:M, 0]
PX2 = np.zeros((M, N+1))
PX2[:, 0] = Y[:M, 1]
PX3 = np.zeros((M, N+1))
PX3[:, 0] = Y[:M, 2]

for j in range(1, N+1):
    PX1[:, j] = Y[j:j+M, 0]
    PX2[:, j] = Y[j:j+M, 1]
    PX3[:, j] = Y[j:j+M, 2]

PX = np.hstack([PX1[:, :N], PX2[:, :N], PX3[:, :N]])
PY = np.hstack([PX1[:, 1:N+1], PX2[:, 1:N+1], PX3[:, 1:N+1]])

G = (PX.T @ PX) / M
A = (PX.T @ PY) / M
L = (PY.T @ PY) / M

G = (G + L) / 2
L = G

E = np.exp(1j * np.array([0.008, 0.019, 0.05, 0.072, 0.099, 0.31, 0.404, 0.499, 0.78]))

# Assuming KoopPseudoSpec is defined elsewhere
_, RES, V = KoopPseudoSpec(G, A, L, [], 'z_pts2', E)
# _, RES, V = eng.KoopPseudoSpec(G, A, L, [], 'z_pts2', E, nargout=3)

V = np.array(V)
# print(V.shape)
# print(V)

for j, e in enumerate(E):
    C = PX[:min(2*10**4, M), :] @ V[:, j]
    
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    scatter = ax.scatter(Y[:min(2*10**4, M), 0], Y[:min(2*10**4, M), 1], Y[:min(2*10**4, M), 2], c=np.real(C), cmap='turbo', s=3)
    plt.colorbar(scatter)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.view_init(20.4, 44.13)
    # Assuming RES is defined
    ax.set_title(f'θ={np.imag(np.log(e))}, res={RES[j]}')
    plt.show()

