import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.linalg import eigh, eig
from scipy.sparse.linalg import eigs
from define_functions_for_rule import define_functions_for_rule
from KoopPseudoSpec import KoopPseudoSpec
from hermpts import hermpts
from reduce_sparse_grid import reduce_sparse_grid
from smolyak_grid import smolyak_grid
from scipy.integrate import solve_ivp
from joblib import Parallel, delayed

# Set up the simulation
SCALE = 1/10
N = 16
dim = 3
delta_t = 0.05
SIGMA = 10
BETA = 8/3
RHO = 40
asq = 4/BETA - 1
dvec = np.arange(1, 21)
dvec = ((2*dvec - 1)**2 + asq) / (1 + asq)

# Define ODEFUN based on dimension
def ODEFUN(t, y):
    if dim==3:
        return [SIGMA*(y[1]-y[0]), y[0]*(RHO-y[2]*SCALE)-y[1], y[0]*y[1]*SCALE-BETA*y[2]]
    elif dim==5:
        return np.array([
            SIGMA * (y[1] - y[0]),
            y[0] * (RHO - y[2] * SCALE) - y[1],
            y[0] * y[1] * SCALE - BETA * y[2] - y[0] * y[3] * SCALE,
            y[0] * y[2] * SCALE - 2 * y[0] * y[4] * SCALE - dvec[1] * y[3],
            2 * y[0] * y[3] * SCALE - 4 * BETA * y[4]
        ])
    elif dim==6:
        return np.array([
            SIGMA * (y[1] - y[0]),
            -y[0] * y[2] * SCALE + y[3] * y[2] * SCALE - 2 * y[3] * y[5] * SCALE + RHO * y[0] - y[1],
            y[0] * y[1] * SCALE - y[0] * y[4] * SCALE - y[3] * y[1] * SCALE - BETA * y[2],
            -dvec[1] * SIGMA * y[3] + SIGMA / dvec[1] * y[4],
            y[0] * y[2] * SCALE - 2 * y[0] * y[5] * SCALE + RHO * y[3] - dvec[1] * y[4],
            2 * y[0] * y[4] * SCALE + 2 * y[3] * y[1] * SCALE - 4 * BETA * y[5]
        ])
    elif dim==8:
        return np.array([
            SIGMA * (y[1] - y[0]),
            -y[0] * y[2] * SCALE + y[3] * y[2] * SCALE - 2 * y[3] * y[5] * SCALE + RHO * y[0] - y[1],
            y[0] * y[1] * SCALE - BETA * y[2] - y[0] * y[4] * SCALE - y[3] * y[1] * SCALE - y[3] * y[6] * SCALE,
            -dvec[1] * SIGMA * y[3] + SIGMA / dvec[1] * y[4],
            y[0] * y[2] * SCALE - 2 * y[0] * y[5] * SCALE - dvec[1] * y[4] + RHO * y[3] - 3 * y[3] * y[7] * SCALE,
            2 * y[0] * y[4] * SCALE - 4 * BETA * y[5] + 2 * y[3] * y[1] * SCALE - 2 * y[0] * y[6] * SCALE,
            2 * y[0] * y[5] * SCALE - 3 * y[0] * y[7] * SCALE + y[3] * y[2] * SCALE - dvec[2] * y[6],
            3 * y[0] * y[6] * SCALE + 3 * y[3] * y[4] * SCALE - 9 * BETA * y[7]
        ])
    elif dim==9:
        return np.array([
            SIGMA * (y[1] - y[0]),
            -y[0] * y[2] * SCALE + RHO * y[0] - y[1] + y[3] * y[2] * SCALE - 2 * y[3] * y[5] * SCALE + 2 * y[6] * y[5] * SCALE - 3 * y[6] * y[8] * SCALE,
            y[0] * y[1] * SCALE - BETA * y[2] - y[0] * y[4] * SCALE - y[3] * y[1] * SCALE - y[3] * y[7] * SCALE - y[6] * y[4] * SCALE,
            -dvec[1] * SIGMA * y[3] + SIGMA / dvec[1] * y[4],
            y[0] * y[2] * SCALE - 2 * y[0] * y[5] * SCALE - dvec[1] * y[4] + RHO * y[3] - 3 * y[3] * y[8] * SCALE + y[6] * y[2] * SCALE,
            2 * y[0] * y[4] * SCALE - 4 * BETA * y[5] + 2 * y[3] * y[1] * SCALE - 2 * y[0] * y[7] * SCALE - 2 * y[6] * y[1] * SCALE,
            -dvec[2] * SIGMA * y[6] + SIGMA / dvec[2] * y[7],
            2 * y[0] * y[5] * SCALE - 3 * y[0] * y[8] * SCALE + y[3] * y[2] * SCALE - dvec[2] * y[7] + RHO * y[6],
            3 * y[0] * y[7] * SCALE + 3 * y[3] * y[4] * SCALE - 9 * BETA * y[8] + 3 * y[6] * y[1] * SCALE
        ])
    elif dim==11:
        return np.array([
            SIGMA * (y[1] - y[0]),
            -y[0] * y[2] * SCALE + RHO * y[0] - y[1] + y[3] * y[2] * SCALE - 2 * y[3] * y[5] * SCALE + 2 * y[6] * y[5] * SCALE - 3 * y[6] * y[8] * SCALE,
            y[0] * y[1] * SCALE - BETA * y[2] - y[0] * y[4] * SCALE - y[3] * y[1] * SCALE - y[3] * y[7] * SCALE - y[6] * y[4] * SCALE - y[6] * y[9] * SCALE,
            -dvec[1] * SIGMA * y[3] + SIGMA / dvec[1] * y[4],
            y[0] * y[2] * SCALE - 2 * y[0] * y[5] * SCALE - dvec[1] * y[4] + RHO * y[3] - 3 * y[3] * y[8] * SCALE + y[6] * y[2] * SCALE - 4 * y[6] * y[10] * SCALE,
            2 * y[0] * y[4] * SCALE - 4 * BETA * y[5] + 2 * y[3] * y[1] * SCALE - 2 * y[0] * y[7] * SCALE - 2 * y[6] * y[1] * SCALE - 2 * y[3] * y[9] * SCALE,
            -dvec[2] * SIGMA * y[6] + SIGMA / dvec[2] * y[7],
            2 * y[0] * y[5] * SCALE - 3 * y[0] * y[8] * SCALE + y[3] * y[2] * SCALE - dvec[2] * y[7] + RHO * y[6] - 4 * y[3] * y[10] * SCALE,
            3 * y[0] * y[7] * SCALE + 3 * y[3] * y[4] * SCALE - 9 * BETA * y[8] + 3 * y[6] * y[1] * SCALE - 3 * y[0] * y[9] * SCALE,
            -4 * y[0] * y[10] * SCALE + 2 * y[3] * y[5] * SCALE + 3 * y[0] * y[8] * SCALE + y[6] * y[2] * SCALE - dvec[3] * y[9],
            4 * y[0] * y[9] * SCALE + 4 * y[3] * y[7] * SCALE + 4 * y[6] * y[4] * SCALE - 16 * BETA * y[10]
        ])


[lev2knots,idxset]=define_functions_for_rule('SM',dim);

# Define the integration options (tolerances)
options = {'rtol': 1e-13, 'atol': 1e-14}

# Find the quadrature/data points
knots = lambda n: hermpts(n)

S, _ = smolyak_grid(dim, int(np.log2(2 * N) + 4), knots, lev2knots, idxset)

# ... (this part requires the smolyak_grid and reduce_sparse_grid functions which are not provided in the MATLAB code)
Sr = reduce_sparse_grid(S)

xQ = np.array(Sr['knots']).T
wQ = np.array(Sr['weights']).T / np.sqrt(np.pi**dim)

M = len(xQ)  # number of data points
DATA = np.zeros((M, dim))

def solve_ode(j):
    Y0 = xQ[j]
    sol = solve_ivp(ODEFUN, [1e-6, delta_t, 2*delta_t], Y0, t_eval=[delta_t], method='RK45')  # Assuming RK45 is suitable
    return sol.y[:, 1]

DATA = Parallel(n_jobs=-1)(delayed(solve_ode)(j) for j in range(M))

X_hermite = [np.zeros((M, N)) for _ in range(dim)]
Y_hermite = [np.zeros((M, N)) for _ in range(dim)]

for jj in range(dim):
    X_hermite[jj][:, 0] = 1
    X_hermite[jj][:, 1] = np.sqrt(2) * xQ[:, jj]
    Y_hermite[jj][:, 0] = 1
    Y_hermite[jj][:, 1] = np.sqrt(2) * DATA[:, jj]
    for j in range(2, N):
        X_hermite[jj][:, j] = (np.sqrt(2) / np.sqrt(j - 1)) * X_hermite[jj][:, j - 1] * xQ[:, jj] - (np.sqrt(j - 2) / np.sqrt(j - 1)) * X_hermite[jj][:, j - 2]
        Y_hermite[jj][:, j] = (np.sqrt(2) / np.sqrt(j - 1)) * Y_hermite[jj][:, j - 1] * DATA[:, jj] - (np.sqrt(j - 2) / np.sqrt(j - 1)) * Y_hermite[jj][:, j - 2]

# Create list of hyperbolic cross indices
Index = [None] * dim
Index0 = np.ones((N, 1))

Index[0] = np.exp(np.arange(1, N+1)).reshape(-1, 1)

for kk in range(1, dim):
    Index[kk] = np.kron(Index0, np.exp(np.arange(1, N+1)).reshape(-1, 1))
    II = np.log(Index[kk])
    
    for jj in range(kk):
        Index[jj] = np.kron(Index[jj], np.ones((N, 1)))
        II = II * np.log(Index[jj])
    
    I = np.where(II < N+1)[0]
    
    for jj in range(kk+1):
        Index[jj] = Index[jj][I]
    
    Index0 = np.ones((len(I), 1))

Index = [np.log(arr) for arr in Index]

Ntrunc = len(Index[0])
A_matrix = np.zeros((Ntrunc, Ntrunc))
L_matrix = np.zeros((Ntrunc, Ntrunc))
G_matrix = np.zeros((Ntrunc, Ntrunc))

def compute_matrices(ii):
    A_row = np.zeros(Ntrunc)
    L_row = np.zeros(Ntrunc)
    G_row = np.zeros(Ntrunc)
    Xprod1 = X_hermite[0][:, int(Index[0][ii])]
    Yprod1 = Y_hermite[0][:, int(Index[0][ii])]
    for ll in range(1, dim):
        Xprod1 = Xprod1 * X_hermite[ll][:, int(Index[ll][ii])]
        Yprod1 = Yprod1 * Y_hermite[ll][:, int(Index[ll][ii])]
    for jj in range(Ntrunc):
        Xprod2 = X_hermite[0][:, int(Index[0][jj])]
        Yprod2 = Y_hermite[0][:, int(Index[0][jj])]
        for ll in range(1, dim):
            Xprod2 = Xprod2 * X_hermite[ll][:, int(Index[ll][jj])]
            Yprod2 = Yprod2 * Y_hermite[ll][:, int(Index[ll][jj])]
        G_row[jj] = np.sum(wQ * np.conj(Xprod1) * Xprod2)
        A_row[jj] = np.sum(wQ * np.conj(Xprod1) * Yprod2)
        L_row[jj] = np.sum(wQ * np.conj(Yprod1) * Yprod2)
    return A_row, L_row, G_row

results = Parallel(n_jobs=10)(delayed(compute_matrices)(ii) for ii in range(Ntrunc))

for ii, (A_row, L_row, G_row) in enumerate(results):
    A_matrix[ii, :] = A_row
    L_matrix[ii, :] = L_row
    G_matrix[ii, :] = G_row

L_matrix = (L_matrix + L_matrix.T) / 2
G_matrix = (G_matrix + G_matrix.T) / 2

# Define the points
x_pts = np.arange(-2, 8.05, 0.05)
y_pts = np.arange(-3, 3.05, 0.05)

# Compute z_pts
z_pts = np.kron(x_pts, np.ones(len(y_pts))) + 1j * np.kron(np.ones(len(x_pts)), y_pts)
z_pts = z_pts.flatten()

# Assuming you have a function called KoopPseudoSpec to compute pseudospectra
RES = KoopPseudoSpec(G_matrix, A_matrix, L_matrix, z_pts, parallel=True)

# Reshape RES
RES = RES.reshape(len(y_pts), len(x_pts))

# Construct cells containing Hermite evaluations
# ... (this part requires the hermpts function which is not provided in the MATLAB code)

# Form the ResDMD matrices
# ... (this part requires the parfor_progress function which is not provided in the MATLAB code)

# Compute pseudospectra
# Assuming KoopPseudoSpec is defined elsewhere
# RES = KoopPseudoSpec(G_matrix, A_matrix, L_matrix, z_pts, parallel='on')

# Compute EDMD eigenvalues
vals, vecs = eig(A_matrix)
E = np.diag(vals)

# Plotting
x_pts = np.arange(-2, 8.05, 0.05)
y_pts = np.arange(-3, 3.05, 0.05)
z_pts = np.kron(x_pts, np.ones(len(y_pts))) + 1j * np.kron(np.ones(len(x_pts)), y_pts)
z_pts = z_pts.flatten()
RES_reshaped = np.reshape(RES, (len(y_pts), len(x_pts)))

plt.figure()
v = [0.01, 0.05, 0.1, 1]
plt.contour(np.reshape(np.real(z_pts), (len(y_pts), len(x_pts))),
            np.reshape(np.imag(z_pts), (len(y_pts), len(x_pts))),
            np.real(RES_reshaped), v, colors='k', linewidths=1.5)
plt.gca().set_aspect('equal', adjustable='box')
plt.plot(np.real(E), np.imag(E), '.m')
plt.ylim([-3, 3])
plt.show()
