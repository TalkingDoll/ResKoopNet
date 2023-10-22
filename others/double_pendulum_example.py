import numpy as np
import matplotlib.pyplot as plt
import scipy.io

from IsomMeas import IsomMeas

# Load data
data = scipy.io.loadmat('double_pendulum_data.mat')
A = data['A']
N1 = data['N1'][0,0]
N2 = data['N2'][0,0]

I1 = np.maximum(np.abs(np.arange(-N1, N1+1)), 1)
I2 = np.arange(1, N2+1)
IDX = np.kron(I1, np.kron(I1, np.kron(I2, I2)))
I = np.where(IDX < 25)[0]
IDX = IDX[I]
I2 = np.where(IDX < N1)[0]

A = A[np.ix_(I2, I2)]  # sort out indexing

# Compute spectral measures
theta = np.arange(-np.pi, np.pi+0.005, 0.005)
theta = np.sort(np.append(theta[theta > 0.002], 0))
epsilon = 0.1

c1 = np.zeros(2*N1+1)
c2 = c1.copy()
c3 = np.zeros(N2)
c4 = c3.copy()

CASE = 1  # the observables in the paper
if CASE == 1:
    c1[N1+1] = 1
    c2[N1] = 1
    c3[0] = 1
    c4[0] = 1
elif CASE == 2:
    c1[N1] = 1
    c2[N1+1] = 1
    c3[0] = 1
    c4[0] = 1
elif CASE == 3:
    c1[N1] = 1
    c2[N1] = 1
    c3[1] = 1
    c4[0] = 1
else:
    c1[N1] = 1
    c2[N1] = 1
    c3[0] = 1
    c4[1] = 1

f = np.kron(c1, np.kron(c2, np.kron(c3, c4)))
f = f[I]
f = f[I2]
f = f / np.linalg.norm(f)

# Assuming you have a Python version of the IsomMeas function
nu1 = IsomMeas(np.eye(A.shape[0]), A, np.eye(A.shape[0]), f, theta, epsilon, order=1)
nu6 = IsomMeas(np.eye(A.shape[0]), A, np.eye(A.shape[0]), f, theta, epsilon, order=6)

# Plot the results
plt.figure()
plt.plot(theta, nu1, linewidth=2)
plt.plot(theta, nu6, linewidth=2)
plt.xlim([-np.pi, np.pi])
plt.legend(['$m=1$', '$m=6$'], loc='upper right', fontsize=20)
plt.show()