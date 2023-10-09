import numpy as np
import matplotlib.pyplot as plt
import pywt
from chebpy.core.chebfun import Chebfun

# Plot colors
fig, ax = plt.subplots()
colors = []
for _ in range(32):
    line, = ax.plot(range(10), range(10))
    colors.append(line.get_color())
plt.close(fig)

# Choose observable
g = lambda x: np.abs(x - 1/3) + (x > 0.78) + np.sin(20*x)
gg = Chebfun.initfun(g, domain=[0, 1])
f = lambda x: g(x) / np.sqrt(np.sum(gg * np.conj(gg)))
ff = Chebfun.initfun(f, domain=[0, 1])

n = 15

# Setup the koopman matrix wrt Haar system
K = np.zeros((2**n, 2**n))
nId = np.zeros(2**n)
kId = np.zeros(2**n)
for j in range(1, n+1):
    nId = np.concatenate([nId, np.zeros(2**(j-1)) + j])
    kId = np.concatenate([kId, np.arange(2**(j-1))])

K[0, 0] = 1
for j in range(1, 2**(n-1)):
    I1 = np.where((nId == (nId[j] + 1)) & (kId == kId[j]))[0]
    if I1.size:
        K[I1, j] = 1/np.sqrt(2)
    I1 = np.where((nId == (nId[j] + 1)) & (kId == (2**nId[j] - (kId[j] + 1))))[0]
    if I1.size:
        K[I1, j] = -1/np.sqrt(2)

# Compute wavelet coefficients
xpts = np.linspace(1/(2**(n+1)), 1, 2**n)
coeffs, _ = pywt.wavedec(f(xpts), 'db1', level=int(np.log2(len(xpts))))
coeffs = coeffs / np.sqrt(len(xpts))
at = coeffs[0]
coeffs[0] = 0

# Compute the Fourier coefficients
MU = np.zeros(2*(2**n) + 1)
N = 1000
MU[N] = (coeffs @ coeffs) / (2*np.pi)
Fc = coeffs.copy()
for j in range(1, N+1):
    Fc = K @ Fc
    MU[N-j] = (coeffs @ Fc) / (2*np.pi)
    MU[N+j] = MU[N-j]
MU += abs(at)**2 / (2*np.pi)

# Spectral measures plot
N1 = 100
N2 = 1000
MU01 = Chebfun.initfun(MU[N-N1:N+N1+1], domain=[-np.pi, np.pi])
# Assuming you have a Python version of the MomentMeas function
# MU1 = MomentMeas(MU[N-N1:N+N1+1])
MU02 = Chebfun.initfun(MU[N-N2:N+N2+1], domain=[-np.pi, np.pi])
# MU2 = MomentMeas(MU[N-N2:N+N2+1])

# N=100 plot
plt.figure()
plt.semilogy(np.maximum(np.real(MU01), 0), color=colors[1], linewidth=1)
# plt.semilogy(np.maximum(np.real(MU1), 1e-16), color=colors[0], linewidth=1)
plt.ylim([1e-2, 50])
plt.legend(['no filter', 'with filter'], fontsize=14, loc='best')

# N=1000 plot
plt.figure()
plt.semilogy(np.maximum(np.real(MU02), 0), color=colors[1], linewidth=1)
# plt.semilogy(np.maximum(np.real(MU2), 1e-16), color=colors[0], linewidth=1)
plt.ylim([1e-2, 50])
plt.legend(['no filter', 'with filter'], fontsize=14, loc='best')

# Define functions
g = Chebfun.initfun(lambda x: np.cos(100*np.pi*x), domain=[0, 1])
f = lambda x: g(x) / np.sqrt(np.sum(g * np.conj(g)))

Nvec = np.arange(1, 13)
LL = np.zeros((len(Nvec), 10))
ct = 1

for n in Nvec:
    K = np.zeros((2**n, 2**n))
    nId = np.array([0])
    kId = np.array([0])
    for j in range(1, n+1):
        nId = np.concatenate([nId, np.zeros(2**(j-1)) + j])
        kId = np.concatenate([kId, np.arange(2**(j-1))])

    K[0, 0] = 1
    for j in range(1, 2**(n-1)):
        I1 = np.where((nId == (nId[j] + 1)) & (kId == kId[j]))[0]
        if I1.size:
            K[I1, j] = 1/np.sqrt(2)
        I1 = np.where((nId == (nId[j] + 1)) & (kId == (2**nId[j] - (kId[j] + 1))))[0]
        if I1.size:
            K[I1, j] = -1/np.sqrt(2)

    xpts = np.arange(1/(2**(n+1)), 1, 2**(-n))
    coeffs = pywt.wavedec([f(x) for x in xpts], 'db1', level=int(np.log2(len(xpts))))
    coeffs = np.concatenate(coeffs) / np.sqrt(len(xpts))
    at = coeffs[0]
    coeffs[0] = 0

    MU = np.zeros(2*(2**n) + 1)
    N = 2**n
    MU[N] = np.dot(coeffs, coeffs) / (2*np.pi)
    LL[ct-1, 0] = np.dot(coeffs, coeffs) / (2*np.pi) + abs(at)**2 / (2*np.pi)
    Fc = coeffs.copy()
    for j in range(1, N+1):
        Fc = K @ Fc
        MU[N-j] = np.dot(coeffs, Fc) / (2*np.pi)
        MU[N+j] = MU[N-j]
        if j < 11:
            LL[ct-1, j] = MU[N-j] + abs(at)**2 / (2*np.pi)
    MU += abs(at)**2 / (2*np.pi)
    
    ct += 1

# Compute the Fourier modes via ergodic sampling
N = 10
SVEC = 2**Nvec[:-2] * 10
MU_an = LL[-1, :N+1]

E = np.zeros((len(SVEC), 2))
ct = 1
for M in SVEC:
    MU2 = np.zeros(N+1)
    x1 = np.random.rand()
    FT = lambda x: np.maximum(np.minimum(2 * (2*x < 1) * x + (2*x >= 1) * 2 * (1-x), 1), 0)
    Ya = np.zeros(M)
    Ya[0] = f(x1)
    Yb = Ya.copy()
    xa = x1
    xb = x1

    for j in range(1, M):
        xa = FT(xa)
        xb = FT(xb) + np.random.rand() * 0.0001
        Ya[j] = f(xa)
        Yb[j] = f(xb)
    for j in range(N+1):
        MU2[j] = np.dot(Ya[:M-j], Ya[j:M]) / (M-j) / (2*np.pi)
    E[ct-1, 0] = np.max(np.abs(MU2 - MU_an))
    for j in range(N+1):
        MU2[j] = np.dot(Yb[:M-j], Yb[j:M]) / (M-j) / (2*np.pi)
    E[ct-1, 1] = np.max(np.abs(MU2 - MU_an))

    ct += 1

# Plot
fig, ax = plt.subplots()
ax.loglog(2**Nvec[:4]*10, np.max(np.abs(LL[:4, :] - np.tile(LL[-1, :], (4, 1))), axis=1), 'o-', color=colors[0], linewidth=2)
ax.loglog(SVEC, E[:len(SVEC), 1], 'o-', color=colors[1], linewidth=2)
ax.loglog(SVEC, SVEC**(-0.5), 'k:', linewidth=2)
ax.set_ylim([1e-16, 1])
ax.set_xlim([10, 10**4*2])
ax.set_fontsize(14)
plt.show()


plt.show()
