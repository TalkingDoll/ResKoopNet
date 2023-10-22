import numpy as np
from scipy.linalg import schur, qz
from scipy.sparse import issparse, eye

def IsomMeas(G, A, L, f, THETA, epsilon, least_squares='off', parallel='off', order=2):
    G = (G + G.T) / 2

    # compute the poles and residues
    delta = (2 * (np.arange(1, order + 1)) / (order + 1) - 1) * 1j + 1
    c, d = unitary_kern(delta, epsilon)

    # perform computation
    nu = np.zeros_like(THETA)
    
    if issparse(A) and issparse(G):
        v1 = G @ f
        v2 = v1
        v3 = A.T @ f
        
        for k in range(len(THETA)):
            for j in range(order):
                lambda_ = np.exp(1j * THETA[k]) * (1 + epsilon * delta[j])
                Ij = np.linalg.solve(A - lambda_ * G, v1)
                nu[k] -= np.real(1 / (2 * np.pi) * (c[j] * np.conj(lambda_) * Ij.T @ v2 + d[j] * v3.T @ Ij))
    else:
        if np.linalg.norm(G - np.eye(G.shape[0]), 'fro') < 1e-14 * np.linalg.norm(G, 'fro'):
            Q, S = schur(A)
            Z = Q
            T = eye(A.shape[0])
        else:
            S, T, Q, Z = qz(A, G, lwork=100)
            Q = Q.T
        
        v1 = T @ Z.T @ f
        v2 = T.T @ Q.T @ f
        v3 = S.T @ Q.T @ f
        
        for k in range(len(THETA)):
            for j in range(order):
                lambda_ = np.exp(1j * THETA[k]) * (1 + epsilon * delta[j])
                Ij = np.linalg.solve(S - lambda_ * T, v1)
                nu[k] -= np.real(1 / (2 * np.pi) * (c[j] * np.conj(lambda_) * Ij.T @ v2 + d[j] * v3.T @ Ij))
    
    return nu

def unitary_kern(Z, epsilon):
    m = len(Z)
    sigma = -np.conj(Z) / (1 + epsilon * np.conj(Z))
    V1 = np.zeros((m, m))
    V2 = np.zeros((m, m))
    
    for i in range(m):
        V1[:, i] = sigma**i
        V2[:, i] = Z**i
    
    rhs = np.eye(m, 1)[:, 0]
    c = np.linalg.lstsq(V1.T, rhs, rcond=None)[0]
    d = np.linalg.lstsq(V2.T, rhs, rcond=None)[0]
    
    return c, d
