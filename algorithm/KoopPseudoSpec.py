import numpy as np
from scipy.linalg import eigh, eig
from scipy.sparse.linalg import eigs
from concurrent.futures import ThreadPoolExecutor

def KoopPseudoSpec(G, A, L, z_pts, parallel='off', z_pts2=None, reg_param=1e-14):
    """
    This function computes pseudospectrum of K (currently written for dense matrices).
    """
    # Safeguards
    G = (G + G.T) / 2
    L = (L + L.T) / 2

    # Compute SQ
    eigvals, eigvecs = eig(G + np.linalg.norm(G) * reg_param * np.eye(G.shape[0]))
    eigvals[eigvals > 0] = np.sqrt(1.0 / eigvals[eigvals > 0])
    SQ = eigvecs @ np.diag(eigvals) @ eigvecs.T

    # Compute RES
    LL = len(z_pts)
    RES = np.zeros(LL)

    def compute_res(jj):
        val = eigs(SQ @ (L - z_pts[jj] * np.conj(A.T) - np.conj(z_pts[jj]) * A + abs(z_pts[jj])**2 * G) @ SQ, k=1, which='SM')[0]
        return np.sqrt(val.real)

    if parallel == 'on':
        with ThreadPoolExecutor() as executor:
            RES = list(executor.map(compute_res, range(LL)))
    else:
        for jj in range(LL):
            RES[jj] = compute_res(jj)

    # Compute RES2 and V2
    RES2 = []
    V2 = []

    if z_pts2 is not None:
        RES2 = np.zeros(len(z_pts2))
        V2 = np.zeros((G.shape[0], len(z_pts2)))

        def compute_res2_v2(jj):
            vals, vecs = eigs(SQ @ (L - z_pts2[jj] * np.conj(A.T) - np.conj(z_pts2[jj]) * A + abs(z_pts2[jj])**2 * G) @ SQ, k=1, which='SM')
            return np.sqrt(vals.real), vecs

        if parallel == 'on':
            results = list(ThreadPoolExecutor().map(compute_res2_v2, range(len(z_pts2))))
            for jj, (res, vec) in enumerate(results):
                RES2[jj] = res
                V2[:, jj] = vec
        else:
            for jj in range(len(z_pts2)):
                temp_x = compute_res2_v2(jj)
                print(temp_x[0].shape, temp_x[1].shape)
                RES2[jj], V2[:, jj] = temp_x[0], np.reshape(temp_x[1], (300,))

        V2 = SQ @ V2

    return RES, RES2, V2
