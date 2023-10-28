import numpy as np

def kernel_ResDMD(Xa, Ya, Xb, Yb, **kwargs):
    """
    This function applies kernelized ResDMD.
    
    Args:
        Xa, Ya: Data matrices used in kernel_EDMD to form dictionary.
        Xb, Yb: Data matrices used in ResDMD.
        
        Optional Args:
        N: Size of computed dictionary, default is number of data points for kernel EDMD.
        kernel_f: Kernel function, default is normalized Gaussian.
        Parallel: Whether to use parallel operations, default is False.
        Sketch: Whether to use sketching, default is False.
        s: Size of sketching.
        cut_off: Stability parameter for SVD, default is 10^(-12).
        Y2: Additional data matrix for stochastic version.

    Returns:
        PSI_x, PSI_y, PSI_y2: Matrices for ResDMD.
    """
    
    # Default values for optional parameters
    N = Xa.shape[1]
    d = np.mean(np.linalg.norm(Xa - np.mean(Xa, axis=1, keepdims=True), axis=0))
    kernel_f = lambda x, y: np.exp(-np.linalg.norm(x-y)/d)
    Parallel = False
    Sketch = False
    cut_off = 1e-12
    s = max(int(5 * np.sqrt(Xa.shape[1] + Xb.shape[1]) * np.log(Xa.shape[1] + Xb.shape[1])), 5000)
    Y2 = None
    
    # Overwrite defaults with provided arguments
    for key, value in kwargs.items():
        if key == "N": N = value
        if key == "kernel_f": kernel_f = value
        if key == "Parallel": Parallel = value
        if key == "Sketch": Sketch = value
        if key == "cut_off": cut_off = value
        if key == "s": s = value
        if key == "Y2": Y2 = value

    M1 = Xa.shape[1]
    M2 = Xb.shape[1]

    if not Sketch:
        G1 = np.zeros((M1, M1))
        A1 = np.zeros((M1, M1))
        for i in range(M1):
            for j in range(M1):
                G1[i, j] = kernel_f(Xa[:, i], Xa[:, j])
                A1[i, j] = kernel_f(Ya[:, i], Xa[:, j])
    else:
        Z = np.sqrt(2/d**2) * np.random.randn(Xa.shape[0], s)
        TH = 2 * np.pi * np.random.rand(s, 1)
        psi_xa = np.sqrt(2/s) * np.cos(TH + Z.T @ Xa)
        psi_ya = np.sqrt(2/s) * np.cos(TH + Z.T @ Ya)
        G1 = psi_xa.T @ psi_xa
        A1 = psi_ya.T @ psi_xa

    # Post processing
    eigvals, U = np.linalg.eig(G1 + cut_off * np.eye(M1) * np.linalg.norm(G1))
    eigvals[eigvals < cut_off] = 0
    SIG = np.sqrt(np.diag(eigvals))
    SIG_dag = np.linalg.pinv(SIG)

    K_hat = SIG_dag @ U.T @ A1 @ U @ SIG_dag
    eigvals1, U1 = np.linalg.eig(K_hat)

    # Filter eigenvalues based on cut_off and get corresponding eigenvectors
    I = np.where(np.abs(eigvals1) > cut_off)[0]
    if len(I) > N:
        I = np.argsort(np.abs(eigvals1))[::-1][:N]
    else:
        N = len(I)

    P, _, _ = np.linalg.svd(U1[:, I], full_matrices=False)
    P = U @ SIG_dag @ P

    # Compute matrices for ResDMD
    PSI_x = np.zeros((M2, M1))
    PSI_y = np.zeros((M2, M1))
    PSI_y2 = np.zeros((M2, M1)) if Y2 is not None else None

    for ii in range(M2):
        for jj in range(M1):
            PSI_x[ii, jj] = kernel_f(Xb[:, ii], Xa[:, jj])
            PSI_y[ii, jj] = kernel_f(Yb[:, ii], Xa[:, jj])
            if Y2 is not None:
                PSI_y2[ii, jj] = kernel_f(Y2[:, ii], Xa[:, jj])

    PSI_x = PSI_x @ P
    PSI_y = PSI_y @ P
    if Y2 is not None:
        PSI_y2 = PSI_y2 @ P

    return PSI_x, PSI_y, PSI_y2

