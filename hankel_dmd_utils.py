import os
import numpy as np
from numpy import linalg as la
from numpy import *
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from scipy.io import loadmat
import os
import scipy.io as sio
import numpy.linalg as la
from tqdm import tqdm
from scipy.special import hermite
from scipy.special import factorial 
from scipy.linalg import toeplitz as toep



# def exact_dmd(gtot, thrshhld, window, ndsets):
#     nrws, nclmns = gtot.shape
#     gm = np.zeros((nrws, ndsets * (window - 1)), dtype=np.float64)
#     gp = np.zeros((nrws, ndsets * (window - 1)), dtype=np.float64)
#     # Perform DMD method.  Note, we need to be careful about how we break the concantenated Hankel matrix apart.
#     for ll in tqdm (range(ndsets)):
#         gm[:, ll * (window - 1):(ll + 1) * (window - 1)] = gtot[:, ll * window:(ll + 1) * window - 1]
#         gp[:, ll * (window - 1):(ll + 1) * (window - 1)] = gtot[:, 1 + ll * window:(ll + 1) * window]
#     u, s, vh = la.svd(gm, full_matrices=False)
#     sm = np.max(s)
#     indskp = np.log10(s / sm) > -thrshhld #indices to keep
#     sr = s[indskp]
#     ur = u[:, indskp]
#     v = np.conj(vh.T)
#     vr = v[:, indskp]
#     kmat = gp @ vr @ np.diag(1. / sr) @ np.conj(ur.T)#Koopman matrix
#     evals, evecs = la.eig(kmat)
    
#     return evals, evecs, kmat

# Assuming V1_trunc is the first 100 columns of V1 from the initial SVD
def exact_dmd(gtot, thrshhld, window, ndsets):
    nrws, nclmns = gtot.shape
    gm = np.zeros((nrws, ndsets * (window - 1)), dtype=np.float64)
    gp = np.zeros((nrws, ndsets * (window - 1)), dtype=np.float64)
    
    # Hankel matrix construction
    for ll in range(ndsets):
        gm[:, ll * (window - 1):(ll + 1) * (window - 1)] = gtot[:, ll * window:(ll + 1) * window - 1]
        gp[:, ll * (window - 1):(ll + 1) * (window - 1)] = gtot[:, 1 + ll * window:(ll + 1) * window]

    # Apply SVD on gm 
    u, s, vh = np.linalg.svd(gm, full_matrices=False)
    
    # Select singular value based on threshold
    sm = np.max(s)
    indskp = np.log10(s / sm) > -thrshhld  # preserved index
    sr = s[indskp]
    ur = u[:, indskp]
    v = np.conj(vh.T)
    vr = v[:, indskp]

    # Compute Koopman matrix
    kmat = gp @ vr @ np.diag(1. / sr) @ np.conj(ur.T)  # Koopman matrix
    evals, evecs = np.linalg.eig(kmat)  # Compute eigenvalue and eigenvector
    
    # Compute Koopman mode
    koopman_modes = ur @ evecs  # Compute Koopman mode

    return evals, evecs, koopman_modes, kmat



def hankel_matrix(tseries, window):
    NT = np.size(tseries)
    nobserves = NT - (window - 1)
    tcol = tseries[:nobserves]
    trow = tseries[(nobserves - 1):]
    hmat = np.flipud(toep(tcol[::-1], trow))
    hmatt = hmat.T
    sclfac = np.linalg.norm(hmatt[:, -1])
    return hmat, sclfac


# def hankel_dmd(raw_data, n_traj, traj_len,  window, thrshhld):
#     #raw data - the array of size (N_traj*traj_len, nobs)
#     #it is assumed that the data contain N_traj trajectories stacked consequently
#     #n_traj -- number of trajectories/ data points
#     #traj_len -- length of each trajectory/ number of evaluations of each data points
#     NT = traj_len
#     nclmns = NT - (window - 1) #number of time delays
#     nobs = raw_data.shape[1] #number of observables
#     hankel_mats = np.zeros((nclmns * nobs, window * n_traj), dtype=np.float64)
#     for ll in tqdm (range(n_traj)):
#         for jj in range(nobs):
#             tseries =raw_data[traj_len*ll: traj_len*(ll+1), jj]
#             hmat, sclfac = hankel_matrix(tseries, window)
#             if jj == 0:
#                 usclfac = sclfac
#             hankel_mats[jj * nclmns:(jj + 1) * nclmns, ll * window:(ll + 1) * window] = usclfac / sclfac * hmat
#     return exact_dmd(hankel_mats, thrshhld, window, n_traj)
def hankel_dmd(raw_data, n_traj, traj_len, window, thrshhld):
    NT = traj_len
    nclmns = NT - (window - 1)  # number of time delays
    nobs = raw_data.shape[1]  # number of observables
    hankel_mats = np.zeros((nclmns * nobs, window * n_traj), dtype=np.float64)
    
    for ll in range(n_traj):
        for jj in range(nobs):
            tseries = raw_data[traj_len * ll: traj_len * (ll + 1), jj]
            hmat, sclfac = hankel_matrix(tseries, window)
            if jj == 0:
                usclfac = sclfac
            hankel_mats[jj * nclmns:(jj + 1) * nclmns, ll * window:(ll + 1) * window] = usclfac / sclfac * hmat

    # Recall exact_dmd to compute Koopman mode and matrix
    return exact_dmd(hankel_mats, thrshhld, window, n_traj)







