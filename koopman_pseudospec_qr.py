import numpy as np
from numpy import linalg as la
from numpy import *
import matplotlib.pyplot as plt

import numpy.linalg as la
from tqdm import tqdm



def  KoopPseudoSpecQR(PX,PY,W,z_pts):
    Q, R= la.qr(sqrt(W)*PX)
    C1 = np.matmul((np.sqrt(W)*PY),la.inv(R))
    
   
    L = np.dot(conj (C1.T), C1)
   
    G = np.eye(PX.shape[1])
    A = np.dot(conj(Q.T), C1)

    RES_list= []
    for jj in tqdm (arange (len (z_pts))):
        L_adjusted= L-z_pts[jj]*conj(A.T)-np.conj(z_pts[jj])*A+(np.abs(z_pts[jj])**2)*G
        curr_ev= la.eigvals(L_adjusted)
        min_abs_ind=  np.argmin (np.abs(curr_ev))
        
        curr_res=  np.emath.sqrt(np.real (curr_ev[min_abs_ind]))
        RES_list.append (curr_res)
    RES= array(RES_list)
    return RES