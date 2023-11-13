# %%
import numpy as np
import sys
sys.path.append("../")
from algorithm.koopmanlib.dictionary import PsiNN

# %%
data_path = r'D:\Residual-Dynamic-Mode-Decomposition-main\Examples_gallery_1\ResDMD_datasets' 
import scipy.io
# temp = scipy.io.loadmat(data_path+'pendulum_data.mat')
temp = scipy.io.loadmat(data_path + '\\data_pendulum_100.mat')
X = temp['DATA_X']
Y = temp['DATA_Y']
print(X.shape)
del temp

len_all = X.shape[0]
data_x_train = X[:int(0.7*len_all),:]
data_x_valid = X[int(0.7*len_all)+1:,:]

data_y_train = Y[:int(0.7*len_all),:]
data_y_valid = Y[int(0.7*len_all)+1:,:]

data_train = [data_x_train, data_y_train]
data_valid = [data_x_valid, data_y_valid]

# %%
import scipy.io
import os

def connector_pendulum(n_psi_train, solver_index):
    basis_function = PsiNN(layer_sizes=[100,100,100], n_psi_train=n_psi_train)

    # Dynamically import the solver module based on solver_index
    solver_module = __import__(f"algorithm.koopmanlib.solver_{solver_index}", fromlist=['KoopmanDLSolver'])
    KoopmanDLSolver = getattr(solver_module, 'KoopmanDLSolver')

    # Using the dynamically imported solver
    solver = KoopmanDLSolver(dic=basis_function, 
                             target_dim=np.shape(data_train)[-1], 
                             reg=0.1)
    solver.build(data_train=data_train, 
                 data_valid=data_valid, 
                 epochs=500, 
                 batch_size=30000, 
                 lr=1e-4, 
                 log_interval=10, 
                 lr_decay_factor=.8)

    # Results from solver
    evalues = solver.eigenvalues
    efuns = solver.eigenfunctions(X)
    N_dict = np.shape(evalues)[0]
    Koopman_matrix_K = solver.K.numpy()
    Psi_X = solver.get_Psi_X().numpy()
    Psi_Y = solver.get_Psi_Y().numpy()

    # # SVD on Psi_X and Psi_Y
    # Psi_X_U, _, _ = np.linalg.svd(Psi_X/np.sqrt(Psi_X.shape[0]), full_matrices=False)
    # Psi_Y_U, _, _ = np.linalg.svd(Psi_Y/np.sqrt(Psi_Y.shape[0]), full_matrices=False)

    # Prepare data to save
    resDMD_DL_outputs = {
        'evalues': evalues,
        'efuns': efuns,
        'N_dict': N_dict,
        'Psi_X': Psi_X,
        'Psi_Y': Psi_Y,
        # 'Psi_X_U': Psi_X_U,
        # 'Psi_Y_U': Psi_Y_U,
        'K': Koopman_matrix_K,
    }

    # Saving the file
    save_path = 'D:\\Residual-Dynamic-Mode-Decomposition-main\\Examples_gallery_1\\ResDMD_datasets'
    filename = f'solver{solver_index}_outputs_{N_dict}basis.mat'
    full_path = os.path.join(save_path, filename)
    scipy.io.savemat(full_path, resDMD_DL_outputs)


