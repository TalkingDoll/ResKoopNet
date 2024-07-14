import os
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from numpy import linalg as la
from numpy import *
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import warnings 

# Settings the warnings to be ignored 
warnings.filterwarnings('ignore') 

device = 'cuda'
torch.set_default_dtype(torch.float64)

class KoopmanNNTorch(nn.Module):
    def __init__(self, input_size, layer_sizes=[64, 64], n_psi_train=22, **kwargs):
        super(KoopmanNNTorch, self).__init__()
        self.layer_sizes = layer_sizes
        self.n_psi_train = n_psi_train  # Using n_psi_train directly, consistent with DicNN
        
        self.layers = nn.ModuleList()
        bias = False
        n_layers = len(layer_sizes)
        
        self.layers.append(nn.Linear(input_size, layer_sizes[0], bias=bias))
        for ii in arange(len(layer_sizes) - 1):
            self.layers.append(nn.Linear(layer_sizes[ii - 1], layer_sizes[ii], bias=True))
        self.layers.append(nn.Tanh())
        self.layers.append(nn.Linear(layer_sizes[n_layers - 1], n_psi_train, bias=True))
    
    def forward(self, x):
        in_x = x
        for layer in self.layers:
            x = layer(x)
        const_out = torch.ones_like(in_x[:, :1])  # print (const_out)
        x = torch.cat([const_out, in_x, x], dim=1)
        return x
    
    def generate_B(self, inputs):
            """
            Correctly generates the B matrix based on the inputs, using the proper attribute.
            """
            target_dim = inputs.shape[-1]
            # Use n_psi_train instead of n_dic_customized
            self.basis_func_number = self.n_psi_train + target_dim + 1
            self.B = np.zeros((self.basis_func_number, target_dim))
            for i in range(0, target_dim):
                self.B[i + 1][i] = 1
            return self.B

class KoopmanModelTorch(nn.Module):
    def __init__(self, dict_net, target_dim, k_dim):
        super(KoopmanModelTorch, self).__init__()
        self.dict_net = dict_net
        self.target_dim = target_dim
        self.k_dim = k_dim
        self.layer_K = nn.Linear(k_dim, k_dim, bias=False)
        self.layer_K.weight.requires_grad = False
        self.layer_eig= nn.Linear(k_dim, k_dim, bias=False)
        #self.layer_eig.weight.requires_grad = False
        self.layer_lambda_diag= nn.Linear(k_dim, k_dim, bias=False)
        #self.layer_lambda_diag.weight.requires_grad = False
    
    def forward(self, input_x, input_y):
        psi_x = self.dict_net.forward(input_x)
        psi_y = self.dict_net.forward(input_y)
        psi_x= torch.tensor(psi_x, dtype= torch.complex128)
        psi_y= torch.tensor(psi_y, dtype= torch.complex128)
        psi_x_v= self.layer_eig(psi_x)
        psi_y_v= self.layer_eig(psi_y)
        psi_x_v_lambda= self.layer_lambda_diag(psi_x_v)
        
        #psi_next = self.layer_K(psi_x)
        outputs_complex =psi_y_v- psi_x_v_lambda
        outputs= torch.cat ([outputs_complex.real, outputs_complex.imag], dim=1)
        return outputs

def fit_koopman_model(koopman_model, koopman_optimizer, checkpoint_file, xx_train, yy_train, xx_test, yy_test,
                      batch_size=32, lrate=1e-4, epochs=1000, initial_loss=10000):
    load_best = False
    xx_train_tensor = torch.DoubleTensor(float64(xx_train)).to(device)
    yy_train_tensor = torch.DoubleTensor(float64(yy_train)).to(device)
    xx_test_tensor = torch.DoubleTensor(float64(xx_test)).to(device)
    yy_test_tensor = torch.DoubleTensor(float64(yy_test)).to(device)
    train_dataset = torch.utils.data.TensorDataset(xx_train_tensor, yy_train_tensor)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
    val_dataset = torch.utils.data.TensorDataset(xx_test_tensor, yy_test_tensor)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    n_epochs = epochs
    best_loss = initial_loss
    mlp_mdl = koopman_model
    #optimizer = torch.optim.Adam(mlp_mdl.parameters(), lr=lrate, weight_decay=1e-5)
    optimizer = koopman_optimizer
    for param_group in optimizer.param_groups:
        param_group['lr'] = lrate
    criterion = nn.MSELoss()
    #criterion =complex_mse_loss

    mlp_mdl.train()
    val_loss_list = []

    for epoch in range(n_epochs):
        train_loss = 0.0
        for data, target in train_loader:
            optimizer.zero_grad()
            output = mlp_mdl(data, target)
            zeros_tensor = torch.zeros_like(output)
            loss = criterion(output, zeros_tensor)
            
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * data.size(0)
        train_loss = train_loss / len(train_loader.dataset)

        val_loss = 0.0
        with torch.no_grad():
            for data, target in val_loader:
                output_val = mlp_mdl(data, target)
                zeros_tensor = torch.zeros_like(output_val)
                loss = criterion(output_val, zeros_tensor)

                val_loss += loss.item() * data.size(0)
        val_loss = val_loss / len(val_loader.dataset)
        val_loss_list.append(val_loss)
        print('Epoch: {} \tTraining Loss: {:.6f} val loss: {:.6f}'.format(
            epoch + 1, train_loss, val_loss))
        if val_loss < best_loss:
            print('saving, val loss enhanced:', val_loss, best_loss)
            #torch.save(mlp_mdl.state_dict(), checkpoint_file)
            torch.save({
            'model_state_dict': mlp_mdl.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            }, checkpoint_file)
            best_loss = val_loss
            load_best = True

    if load_best:
        #mlp_mdl.load_state_dict(torch.load(checkpoint_file))
        checkpoint = torch.load(checkpoint_file)
        mlp_mdl.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        mlp_mdl.layer_K.requires_grad = False
        koopman_model = mlp_mdl
        koopman_optimizer= optimizer

    return val_loss_list, best_loss

class KoopmanSolverTorch(object):
    '''
    Build the Koopman solver

    This part represents a Koopman solver that can be used to build and solve Koopman operator models.

    Attributes:
        dic (class): The dictionary class used for Koopman operator approximation.
        dic_func (function): The dictionary functions used for Koopman operator approximation.
        target_dim (int): The dimension of the variable of the equation.
        reg (float, optional): The regularization parameter when computing K. Defaults to 0.0.
        psi_x (None): Placeholder for the feature matrix of the input data.
        psi_y (None): Placeholder for the feature matrix of the output data.
    '''

    def __init__(self, dic, target_dim, reg=0.0, checkpoint_file='example_koopman_net001.torch'):
        """Initializer

        :param dic: dictionary
        :type dic: class
        :param target_dim: dimension of the variable of the equation
        :type target_dim: int
        :param reg: the regularization parameter when computing K, defaults to 0.0
        :type reg: float, optional
        """
        self.dic = dic  # dictionary class
        self.dic_func = dic.forward  # dictionary functions
        self.target_dim = target_dim
        self.reg = reg
        self.psi_x = None
        self.psi_y = None
        self.checkpoint_file = checkpoint_file

    def separate_data(self, data):
        data_x = data[0]
        data_y = data[1]
        return data_x, data_y

    def build(self, data_train):
        # Separate data
        self.data_train = data_train
        self.data_x_train, self.data_y_train = self.separate_data(self.data_train)

        # Compute final information
        self.compute_final_info(reg_final=0.0)

    def compute_final_info(self, reg_final):
        # Compute K
        self.K = self.compute_K(self.dic_func, self.data_x_train, self.data_y_train, reg=reg_final)
        self.K_np = self.K.detach().cpu().numpy()
        self.eig_decomp(self.K_np)
        self.compute_mode()

    def eig_decomp(self, K):
        """ eigen-decomp of K """
        self.eigenvalues, self.eigenvectors = la.eig(K)
        idx = self.eigenvalues.real.argsort()[::-1]
        self.eigenvalues = self.eigenvalues[idx]
        self.eigenvectors = self.eigenvectors[:, idx]
        self.eigenvectors_inv = la.inv(self.eigenvectors)

    def eigenfunctions(self, data_x):
        """ estimated eigenfunctions """
        data_x = torch.DoubleTensor(data_x).to(device)
        psi_x = self.dic_func(data_x)
        psi_x = psi_x.detach().cpu().numpy()
        val = np.matmul(psi_x, self.eigenvectors)
        return val

    def compute_mode(self):
        self.basis_func_number = self.K.shape[0]

        # Form B matrix
        self.B = self.dic.generate_B(self.data_x_train)

        # Compute modes
        self.modes = np.matmul(self.eigenvectors_inv, self.B).T
        return self.modes

    def calc_psi_next(self, data_x, K):
        psi_x = self.dic_func(data_x)
        psi_next = tf.matmul(psi_x, K)
        return psi_next

    def compute_K(self, dic, data_x, data_y, reg):
        data_x = torch.DoubleTensor(data_x).to(device)
        data_y = torch.DoubleTensor(data_y).to(device)
        psi_x = dic(data_x)
        psi_y = dic(data_y)
        
        # Compute Psi_X and Psi_Y
        self.Psi_X = dic(data_x)
        self.Psi_Y = dic(data_y)
        
        psi_xt = psi_x.T
        idmat = torch.eye(psi_x.shape[-1]).to(device)
        xtx_inv = torch.linalg.pinv(reg * idmat + torch.matmul(psi_xt, psi_x))
        xty = torch.matmul(psi_xt, psi_y)
        self.K_reg = torch.matmul(xtx_inv, xty)
        return self.K_reg

    def get_Psi_X(self):
        return self.Psi_X

    def get_Psi_Y(self):
        return self.Psi_Y

    def build_model(self):
        self.koopman_model = KoopmanModelTorch(dict_net=self.dic, target_dim=self.target_dim, k_dim=self.K.shape[0]).to(device)

    def train_psi(self, koopman_model, koopman_optimizer, epochs, lr, initial_loss=10000):
        data_x_val, data_y_val = self.separate_data(self.data_valid)
        psi_losses, best_psi_loss = fit_koopman_model(self.koopman_model, koopman_optimizer, self.checkpoint_file, self.data_x_train,
                                                      self.data_y_train, data_x_val, data_y_val, self.batch_size,
                                                      lrate=lr, epochs=epochs, initial_loss=initial_loss)
        return psi_losses, best_psi_loss

    def build(self, data_train, data_valid, epochs, batch_size, lr, log_interval, lr_decay_factor):
        """Train Koopman model and calculate the final information,
        such as eigenfunctions, eigenvalues and K.
        For each outer training epoch, the koopman dictionary is trained
        by several times (inner training epochs), and then compute matrix K.
        Iterate the outer training.

        :param data_train: training data
        :type data_train: [data at the current time, data at the next time]
        :param data_valid: validation data
        :type data_valid: [data at the current time, data at the next time]
        :param epochs: the number of the outer epochs
        :type epochs: int
        :param batch_size: batch size
        :type batch_size: int
        :param lr: learning rate
        :type lr: float
        :param log_interval: the patience of learning decay
        :type log_interval: int
        :param lr_decay_factor: the ratio of learning decay
        :type lr_decay_factor: float
        """
        # Separate training data
        self.data_train = data_train
        self.data_x_train, self.data_y_train = self.separate_data(self.data_train)

        self.data_valid = data_valid

        self.batch_size = batch_size
        self.K = self.compute_K(self.dic_func, self.data_x_train, self.data_y_train, self.reg)
        
        # Build the Koopman DL model
        self.build_model()

        losses = []
        curr_lr = lr
        curr_last_loss = 10000
        self.koopman_optimizer= torch.optim.Adam(self.koopman_model.parameters(), lr=lr, weight_decay=1e-5)
        for ii in arange(epochs):
            start_time = time.time()
            print(f"Outer Epoch {ii+1}/{epochs}")
            
            # One step for computing K
            self.K = self.compute_K(self.dic_func, self.data_x_train, self.data_y_train, self.reg)
            self.compute_final_info(reg_final=0.01)
            with torch.no_grad():
                self.koopman_model.layer_K.weight.data = self.K
                self.koopman_model.layer_eig.weight.data= torch.tensor(self.eigenvectors, dtype= torch.complex128).to(device)
                self.koopman_model.layer_lambda_diag.weight.data = torch.tensor(np.diag(self.eigenvalues), dtype= torch.complex128).to(device)
               
            # Two steps for training PsiNN
            curr_losses, curr_best_loss = self.train_psi(self.koopman_model, self.koopman_optimizer, epochs=4, lr=curr_lr, initial_loss=curr_last_loss)
            
            if curr_last_loss > curr_best_loss:
                curr_last_loss = curr_best_loss

            if ii % log_interval == 0:
                losses.append(curr_losses[-1])

                # Adjust learning rate:
                if len(losses) > 2:
                    if losses[-1] > losses[-2]:
                        print("Error increased. Decay learning rate")
                        curr_lr = lr_decay_factor * curr_lr

            end_time = time.time()
            epoch_time = end_time - start_time
            print(f"Epoch {ii+1} time: {epoch_time:.2f} seconds")

        # Compute final information
        checkpoint = torch.load(self.checkpoint_file)
        self.koopman_model.load_state_dict(checkpoint['model_state_dict'])
        self.koopman_optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.compute_final_info(reg_final=0.01)