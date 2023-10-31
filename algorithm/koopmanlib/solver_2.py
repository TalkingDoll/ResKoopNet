from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Layer, Dense
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
import numpy as np
import tensorflow as tf
tf.keras.backend.set_floatx('float64')


class KoopmanGeneralSolver(object):
    '''
    Build the Koopman solver
    '''

    def __init__(self, dic, target_dim, reg=0.0):
        """Initializer

        :param dic: dictionary
        :type dic: class
        :param target_dim: dimension of the variable of the equation
        :type target_dim: int
        :param reg: the regularization parameter when computing K, defaults to 0.0
        :type reg: float, optional
        """
        self.dic = dic  # dictionary class
        self.dic_func = dic.call  # dictionary functions
        self.target_dim = target_dim
        self.reg = reg
        self.psi_x = None
        self.psi_y = None

    def separate_data(self, data):
        data_x = data[0]
        data_y = data[1]
        return data_x, data_y

    def build(self, data_train):
        # Separate data
        self.data_train = data_train
        self.data_x_train, self.data_y_train = self.separate_data(
            self.data_train)

        # Compute final information
        self.compute_final_info(reg_final=0.0)

    def compute_final_info(self, reg_final):
        # Compute K
        self.K = self.compute_K(self.dic_func,
                                self.data_x_train,
                                self.data_y_train,
                                reg=reg_final)
        self.eig_decomp(self.K)
        self.compute_mode()

    def eig_decomp(self, K):
        """ eigen-decomp of K """
        self.eigenvalues, self.eigenvectors = np.linalg.eig(K)
        idx = self.eigenvalues.real.argsort()[::-1]
        self.eigenvalues = self.eigenvalues[idx]
        self.eigenvectors = self.eigenvectors[:, idx]
        self.eigenvectors_inv = np.linalg.inv(self.eigenvectors)

    def eigenfunctions(self, data_x):
        """ estimated eigenfunctions """
        psi_x = self.dic_func(data_x)
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

    def predict(self, x0, traj_len):
        """ predict the trajectory """
        traj = [x0]
        for _ in range(traj_len - 1):
            x_curr = traj[-1]
            efunc = self.eigenfunctions(x_curr)
            x_next = np.matmul(self.modes, (self.eigenvalues * efunc).T)
            traj.append((x_next.real).T)
        traj = np.transpose(np.stack(traj, axis=0), [1, 0, 2])
        return traj.squeeze()

    def compute_K(self, dic, data_x, data_y, reg):
        psi_x = dic(data_x)
        psi_y = dic(data_y)
        psi_xt = tf.transpose(psi_x)
        idmat = tf.eye(psi_x.shape[-1], dtype='float64')
        xtx_inv = tf.linalg.pinv(reg * idmat + tf.matmul(psi_xt, psi_x))
        xty = tf.matmul(psi_xt, psi_y)
        self.K_reg = tf.matmul(xtx_inv, xty)
        return self.K_reg


class KoopmanDLSolver(KoopmanGeneralSolver):
    '''
    Build the Koopman model with dictionary learning
    '''

    def build_model(self):
        """Build model with trainable dictionary

        The loss function is ||Psi(y) - Psi(x)K||^2.

        """
        inputs_x = Input((self.target_dim,))
        inputs_y = Input((self.target_dim,))

        self.psi_x = self.dic_func(inputs_x)
        self.psi_y = self.dic_func(inputs_y)

        Layer_K = Dense(units=self.psi_y.shape[-1],
                        use_bias=False,
                        name='Layer_K',
                        trainable=False)

        # Calculation of residuals as per ResDMD paper
        G = tf.matmul(self.psi_x, self.psi_x, transpose_a=True) / self.batch_size # Weighted matrix G: \Psi_X^* W \Psi_X
        idmat = tf.eye(self.psi_x.shape[-1], dtype='float64')
        xtx_inv = tf.linalg.pinv(self.reg * idmat + G)
        A = tf.matmul(self.psi_x, self.psi_y, transpose_a=True) / self.batch_size # Weighted matrix A: \Psi_X^* W \Psi_Y
        K = tf.matmul(xtx_inv, A)

        eigen_values, eigen_vectors = tf.eig(K)
        
        # Matrix multiplication is only possible between same data types. So floats need to be converted to complex128 data types using tf.cast.
        term_1 = tf.matmul(tf.cast(self.psi_y, tf.complex128), eigen_vectors) # Psi_Y V
        term_2 = tf.matmul(self.psi_x, K)                                     # Psi_X K
        term_3 = tf.matmul(tf.cast(term_2, tf.complex128), eigen_vectors) # Psi_X K V
        term_4 = tf.matmul(tf.cast(K, tf.complex128), eigen_vectors)      # K V

        # Formula from the document
        resdmd_residuals = tf.norm(term_1 - term_3)**2 + self.reg*tf.norm(term_4)**2    # || Psi_Y . V - Psi_X. K . V ||_F + reg . ||K. V ||^2
        
        model = Model(inputs=[inputs_x, inputs_y], outputs=resdmd_residuals)
        return model

    def train_psi(self, model, epochs):
        """Train the trainable part of the dictionary

        :param model: koopman model
        :type model: model
        :param epochs: the number of training epochs before computing K for each inner training epoch
        :type epochs: int
        :return: history
        :rtype: history callback object
        """
        history = model.fit(
            x=self.data_train,
            y=self.zeros_data_y_train,
            epochs=epochs,
            validation_data=(
                self.data_valid,
                self.zeros_data_y_valid),
            batch_size=self.batch_size,
            verbose=1)
        return history
    
    def get_basis(self, x, y):
        """Returns the dictionary(matrix) consisting of basis.

        :param x: array of snapshots
        :type x: numpy array
        :param y:array of snapshots
        :type y: numpy array
        """
        psi_x = self.dic_func(x)
        # Calculate column norms
        psi_x_column_norms = np.linalg.norm(psi_x, axis=0)
        # Handle the case where norm is zero
        psi_x_column_norms[psi_x_column_norms == 0] = 1
        psi_x_normalized = psi_x / psi_x_column_norms

        # Repeat the steps for psi_y
        psi_y = self.dic_func(y)
        # Calculate column norms
        psi_y_column_norms = np.linalg.norm(psi_y, axis=0)
        # Handle the case where norm is zero
        psi_y_column_norms[psi_y_column_norms == 0] = 1
        psi_y_normalized = psi_y / psi_y_column_norms

        return psi_x_normalized, psi_y_normalized
    
    def build(
            self,
            data_train,
            data_valid,
            epochs,
            batch_size,
            lr,
            log_interval,
            lr_decay_factor):
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
        self.data_x_train, self.data_y_train = self.separate_data(
            self.data_train)

        self.data_valid = data_valid
        self.zeros_data_y_train = tf.zeros_like(
            self.dic_func(self.data_y_train))
        self.zeros_data_y_valid = tf.zeros_like(
            self.dic_func(self.data_valid[1]))
        self.batch_size = batch_size

        # Build the Koopman DL model
        self.model = self.build_model()

        # Compile the Koopman DL model
        opt = Adam(lr)
        self.model.compile(optimizer=opt, loss='mse')

        # Training Loop
        losses = []
        for i in range(epochs):
            # One step for computing K
            self.K= self.compute_K(self.dic_func,
                                    self.data_x_train,
                                    self.data_y_train,
                                    self.reg)
            # self.model.get_layer('Layer_K').weights[0].assign(self.K)

            # Two steps for training PsiNN
            self.history = self.train_psi(self.model, epochs=2)

            print('number of the outer loop:', i)
            if i % log_interval == 0:
                losses.append(self.history.history['loss'][-1])

                # Adjust learning rate:
                if len(losses) > 2:
                    if losses[-1] > losses[-2]:
                        print("Error increased. Decay learning rate")
                        curr_lr = lr_decay_factor * self.model.optimizer.lr
                        self.model.optimizer.lr = curr_lr

        # Compute final information
        self.compute_final_info(reg_final=0.01)
