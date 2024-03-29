import os

# from autograd import jacobian, hessian
import tensorflow as tf
from tensorflow.keras.layers import Dense, Layer, Concatenate, Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.python.ops.numpy_ops import np_config

import numpy as np


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.keras.backend.set_floatx('float64')


class KoopmanNN(tf.keras.layers.Layer):
    def __init__(self, layer_sizes=[64, 64], n_psi_train=22, **kwargs):
        super(KoopmanNN, self).__init__(**kwargs)
        self.layer_sizes = layer_sizes
        self.n_psi_train = n_psi_train  # Using n_psi_train directly
        self.input_layer = tf.keras.layers.Dense(layer_sizes[0], use_bias=False)
        self.hidden_layers = [tf.keras.layers.Dense(size, activation='tanh') for size in layer_sizes]
        self.output_layer = tf.keras.layers.Dense(n_psi_train)

    def call(self, inputs):
        x = self.input_layer(inputs)
        for layer in self.hidden_layers:
            x = layer(x)
        psi_x_train = self.output_layer(x)
        
        # Directly integrating the generation of the constant and concatenation
        constant = tf.ones_like(inputs[:, :1])
        outputs = tf.keras.layers.Concatenate()([constant, inputs, psi_x_train])
        return outputs

    def get_config(self):
        config = super(KoopmanNN, self).get_config()
        config.update({
            'layer_sizes': self.layer_sizes,
            'n_psi_train': self.n_psi_train
        })
        return config
    
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


class KoopmanSolver(object):


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
        # Compute Psi_X and Psi_Y
        self.Psi_X = dic(data_x)
        self.Psi_Y = dic(data_y)
        psi_xt = tf.transpose(psi_x)
        idmat = tf.eye(psi_x.shape[-1], dtype='float64')
        xtx_inv = tf.linalg.pinv(reg * idmat + tf.matmul(psi_xt, psi_x))
        xty = tf.matmul(psi_xt, psi_y)
        self.K_reg = tf.matmul(xtx_inv, xty)
        return self.K_reg

    def get_Psi_X(self):
        return self.Psi_X

    def get_Psi_Y(self):
        return self.Psi_Y


    '''
    Build the Koopman model with dictionary learning
    '''

    def build_model(self):
        """Build model with trainable dictionary

        The loss function is ||Psi(y) - K Psi(x)||^2 .

        """
        inputs_x = Input((self.target_dim,))
        inputs_y = Input((self.target_dim,))

        self.psi_x = self.dic_func(inputs_x)
        self.psi_y = self.dic_func(inputs_y)

        Layer_K = Dense(units=self.psi_y.shape[-1],
                        use_bias=False,
                        name='Layer_K',
                        trainable=False)
        psi_next = Layer_K(self.psi_x)

        outputs = psi_next - self.psi_y
        model = Model(inputs=[inputs_x, inputs_y], outputs=outputs)
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

    def get_derivatives(self, inputs):
        # Compute the first and second derivatives
        inputs = tf.convert_to_tensor(inputs)
        with tf.GradientTape() as tape2:
            with tf.GradientTape() as tape1:
                tape1.watch(inputs)
                tape2.watch(inputs)
                outputs = self.dic(inputs)
            first_derivatives = tape1.batch_jacobian(outputs, inputs)
        second_derivatives = tape2.batch_jacobian(first_derivatives, inputs)
        # print(outputs.shape, first_derivatives.shape, second_derivatives.shape)
        # return (outputs, first_derivatives, second_derivatives)
        return (first_derivatives, second_derivatives)

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
            print(f"Outer Epoch {i+1}/{epochs}")
            # One step for computing K
            self.K = self.compute_K(self.dic_func,
                                    self.data_x_train,
                                    self.data_y_train,
                                    self.reg)
            self.model.get_layer('Layer_K').weights[0].assign(self.K)

            # Two steps for training PsiNN
            self.history = self.train_psi(self.model, epochs=2)

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
