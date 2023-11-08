# DNN-resDMD
In this project, we will compare DNN-resDMD and kernel-resDMD.

- Install the environment:

1. Navigate to a directory containing the file.
2. Create a new conda virtual environment(e.g., Koopman). Use following command:
```bash
conda create --name koopman python=3.8.18
```

3. Activate the environment:
```bash
conda activate Koopman
```

4. Now we will install the environment file:
```bash
conda env update --name koopman --file environment.yaml
```



- Inside DNN_resDMD/examples, you can run several tests with only DNN-resDMD. The followings are different solver scripts defining the loss function in different ways:

>solver_0: original EDMD-DL

>solver_1: Directly optimize the loss function which is defined as eqn(3.2) defined in paper "Residual dynamic mode decomposition: robust and verified Koopmanism" or (4.6) in "Rigorous data-driven computation of spectral properties of Koopman operators for dynamical systems"(More stable)

>solver_2: Multiply eigenvector matrix using manually defined matrices $G$, $A$ and $K$

>solver_3: Mutiply the eigenvector matrix using the default 'Layer_K'

>solver_4: $J_i(K, \Psi) = \mathbf{g_i^*}\left( L - K^*A - A^*K + K^*GK +\mu K^*K \right)\mathbf{g_i}$, which is eqn(1) in scratch paper(Not bad)

>solver_5: Assuming the eigenfunction in the denominator is normalized, the numerator $g^* M g$ is a quadratic form and is symmetric, so it's same as the operator norm.

- Inside kernel_resDMD,

The entire code is modifed based on the package from https://github.com/MLDS-NUS/KoopmanDL


