# DNN-resDMD
In this project, we will compare DNN-resDMD with kernel-resDMD.

Install the environment:

1. Navigate to a directory containing the file.
2. Create a new conda virtual environment(I will call it Koopman). Use following command:
```bash
conda create --name koopman python=3.8.18
3. Activate the environment:
```bash
conda activate Koopman
4. Now we will install all dependancies, with following command
```bash
conda env update --name koopman --file environment.yaml


Inside DNN_resDMD/examples, you can run several tests with only DNN-resDMD.

Inside kernel_resDMD,

solver_0: original EDMD-DL

solver_1: Directly optimize the loss function which is defined as eqn(3.2) defined in paper "Residual dynamic mode decomposition: robust and verified Koopmanism" or (4.6) in "Rigorous data-driven computation of spectral properties of Koopman operators for dynamical systems"(More stable)

solver_2: Similar to solver_3 but coded in terms of solver_1(Less stable)

solver_3: Multiply eigenvector matrix $V$ after $\Psi_Y$ and $\Psi_XK$ as eqn(2) in scratch paper(Preferred)

solver_4: $J_i(K, \Psi) = \mathbf{g_i^*}\left( L - K^*A - A^*K + K^*GK +\mu K^*K \right)\mathbf{g_i}$, which is eqn(1) in scratch paper(Not bad)

The entire code is modifed based on the package from https://github.com/MLDS-NUS/KoopmanDL
