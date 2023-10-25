# DNN-resDMD
resDMD combined with DNN

solver_0: original EDMD-DL

solver_1: Directly optimize the loss function which is defined as eqn(3.2) defined in paper "Residual dynamic mode decomposition: robust and verified Koopmanism" or (4.6) in "Rigorous data-driven computation of spectral properties of Koopman operators for dynamical systems"(More stable)

solver_2: Similar to solver_3 but coded in terms of solver_1(Less stable)

solver_3: Multiply eigenvector matrix $V$ after $\Psi_Y$ and $\Psi_XK$ as eqn(2) in scratch paper(Preferred)

solver_4: $J_i(K, \Psi) = \mathbf{g_i^*}\left( L - K^*A - A^*K + K^*GK +\mu K^*K \right)\mathbf{g_i}$, which is eqn(1) in scratch paper(Not bad)

The entire code is modifed based on the package from https://github.com/MLDS-NUS/KoopmanDL
