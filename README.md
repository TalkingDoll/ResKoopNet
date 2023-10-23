# DNN-resDMD
resDMD combined with DNN

solver_0: original EDMD-DL
solver_1: Eqn(3.2) defined in paper "Residual dynamic mode decomposition: robust and verified Koopmanism" or (4.6) in "Rigorous data-driven computation of spectral
properties of Koopman operators for dynamical systems"

solver_2: Similar to solver_3 but coded in terms of solver_1

solver_3: Multiply eigenvector matrix $V$ after $\Psi_Y$ and $\Psi_XK$ in scratch paper

solver_4: $J_i(K, \Psi) = \mathbf{g_i^*}\left( L - K^*A - A^*K + K^*GK +\mu K^*K \right)\mathbf{g_i}$
