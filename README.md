# ResKoopNet instructions

---

## Experiments 1: Pendulum

I have provided two versions of Pendulum system using ResKoopNet and a comparison test with other 4 methods:

1. The `pendulum.m` file is used to generate pendulum data.
2. **TensorFlow Version**: `pendulum_reskoopnet_tf.ipynb`
3. **PyTorch Version**: `pendulum_reskoopnet_torch.ipynb`
4. Comparison test: `pendulum_edmd_reskoopnet_hankeldmd_edmddl.ipynb`
5. Algorithm of Hankel-DMD: `hankel_dmd_utils.py`

## Experiments 2: Turbulence

We have computed Koopman modes using ResKoopNet and Hankel-DMD in these experiments:

1. Experiment on turbulence using ResKoopNet: `turbulence_reskoopnet.ipynb`
2. Plotting Koopman modes computed from ResKoopNet method: `turbulence_reskoopnet_plot.ipynb`
3. Experiment on turbulence using ResKoopNet with plotting: `turbulence_hankeldmd.ipynb`

## Solvers

1. The `solver_edmd_tf.py` is based on a modified version of the EDMD-DL solver. You can find more details [here](https://github.com/MLDS-NUS/KoopmanDL?tab=readme-ov-file).
2. `solver_resdmd_tf.py` and `solver_resdmd_torch.py` are also modified based on the EDMD-DL solver.

## Plotting Tools

1. `koopman_pseudospec_qr` - A plotting tool to visualize the pseudospectrum. This Python file is translated from [here](https://github.com/MColbrook/Residual-Dynamic-Mode-Decomposition/blob/main/main_routines/KoopPseudoSpecQR.m).


##


---




