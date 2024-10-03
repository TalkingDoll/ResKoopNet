# NN-ResDMD



---

## Instructions

### pendulum.m

The `pendulum.m` file is used to generate pendulum data.

### Experiments 1

I have provided two versions of Pendulum system using NN-ResDMD and a comparison test with other 4 methods:

1. **TensorFlow Version**: `pendulum_nnresdmd_tf.ipynb`
2. **PyTorch Version**: `pendulum_nnresdmd_torch.ipynb`
3. Comparison test: `pendulum_edmd_resdmd_hankeldmd_edmddl.ipynb`
4. Algorithm of Hankel-DMD: `hankel_dmd_utils.py`

### Experiments 2

We have computed Koopman modes using NN-ResDMD and Hankel-DMD in these experiments:

1. Experiment on turbulence using NN-ResDMD: `turbulence_nnresdmd.ipynb`
2. Plotting Koopman modes computed from NN-ResDMD method: `turbulence_nnresdmd_plot.ipynb`
3. Experiment on turbulence using NN-ResDMD with plotting: `turbulence_hankeldmd.ipynb`

### Solvers

1. The `solver_edmd_tf.py` is based on a modified version of the EDMD-DL solver. You can find more details [here](https://github.com/MLDS-NUS/KoopmanDL?tab=readme-ov-file).
2. `solver_resdmd_tf.py` and `solver_resdmd_torch.py` are also modified based on the EDMD-DL solver.

### Plotting Tools

1. `koopman_pseudospec_qr` - A plotting tool to visualize the pseudospectrum. This Python file is translated from [here](https://github.com/MColbrook/Residual-Dynamic-Mode-Decomposition/blob/main/main_routines/KoopPseudoSpecQR.m).

### 

Feel free to explore these files and use them as needed for your experiments and analysis.

---



