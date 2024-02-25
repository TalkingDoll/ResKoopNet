# NN-ResDMD
In this project, we will compare NN-ResDMD and other methods such as Hankle-DMD and EDMD-DL.

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

5. Install the package Tensorflow:
```bash
pip install tensorflow==2.8
```



- Inside DNN_resDMD/examples, you can run several tests with only NN-resDMD. 

>solver_edmd: original EDMD-DL

>solver_resdmd: Directly optimize the eqn(4.6) defined in paper "Rigorous ...". In the script, we mutiply the eigenvector matrix in the loss function.


## Acknowledgements

This project's codebase is significantly modified based on the [KoopmanDL package](https://github.com/MLDS-NUS/KoopmanDL) developed by MLDS-NUS. We extend our gratitude to the authors and contributors of KoopmanDL for their valuable work.



