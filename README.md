### Path-Dependent SDEs: Solutions and Parameter Estimation

This Python code repository generates simulated trajectory data from linear signature SDEs for the experiments in the paper "Path-Dependent SDEs: Solutions and Parameter Estimation".

The main code is contained in `data_generation.py`, which performs two main steps:

1. simulates trajectory data from a specified linear signature SDE by sampling from the solution to the lifted Stratonovich SDE. The solution is estimated using the Heun-Stratonovich solver from the Diffrax package.
2. computes empirical signature terms (up to a specified truncation level). The empirical signatures are estimated directly from the level 1 path, using the iisignature package.

### Usage tips
- First, to install necessary packages, run ``conda env create -f environment.yml`` and then activate the environment by running ``conda activate signature_sde``
- To run `data_generation.py` for a desired SDE, you must include an appropriate config file. For examples, see the configuration files in `sde_configs`. 
  - In particular, the SDE is specified according to a dictionary, and the config also provides hyperparameters, such as the time discretization `dt`, the total time `T`, and the truncation level for empirical signature terms `q_iterated`.
  - Once an appropriate config .json file is created, run ``data_generation.py --config PATH_TO_CONFIG_FILENAME --experiment_name YOUR_EXPERIMENT_NAME``
    - A .csv file will be created in the directory `YOUR_EXPERIMENT_NAME`, which contains all signature terms up to the specified truncation level `q_iterated`, evaluated at time `T`. 
- To create the data for reproducing the main experiments from the paper, simply run ``./run_experiments.sh ``
