# tautomer_ratios

[//]: # (Badges)
[![GitHub Actions Build Status](https://github.com/REPLACE_WITH_OWNER_ACCOUNT/taut_diff/workflows/CI/badge.svg)](https://github.com/REPLACE_WITH_OWNER_ACCOUNT/taut_diff/actions?query=workflow%3ACI)
[![codecov](https://codecov.io/gh/REPLACE_WITH_OWNER_ACCOUNT/taut_diff/branch/main/graph/badge.svg)](https://codecov.io/gh/REPLACE_WITH_OWNER_ACCOUNT/taut_diff/branch/main)


This package can be used to calculate the free energy difference between tautomer pairs in solution using a neural network potential (ANI-2x or MACE-OFF23)


## Installation

It is recommended to set up a new python conda environment with `python=3.12` and installing the packages defined [here](https://github.com/saratk1/tautomer_ratios/blob/main/devtools/conda-envs/test_env.yaml) using `mamba`.
This package can be installed using:
`pip install git+https://github.com/saratk1/tautomer_ratios.git`.


## How to use this package

There are three scripts that should help to use this package; they are located in `tautomer_ratios/scripts`.

### 1. Sampling


`run_equ_sim.py` should be used first to generate samples from the equilibrium distribution of the system. It should be used in combination with a `config.yaml` file, which holds information about the tautomer system and simulation parameters.
Here is an example of a `config_tp_558.yaml` to generate samples for the enol form of the tautomer (i.e. at `lambda=0`):

```yaml

exp: "test_simulation" # name of the experiment
base: "tautomer_ratios/data/" # base directory for output files, adjust this path
sim_control_params:
  n_samples: 1200 # number of samples to generate
  n_steps_per_sample: 1000 # number of steps between each sample
  nnp: "ani2x" # or mace-off23-small  
  precision: "sinlge" # or double, only for mace models
  ensemble: "npt" # or nvt
  environment: "waterbox" # or vacuum
  box_length: 21 # in Angstrom
  restraints:
   bond: True # harmonic bond restraint
   flat_bottom_bond: True # flat bottom restraint
   angle: True # harmonic angle restraint
   bond_constant: 100 # kcal/mol/A^2
   angle_constant: 10 # kcal/mol/rad^2
  overwrite: False # if existing trajectories should be overwritten
  minimize: False # if system should be minimized before simulation
tautomer_systems: # test system from the Tautobase
  name: "tp_558" 
  smiles_t1: "Cc1nccc(O)c1" #enol 
  smiles_t2: "CC(NC=C1)=CC1=O" #keto
  dG: -5.730060000000001 # reference value in kcal/mol
analysis:
  lambda_val: 0 
  lambda_scheme:  [0.0, 0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 0.8, 0.9, 0.95, 1.0] 
  torsion_indices:  [7,5,6,13] # for torsion distribution analysis
  runs_analysis: ["run01", "run02", "run03"] # number of simulation runs
  every_nth_frame: 2 # for free energy differnce calculation
```

A simulation can be run with `python run_equ_sim.py config_tp_558.yaml 0 run01`. The first argument specifies the path to the `config_tp_558.yaml` file, the second one the lamba value, the third one the run number.
This script will first create a PDB file of the solvated (21x21x21 A box) tautomer system (including hybrid tautomer structure). Afterwards, a 1.2 ns simulation  will be run for `lambda = 0` (the topology specified by `smiles_t1`).
The samples are saved in `{base}/{name}/{run}`.

### 2. Visualization

The `wrapping.py` script can be used to visualize the samples generated during the simulation. This script takes the same `config_tp_558.yaml` file as input, centers the solute and wraps each frame of either one or all provided trajectories (depending on the input in the `config_tp_558.yaml` file; if `lambda_val = null`, all trajectories indicated in the `lambda_scheme` are wrapped). The output is saved as new `.dcd` files (in `{base}/{name}/{analysis}`).

### 3. Free energy estimation

After running a series of equilibrium simulations (e.g. 11 lambda states from 0 to 1), the free energy difference between the two tautomers can be estimated using the `calculate_dG.py` script. 
To this end, the lambda scheme used for the simulations needs to be specified in the `config_tp_558.yaml` file (``lambda_scheme: [0.0, 0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 0.8, 0.9, 0.95, 1.0] ``). 
The script can then be run with `python calculate_dG.py config_tp_558.yaml run01`. Again, the first argument specifies the path to the `config.yaml` file, the second one the run.
The script will read in the samples, discard the first 200 samples of the trajectory (can be adjusted in the `calculate_dG.py` script), compute the free energy difference between the two tautomers using the `MBAR` estimator, and print the calculated and the reference result.
The ``u_kn`` and ``N_k`` needed for the `MBAR` estimation and/or subsequent visualization of the overlap matrix are saved in `{base}/{name}/{run}/{analysis}`.


### Copyright

Copyright (c) 2024, Sara Tkaczyk


#### Acknowledgements
 
Project based on the 
[Computational Molecular Science Python Cookiecutter](https://github.com/molssi/cookiecutter-cms) version 1.1.<br>
Code contained in this repository builds on https://github.com/choderalab/neutromeratio.<br>
Implementations of ANI-2x and MACE-OFF23 were adapted from https://github.com/openmm/openmm-ml.<br>
