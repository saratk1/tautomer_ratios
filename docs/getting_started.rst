Getting Started
===============

This is a broad overview of how to calculate the free energy difference between two tautomers in solution with a neural network potential using this package.
Currently, multistate free energy calculations are supported; this will be extended.


Installation
-----------------
It is recommended to set up a new python conda environment with :code:`python=3.11` and installing the packages defined `here <https://github.com/saratk1/taut_diff/blob/main/devtools/conda-envs/test_env.yaml>`_ using :code:`mamba`.
This package can be installed using:
:code:`pip install git+https://github.com/saratk1/taut_diff.git`.


How to use this package
----------------------------------
There are three scripts that should help to use this package; they are located in :code:`taut_diff/scripts`.

1. Sampling
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

:code:`run_equ_sim.py` should be used first to generate samples from the equilibrium distribution of the system. It should be used in combination with a :code:`config.yaml` file, which holds information about the tautomer system and simulation parameters.
Here is an example of a :code:`config.yaml` to generate samples for the first tautomer (`lambda=0`):

.. code:: yaml

    base: "../testing" # base directory for output files
    tautomer_systems: # test system from the Tautobase
        name: "tp_44"
        smiles_t1: "CC=O" # keto
        smiles_t2: "C=CO" #enol
        dG: 1.132369 # experimental value in kcal/mol

    sim_control_params:
        n_samples: 500 # number of samples to generate
        n_steps_per_sample: 10 # number of steps between each sample
        nnp: "tautani2x" # name of the NNP 

    analysis:
        lambda_scheme: [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
        lambda_val: null
        torsion_indices: [3, 2, 1, 5]

The :code:`base`, :code:`tautomer_systems` and :code:`sim_control_params` blocks are important for the simulation.
An equilibrium simulation can be run with :code:`python run_equ_sim.py 0 config.yaml 0`. The first argument specifies the device index of the GPU, the second one gives the path to the :code:`config.yaml` file, the third one specifies the lamba value.
This script will first create a PDB file of the solvated (30x30x30 A box) tautomer system. Afterwards, a 5 ps test-simulation for `lambda = 0` (the topology specified by :code:`smiles_t1`) will be run.
The samples are saved in :code:`{base}/{name}`.

2. Visualization
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

To visualize the samples generated in the equilibrium simulation, the :code:`wrapping.py` script can be used. This script takes the same :code:`config.yaml` file as input, centers the solute and wraps each frame of either one or all provided trajectories (depending on the input in the :code:`config.yaml` file; if :code:`lambda_val = null`, all trajectories indicated in the :code:`lambda_scheme` are wrapped). The output is saved as a new `.dcd` file (in :code:`{base}/{name}/{analysis}`).

3. Free energy estimation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

After running a series of equilibrium simulations (e.g. 11 lambda states from 0 to 1), the free energy difference between the two tautomers can be estimated using the :code:`calculate_dG.py` script. 
To this end, the lambda scheme used for the simulations in the first step needs to be specified in the :code:`config.yaml` file (``lambda_scheme: [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]``). 
The script can then be run with :code:`python calculate_dG.py 0 config.yaml`. Again, the first argument specifies the device index of the GPU, the second one gives the path to the :code:`config.yaml` file.
The script will read in the samples from the equilibrium simulations, discard the first 20% of the trajectory, compute the free energy difference between the two tautomers using the `MBAR` estimator, and print the calculated and the experimental result.
The ``u_kn`` and ``N_k`` needed for the `MBAR` estimation and/or subsequent visualization of the overlap matrix are saved in :code:`{base}/{name}/{analysis}`.

