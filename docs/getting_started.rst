Getting Started
===============

This is a broad overview of how to perform free energy calculations between two tautomers in solution with a neural network potential using this package.
Currently, multistate free energy calculations are supported; this will be extended.


Installation
-----------------
It is recommended to set up a new python conda environment with :code:`python=3.11` and installing the packages defined `here <https://github.com/saratk1/blob/main/devtools/conda-envs/test_env.yaml>`_ using :code:`mamba`.
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
    sim_control_params:
        n_samples: 100 # number of samples to generate
        n_steps_per_sample: 50 # number of steps between each sample
        nnp: "tautani2x" # name of the NNP 
        lambda_val: 0 # lambda value (0 == tautomer 1; 1 == tautomer 2)
        nr_lambda_states: 1 # number of lambda states 
    tautomer_systems: # test system from the Tautobase
        name: "acetylacetone"
        smiles_t1: "CC(CC(C)=O)=O"
        smiles_t2: "CC(/C=C(/C)\\O)=O"
        dG: 1.132369 # experimental value in kcal/mol
    
An equilibrium simulation can be run with :code:`python run_equ_sim.py 0 config.yaml`. The first argument specifies the device index of the GPU, the second one gives the path to the :code:`config.yaml` file.
This script will first create a PDB file of the solvated (30x30x30 A box) tautomer system. Afterwards, a 5 ps test-simulation will be run.
If ``nr_lambda_states: 1``, only one simulation will be run for `lambda = 0`. If ``nr_lambda_states: 8``, eight simulations will be run (for `lambda = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]`).

2. Visualization
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

To visualize the samples generated in the equilibrium simulation, the :code:`vis.py` script can be used. This script takes the same :code:`config.yaml` file as input and wraps each frame of either one or all provided trajectories. The output is saved as a new `.dcd` file.

3. Free energy estimation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

After running a series of equilibrium simulations (e.g. 8 lambda states from 0 to 1), the free energy difference between the two tautomers can be estimated using the :code:`calculate_dG.py` script. 
To this end, the number of lambda states needs to be specified in the :code:`config.yaml` file (e.g. ``nr_lambda_states: 8``). 
The script can then be run with :code:`python calculate_dG.py 0 config.yaml`. Again, the first argument specifies the device index of the GPU, the second one gives the path to the :code:`config.yaml` file.
The script will read in the samples from the equilibrium simulations, compute the free energy difference between the two tautomers using the `MBAR` estimator, and print the calculated and the experimental result.

