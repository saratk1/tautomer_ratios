import numpy as np
from openmm.app import (
    Simulation,
    DCDReporter,
    StateDataReporter,
    PDBFile,
    Topology,
)
from openmm import System
from openmmml import MLPotential
from openmm import app
from openmm import Platform
from openmm import LangevinIntegrator
from openmm import unit
from openmm import MonteCarloBarostat
from openmmtools.constants import kB
from openmmtools.forces import FlatBottomRestraintBondForce
import os
from sys import stdout
import torch 
import sys
import openmm
import yaml
from solvate_hybrid import save_solv_pdb
from equ import run_mfes

# read yaml file
with open('config.yaml', 'r') as file:
    config = yaml.safe_load(file)

###################################################################################################################################
#                                          generate solvated hybrid tautomer structure                                            #
###################################################################################################################################
name = config["tautomer_systems"]["name"]
smiles_t1 = config["tautomer_systems"]["smiles_t1"]
smiles_t2 = config["tautomer_systems"]["smiles_t2"]

print("\n")
print("############################################################")
print(f"Generating solvated hybrid tautomer structure for {name}...")
print("############################################################ \n")
save_solv_pdb(name=name, smiles_t1=smiles_t1, smiles_t2=smiles_t2)

###################################################################################################################################
#                                                       run MFES                                                                  #
###################################################################################################################################

n_samples = config['sim_control_params']['n_samples']
n_steps_per_sample = config['sim_control_params']['n_steps_per_sample']
#lambda_val = config['sim_control_parameters']['lambda_val']
nr_lambda_states = config['sim_control_params']['nr_lambda_states']
nnp = config['sim_control_params']['nnp']

lambs = np.linspace(0, 1, nr_lambda_states)

print("\n")
print("############################################################")
print("Starting equilibrium simulations")
print("############################################################ \n")
for lamb in lambs:
    print(f"Running equilibrium simulation for {lamb=}")
    run_mfes(name=name, 
         lambda_val=lamb, 
         nnp=nnp,
         n_samples=n_samples, 
         n_steps_per_sample=n_steps_per_sample)
    print("\n")

###################################################################################################################################
#                                       calculate free energy difference                                                          #
###################################################################################################################################