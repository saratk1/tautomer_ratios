# first sys.argv is device index, second specifies the path to the config.yaml file, third specifies the lambda value

import numpy as np
import os
import sys
from sys import stdout
import torch 
import yaml
from openmm.app import (
    DCDReporter,
    StateDataReporter,
)
from openmm import app
from openmm import Platform
from taut_diff.tautomers import save_solv_pdb
from taut_diff.equ import get_sim
from openmm import unit

# read yaml file
config_path = sys.argv[2]
with open(config_path, 'r') as file:
        config = yaml.safe_load(file)

name = config["tautomer_systems"]["name"]
smiles_t1 = config["tautomer_systems"]["smiles_t1"]
smiles_t2 = config["tautomer_systems"]["smiles_t2"]
base = config["base"]
n_samples = config['sim_control_params']['n_samples']
n_steps_per_sample = config['sim_control_params']['n_steps_per_sample']
nnp = config['sim_control_params']['nnp']
lambda_val = float(sys.argv[3])

print("\n")
# create directory to store results
if not os.path.exists(f"{base}/{name}"):
    print("Creating directory:", f"{base}/{name}")
    os.makedirs(f"{base}/{name}")

print(f"Working directory: {base}/{name}")

###################################################################################################################################
#                                          generate solvated hybrid tautomer structure                                            #
##################################################################################################################################

if not os.path.exists(f"{base}/{name}/{name}_hybrid_solv.pdb"):
    print("\n")
    print("############################################################")
    print(f"Generating solvated hybrid tautomer structure for {name}...")
    print("############################################################ \n")
    save_solv_pdb(name=name, 
                    smiles_t1=smiles_t1, 
                    smiles_t2=smiles_t2, 
                    base = base)

###################################################################################################################################
#                                                       run MFES                                                                  #
###################################################################################################################################
print("\n")
print("############################################################")
print("Starting equilibrium simulations")
print("############################################################ \n")

################################################ set up simulation ###################################################################
torch._C._jit_set_nvfuser_enabled(False) # to prevent decrease of performance

print(f"Setting up equilibrium simulation for lambda value={lambda_val:.4f}")

solv_system = app.PDBFile(f'{base}/{name}/{name}_hybrid_solv.pdb')
system_topology = solv_system.getTopology()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
platform = Platform.getPlatformByName("CUDA")  
device_index = sys.argv[1]

sim = get_sim(solv_system=solv_system, 
            name=name,
            base=base,
            nnp=nnp, 
            lambda_val=lambda_val, 
            device=device,
            platform=platform,
            bond_restraints=True,
            angle_restraints=True,
            control_param=4,
            device_index=device_index)

################################################ data collection setup ###############################################################

# define where to store simulation info
statereport_file = f"{base}/{name}/{name}_samples_{n_samples}_steps_{n_steps_per_sample}_lamb_{lambda_val:.4f}_report.csv" 
sim.reporters.append(
    StateDataReporter(
        statereport_file,
        reportInterval=n_steps_per_sample,
        step=True,
        time=True,
        potentialEnergy=True,
        totalEnergy=True,
        temperature=True,
        density=True,
        speed=True,
        elapsedTime=True,
        separator="\t",
    )
)
# define where to store samples
trajectory_file = f"{base}/{name}/{name}_samples_{n_samples}_steps_{n_steps_per_sample}_lamb_{lambda_val:.4f}.dcd" 
sim.reporters.append(
    DCDReporter(
        trajectory_file,
        reportInterval=n_steps_per_sample,
    )
)
################################################### sampling #####################################################################

# set coordinates
sim.context.setPositions(solv_system.getPositions())

# get potential energy of initial state (for testing purposes)
# initial_state = sim.context.getState(getEnergy=True)  
# initial_potential_energy = initial_state.getPotentialEnergy().value_in_unit(unit.kilojoules_per_mole)
# print(f"Initial Potential Energy = {initial_potential_energy} kJ/mol")

# perform sampling
print(f"Running equilibrium simulation for lambda value={lambda_val:.4f}")
sim.step(n_samples * n_steps_per_sample)
sim.reporters.clear()
print(f"State report saved to: {statereport_file}")
print(f"Trajectory saved to: {trajectory_file}")

file_path = f"{base}/{name}/{name}_samples_{n_samples}_steps_{n_steps_per_sample}_lamb_{lambda_val:.4f}_DONE.txt"
if not os.path.exists(file_path):
    print("Writing file which marks end of simulation:", file_path)
    with open(file_path, 'w'):
        pass 