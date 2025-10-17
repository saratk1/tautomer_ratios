# script for running simulations

# first argument specifies the path to the config.yaml file, 
# second specifies the lambda value,
# third specifies the run number

import numpy as np
import os
import sys
import glob
import socket
from sys import stdout
import torch 
import yaml
from openmm.app import (
    DCDReporter,
    StateDataReporter,
    PDBReporter
)
from openmm import app
from openmm import Platform
from taut_diff.tautomers import save_pdb, solvate
from taut_diff.equ import get_sim
from openmm import unit
from taut_diff.constant import temperature

hostname = socket.gethostname()
print(f"Running on node: {hostname}")

# read yaml file
config_path = sys.argv[1] 
with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
        
lambda_val = float(sys.argv[2]) # lambda value for which simulation should be performed 
run = sys.argv[3]

name = config["tautomer_systems"]["name"]
smiles_t1 = config["tautomer_systems"]["smiles_t1"]
smiles_t2 = config["tautomer_systems"]["smiles_t2"]
base = config["base"]
exp = config["exp"]
precision = config['sim_control_params']["precision"]
n_samples = config['sim_control_params']['n_samples']
n_steps_per_sample = config['sim_control_params']['n_steps_per_sample']
nnp = config['sim_control_params']['nnp']
ensemble = config['sim_control_params']['ensemble']
bond_restraints = config['sim_control_params']['restraints']['bond']
flat_bottom_restraints = config['sim_control_params']['restraints']['flat_bottom_bond']
angle_restraints = config['sim_control_params']['restraints']['angle']
bond_restr_constant = config['sim_control_params']['restraints']['bond_constant']
angle_restr_constant = config['sim_control_params']['restraints']['angle_constant']
environment = config['sim_control_params']['environment']
box_length = config['sim_control_params']['box_length']
minimize = config['sim_control_params']['minimize']
overwrite = config['sim_control_params']['overwrite']
lambda_scheme = config['analysis']['lambda_scheme']

print("\n")
print("-----------------------------------------------------------------------------------")
print("-----------------------------------------------------------------------------------")
print(f"{run} of: {exp}")
print("-----------------------------------------------------------------------------------")
print("-----------------------------------------------------------------------------------")
print(f"Simulation will be set up for for {name} in {environment} with {nnp}")
print(f"smiles t1 = {smiles_t1}\nsmiles t2 = {smiles_t2}")
print(f"Data will be stored at {base}/{exp}/{run}/{name}")
print(f"Harmonic bond restraints are set to {bond_restraints} & harmonic angle restraints are set to {angle_restraints}")
print(f"Flat bottom bond restraint is set to {flat_bottom_restraints}")
print(f"Simulation will be run for a total of {(n_samples*n_steps_per_sample/1000):.3f} ps; {n_samples} samples will be saved every {n_steps_per_sample/1000} ps")
print(f"Config file used to load simulation information: {config_path}")
print(f"Simulation will be performed in {precision} precision")

print("\n")
# create directory to store results
if not os.path.exists(f"{base}/{exp}/{run}/{name}"):
    print("Creating directory to store results:", f"{base}/{exp}/{run}/{name}")
    os.makedirs(f"{base}/{exp}/{run}/{name}")

print(f"Working directory where results will be stored: {base}/{exp}/{run}/{name}")

###################################################################################################################################
#                                          generate solvated hybrid tautomer structure                                            #
##################################################################################################################################
if environment == "vacuum":
    pdb_file_name = f"{base}/{exp}/{name}/{run}/{name}_hybrid.pdb"
   
elif environment == "waterbox":
    pdb_file_name = f"{base}/{exp}/{name}/{run}/{name}_hybrid_solv.pdb"
    
    
pdb_file_name_vac = f"{base}/{exp}/{name}/{run}/{name}_hybrid.pdb"


if not os.path.exists(pdb_file_name_vac):
    print("\n")
    print("############################################################")
    print(f"Generating hybrid tautomer structure for {name} in {environment}...")
    print("############################################################ \n")
    pdb_filepath = save_pdb(name=name, 
                    smiles_t1=smiles_t1, 
                    smiles_t2=smiles_t2, 
                    base = f"{base}/{exp}/{name}/{run}/",
                    environment=environment,
                    box_length=box_length,
                    minimize=minimize)
elif environment == "waterbox" and not os.path.exists(pdb_file_name):
    print("\n")
    print("############################################################")
    print(f"Hybrid form already exists and will be solvated:")
    print("############################################################ \n")

    pdb_filepath = solvate(pdb_filepath=pdb_file_name_vac, 
                           box_length=box_length, 
                           base=f"{base}/{exp}/{name}/{run}/",
                           name=name)
else:
    print("\n")
    print("############################################################")
    print(f"PDB input files already exist and will be loaded:")
    print("############################################################ \n")
    
    print("pdb file: ", pdb_file_name)
    print("pdb file (vacuum): ", pdb_file_name_vac)

###################################################################################################################################
#                                                       run MFES                                                                  #
###################################################################################################################################
# check if 11 trajectories exist

if overwrite == False and os.path.exists(f"{base}/{exp}/{name}/{run}/{name}_samples_{n_samples}_steps_{n_steps_per_sample}_lamb_{lambda_val:.4f}_DONE.txt"):
    #raise RuntimeError(f"This directory already contains 11 trajectories. Please change the 'overwrite' keyword in the config.yaml file to True if you want to overwrite the existing trajectories or add new ones.")
    raise RuntimeError(f"Trajectory for lambda = {lambda_val:.4f} already exists. Please change the 'overwrite' keyword in the config.yaml file to True if you want to overwrite the existing trajectory...")

print("\n")
print("############################################################")
print("Setting up equilibrium simulations")
print("############################################################ \n")

################################################ set up simulation ###################################################################

torch._C._jit_set_nvfuser_enabled(False) # to prevent decrease of performance

#for lambda_val in lambda_scheme: # uncomment to run simulation for all lambda values in lambda_scheme
print(f"Setting up equilibrium simulation for lambda value={lambda_val:.4f} in {environment}")

print(f"Loading system from: {pdb_file_name}")
solv_system = app.PDBFile(pdb_file_name) 
system_topology = solv_system.getTopology()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
platform = Platform.getPlatformByName("CUDA")  
    
sim = get_sim(solv_system=solv_system, 
            environment=environment,
            nnp=nnp, 
            lambda_val=lambda_val, 
            device=device,
            platform=platform,
            bond_restraints=bond_restraints,
            flat_bottom_restraints=flat_bottom_restraints,
            angle_restraints=angle_restraints,
            bond_restr_constant=bond_restr_constant,
            angle_restr_constant=angle_restr_constant,
            ensemble=ensemble,
            pdb_path_vacuum=pdb_file_name_vac,
            precision=precision)

################################################ data collection setup ###############################################################

# define where to store simulation info
statereport_file = f"{base}/{exp}/{name}/{run}/{name}_samples_{n_samples}_steps_{n_steps_per_sample}_lamb_{lambda_val:.4f}_report.csv"
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
        totalSteps=n_samples * n_steps_per_sample,
        remainingTime=True,
        separator="\t",
    )
)

# define where to store samples
trajectory_file = f"{base}/{exp}/{name}/{run}/{name}_samples_{n_samples}_steps_{n_steps_per_sample}_lamb_{lambda_val:.4f}.dcd" 
sim.reporters.append(
    DCDReporter(
        trajectory_file,
        reportInterval=n_steps_per_sample,
    )
)

################################################# restraints   ###################################################################

if flat_bottom_restraints:
    # Define where to store restraint energy
    restraint_energy_file = f"{base}/{exp}/{name}/{run}/{name}_samples_{n_samples}_steps_{n_steps_per_sample}_lamb_{lambda_val:.4f}_restraint_energy.csv"

    # Custom restraint energy reporter
    class RestraintEnergyReporter:
        def __init__(self, file, reportInterval, restraintGroup):
            self.file = open(file, "w")
            self.reportInterval = reportInterval
            self.restraintGroup = restraintGroup
            self.file.write("Step\tRestraint Energy (kJ/mol)\n")

        def report(self, simulation, state):
            step = simulation.currentStep
            restraint_energy = simulation.context.getState(getEnergy=True, groups={self.restraintGroup}).getPotentialEnergy()
            self.file.write(f"{step}\t{restraint_energy / unit.kilojoule_per_mole:.6f}\n")

        def finalize(self):
            self.file.close()
            
    # Add custom restraint energy reporter
    restraint_reporter = RestraintEnergyReporter(
        file=restraint_energy_file,
        reportInterval=n_steps_per_sample, 
        restraintGroup=1   # Force group for the restraint
    )
################################################### sampling #####################################################################
# save final state
# Define where to store the final PDB file
final_pdb_file = f"{base}/{exp}/{name}/{run}/{name}_samples_{n_samples}_steps_{n_steps_per_sample}_lamb_{lambda_val:.4f}_final.pdb"

# Add PDBReporter to save the final structure at the end of the simulation
sim.reporters.append(
    PDBReporter(final_pdb_file, n_samples * n_steps_per_sample)
)

##################################################################################################################
# set coordinates
print("Positions will be set....")
sim.context.setPositions(solv_system.getPositions())
print("Velocities will be set to temperature....")
sim.context.setVelocitiesToTemperature(temperature) # here we go through forward

# get potential energy of initial state (for testing purposes)
initial_state = sim.context.getState(getEnergy=True)  # here we go through forward
initial_potential_energy = initial_state.getPotentialEnergy().value_in_unit(unit.kilojoules_per_mole)
print(f"Initial Potential Energy = {initial_potential_energy} kJ/mole")


print("\n")
print("############################################################")
print("Starting equilibrium simulations")
print("############################################################ \n")

# perform sampling
print(f"Running equilibrium simulation for lambda value={lambda_val:.4f}")
#sim.step(n_samples * n_steps_per_sample)
# Perform sampling
for sample in range(n_samples):
    sim.step(n_steps_per_sample)  # Step n_steps_per_sample at a time
    if flat_bottom_restraints:
        state = sim.context.getState(getEnergy=True)  # Get the state after the step
        restraint_reporter.report(sim, state)  # Report restraint energy
    
if flat_bottom_restraints:
    restraint_reporter.finalize()
sim.reporters.clear()

print(f"State report saved to: {statereport_file}")
print(f"Trajectory saved to: {trajectory_file}")
if flat_bottom_restraints:
    print(f"Restraint energy saved to: {restraint_energy_file}")

file_path = f"{base}/{exp}/{name}/{run}/{name}_samples_{n_samples}_steps_{n_steps_per_sample}_lamb_{lambda_val:.4f}_DONE.txt"
if not os.path.exists(file_path):
    print("Writing file which marks end of simulation:", file_path)
    with open(file_path, 'w'):
        pass 