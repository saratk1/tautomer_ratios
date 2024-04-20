# first sys.argv is device index, second specifies the path to the config.yaml file

import numpy as np
import yaml
import sys
import torch
from tqdm import tqdm
from typing import Tuple

from openmm import unit
from openmm import unit
from openmm import app
from openmm import Platform
from openmm import LangevinIntegrator
from openmmml import MLPotential
from openmm.app import Simulation
from openmmtools.constants import kB

import mdtraj as md
from pymbar import MBAR
from taut_diff.equ import calculate_u_kn

print("\n")
print("############################################################")
print("Free energy calculation")
print("############################################################ \n")

# get system information and simulation parameters
config_path = sys.argv[2] # path to config.yaml file
with open(config_path, 'r') as file:
        config = yaml.safe_load(file)

name = config["tautomer_systems"]["name"]
lambs = config['analysis']['lambda_scheme']
nnp = config['sim_control_params']['nnp']
n_samples = config['sim_control_params']['n_samples']
n_steps_per_sample = config['sim_control_params']['n_steps_per_sample']
base = config['base']
experiment = config['tautomer_systems']['dG']

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
platform = Platform.getPlatformByName("CUDA")  
device_index = sys.argv[1]

solv_system=app.PDBFile(f'{base}/{name}/{name}_hybrid_solv.pdb')

print(f"Loading samples for {name}...")
trajs = []
for lambda_val in lambs:
    traj = md.load_dcd(
            f"{base}/{name}/{name}_samples_{n_samples}_steps_{n_steps_per_sample}_lamb_{lambda_val:.4f}.dcd",
            top=f"{base}/{name}/{name}_hybrid_solv.pdb", # also possible to use the tmp.pdb
        )
    trajs.append(traj)

discard_frames=int((n_samples / 100) * 20) # discard first 20%

N_k, u_kn = calculate_u_kn(
    trajs=trajs,
    solv_system=solv_system,
    nnp=nnp,
    name=name,
    base=base,
    lambda_scheme=lambs,
    platform=platform,
    device=device,
    device_index=device_index,
    discard_frames=discard_frames,
    every_nth_frame=30,
    )

# initialize the MBAR maximum likelihood estimate
mbar = MBAR(u_kn, N_k)
r = mbar.compute_free_energy_differences()["Delta_f"][0][-1]

print("##################################################")
print(f"Computed dG: {r*0.5922:.2f} kcal/mol") # convert from kBT to kcal/mol
print(f"Experiment dG: {experiment:.2f} kcal/mol")
print("##################################################")