# module for wrapping the trajectory

import mdtraj as md
import numpy as np
import yaml
import sys

config_path = sys.argv[1] # path to yaml file
with open(config_path, 'r') as file:
        config = yaml.safe_load(file)

name = config["tautomer_systems"]["name"]
n_samples = config['sim_control_params']['n_samples']
n_steps_per_sample = config['sim_control_params']['n_steps_per_sample']
lambda_val = config['sim_control_params']['lambda_val']
base = config['base']

# load dcd file containting samples and topology from pdb file of the whole system
box_traj = md.load_dcd(
            f"{base}/{name}_samples_{n_samples}_steps_{n_steps_per_sample}_lamb_{lambda_val:.4f}.dcd",
            top=f"{base}/{name}/{name}_hybrid_solv.pdb", # also possible to use the tmp.pdb
        )
tautomer = md.load_pdb(f"{base}/{name}/{name}_hybrid.pdb")

# get array of tautomer bonds
def get_sorted_bonds(traj):
    sorted_bonds = sorted(traj.topology.bonds, key=lambda bond: bond[0].index)
    sorted_bonds = np.asarray([[b0.index, b1.index] for b0, b1 in sorted_bonds], dtype=np.int32)    
    return sorted_bonds

sorted_bonds = np.concatenate((get_sorted_bonds(tautomer), get_sorted_bonds(box_traj)), axis=0)

# wrap
new_traj = box_traj.make_molecules_whole(sorted_bonds=sorted_bonds)
traj_output = f'{base}/{name}_samples_{n_samples}_steps_{n_steps_per_sample}_lamb_{lambda_val:.4f}_wrapped.dcd'
print(f"writing wrapped trajectory to {traj_output}")
new_traj.save_dcd(traj_output)