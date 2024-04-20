# script for getting torsion profiles, first argument is the config.yaml file
import matplotlib.pyplot as plt
import numpy as np
import mdtraj as md
import yaml
import os
import seaborn as sns
import sys

# get system information and simulation parameters
config_path = sys.argv[1] # path to config.yaml file
with open(config_path, 'r') as file:
        config = yaml.safe_load(file)

name = config["tautomer_systems"]["name"]
lambs = config['analysis']['lambda_scheme']
n_samples = config['sim_control_params']['n_samples']
n_steps_per_sample = config['sim_control_params']['n_steps_per_sample']
base = config['base']
indices = config['analysis']['torsion_indices']

if not os.path.exists(f"{base}/{name}/analysis"):
    print("Creating directory:", f"{base}/{name}/analysis")
    os.makedirs(f"{base}/{name}/analysis")

if lambs == None:
    lambs = [float(config['analysis']['lambda_val'])]

def plot_torsion(lambs, name, base, n_samples, n_steps_per_sample, indices):

    if not isinstance(lambs, list):
        lambs = [lambs]

    if len(lambs) == 1:
        fig, axs = plt.subplots(1, 1, figsize=(8, 2))
        axs = [axs]  # convert single Axes object to a list for consistency
    else:
        fig, axs = plt.subplots(len(lambs), 1, figsize=(8, 2 * len(lambs))) 

    for i, lambda_val in enumerate(lambs):
        print(f"Loading trajectory for lambda = {lambda_val:.4f}...")
        
        traj = md.load_dcd(
            f"{base}/{name}/{name}_samples_{n_samples}_steps_{n_steps_per_sample}_lamb_{lambda_val:.4f}.dcd",
            top=f"{base}/{name}/{name}_hybrid_solv.pdb"
        )
        
        # discard first 20% of the frames
        n_frames = len(traj)
        discard_frames = int(0.2 * n_frames)
        traj = traj[discard_frames:]
        print(f"Discarding first {discard_frames} frames of the trajectory (20%)...")
        
        torsions = md.compute_dihedrals(traj, indices=[indices], periodic=True, opt=True)

        axs[i].hist(np.degrees(torsions), bins=180, range=(-180, 180), density=True, alpha=0.5)

        sns.kdeplot(np.degrees(torsions), common_norm=False, fill=True, linewidth=2, ax=axs[i])

        axs[i].set_ylabel('Density')
        axs[i].set_title(f'lambda = {lambda_val:.4f}')
        axs[i].set_xlim([-180, 180])
        
        # Remove legend
        axs[i].legend().remove()

    axs[-1].set_xlabel('Torsion Angle [degrees]')

    fig.subplots_adjust(top=0.8)  
    fig.suptitle(f"{name} torsion profile ({indices})", fontsize=15, y=0.99)  

    plt.tight_layout()  #
    plt.show()
    plt.savefig(f"{base}/{name}/analysis/{name}_samples_{n_samples}_steps_{n_steps_per_sample}_torsion_{indices}.png")

plot_torsion(lambs=lambs, name=name, base=base, n_samples=n_samples, n_steps_per_sample=n_steps_per_sample, indices=indices)
