# first sys.argv is device index, 
# second specifies the path to the config.yaml file
# third specifies the run to be analyzed

import numpy as np
import yaml
import sys
import torch
from tqdm import tqdm
from typing import Tuple
import os
from openmm import app
from openmm import Platform
import matplotlib.pyplot as plt
import mdtraj as md
from pymbar import MBAR
from taut_diff.equ import calculate_u_kn, compute_dG_flat_bottom_restraint

print("\n")
print("#################################################################################################################")
print("#################################################################################################################")
print("Free energy calculation")
print("#################################################################################################################")
print("################################################################################################################# \n")

# get system information and simulation parameters
config_path = sys.argv[1] # path to config.yaml file
with open(config_path, 'r') as file:
    config = yaml.safe_load(file)

if sys.argv[2] == "runs_analysis": # if all the samples from all available runs should be analyzed
    runs_analysis = config['analysis']['runs_analysis']
    run = runs_analysis[0]
else:
    runs_analysis = [sys.argv[2]] # if only one run should be analyzed
    run = sys.argv[2]

reweighting = True ###################################################################################

name = config["tautomer_systems"]["name"]
exp = config["exp"]
lambs = config['analysis']['lambda_scheme']
nnp = config['sim_control_params']['nnp']
n_samples = config['sim_control_params']['n_samples']
n_steps_per_sample = config['sim_control_params']['n_steps_per_sample']
base = config['base']
experiment = config['tautomer_systems']['dG']
ensemble = config['sim_control_params']['ensemble']
environment = config['sim_control_params']['environment']
implementation = config['sim_control_params']['implementation']
bond_restraints = config['sim_control_params']['restraints']['bond']
flat_bottom_restraints = config['sim_control_params']['restraints']['flat_bottom_bond']
angle_restraints = config['sim_control_params']['restraints']['angle']
bond_restr_constant = config['sim_control_params']['restraints']['bond_constant']
angle_restr_constant = config['sim_control_params']['restraints']['angle_constant']
control_param = config['sim_control_params']['restraints']['control_param']
every_nth_frame = config['analysis']['every_nth_frame']
precision = "single"

perc = 100


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
platform = Platform.getPlatformByName("CUDA")  
#device_index = sys.argv[3]

# create directory for results
if not os.path.exists(f"{base}/{exp}/{run}/{name}/analysis"):
    print("Creating directory:", f"{base}/{exp}/{run}/{name}/analysis")
    os.makedirs(f"{base}/{exp}/{run}/{name}/analysis")

# load pdb file depending on the environment of the analysed simulation
if environment == "waterbox":
    solv_system = app.PDBFile(f'{base}/{exp}/{run}/{name}/{name}_hybrid_solv.pdb') 
elif environment == "vacuum":
    solv_system = app.PDBFile(f'{base}/{exp}/{run}/{name}/{name}_hybrid.pdb')

print("\n")
print("############################################################")
print(f"Loading samples for {name}...")
print("############################################################ \n")

# list for collecting samples. 
# if only one run is analyzed, the list will have the length of the lambda states and each entry will be a list of used samples for each lambda state
trajs = []
if environment == "waterbox":
    pdb_file = f'{base}/{exp}/{run}/{name}/{name}_hybrid_solv.pdb'
    # list for collecting box vectors
    #box_info =[]
elif environment == "vacuum":
    pdb_file = f'{base}/{exp}/{run}/{name}/{name}_hybrid.pdb'

pdb_file_name_vac = f"{base}/{exp}/{run}/{name}/{name}_hybrid.pdb"

#discard_frames=int((n_samples / 100) * 20) # discard first 20%
discard_frames = 200
print(f"Will discard {discard_frames} samples from the beginning of the trajectory ...")

print(f"Loading trajectories from {base}/{exp}/{run}/{name}/")
for lambda_val in lambs:
    for run in runs_analysis:
        print(f"Loading: {base}/{exp}/{run}/{name}/{name}_samples_{n_samples}_steps_{n_steps_per_sample}_lamb_{lambda_val:.4f}.dcd")
        traj_raw = md.load_dcd(
                f"{base}/{exp}/{run}/{name}/{name}_samples_{n_samples}_steps_{n_steps_per_sample}_lamb_{lambda_val:.4f}.dcd",
                top=pdb_file, # also possible to use the tmp.pdb # CHANGED
            )
        #print(len(traj_raw))
        print(f"Discarding {discard_frames} frames from the beginning of the trajectory...")
        traj = traj_raw[discard_frames:]  # remove first 20%
        if perc == 100:
            traj = traj
        else:
            print(f"Keeping only the first {perc}% of the trajectory...")
            keep_frames = int(len(traj) * (perc / 100))  # Compute frames to keep
            traj = traj[:keep_frames]
            print(f"length of trajectory at lambda {lambda_val}:", len(traj))
        #print(len(traj))
        
        # consider box information
        # if environment == "waterbox":
        #     for i in range(len(traj)):  
        #         box_info.append(traj.openmm_boxes(i))
            
        # elif environment == "vacuum":
        #     box_info = None

        #print("box_info[0]-----------------------------------", box_info[0])
        #print("box_info[0][i]--------------------------------", box_info[0][i])
        #print("*box_info[0][i])------------------------------", *box_info[0][i])
        trajs.append(traj)
        
print("-------------------------------------------------------------------------------------------------------------------------------------")
print("Loaded data:")
print("length of the cumulated trajectory that will be passed to the calculate_u_kn function (should be the number of lambda states loaded): ", len(trajs))
print(f"length of the first entry in the cumulated trajectory (should be the number of frames per lambda state): ", len(trajs[0]))
print("-------------------------------------------------------------------------------------------------------------------------------------")


N_k, u_kn, total_number_of_samples = calculate_u_kn(
    trajs=trajs,
    solv_system=solv_system,
    environment=environment,
    bond_restraints=bond_restraints,
    angle_restraints=angle_restraints,
    flat_bottom_restraints=flat_bottom_restraints,
    bond_restr_constant=bond_restr_constant,
    angle_restr_constant=angle_restr_constant,
    nnp=nnp,
    implementation=implementation,
    name=name,
    base=f"{base}/{exp}/{run}/{name}",
    lambda_scheme=lambs,
    platform=platform,
    device=device,
    #device_index=device_index,
    every_nth_frame=every_nth_frame,
    #box_info = box_info,
    pdb_path = pdb_file_name_vac,
    ensemble=ensemble,
    control_param=control_param,
    precision=precision
    )

# if more than one run used for analysis
if sys.argv[2] == "runs_analysis":
    np.save(f'{base}/{exp}/u_kn_{len(lambs)}_{total_number_of_samples:.0f}_{name}_nr_runs_{len(runs_analysis)}_{perc}.npy', u_kn)
    np.save(f'{base}/{exp}/N_k_{total_number_of_samples:.0f}_{len(lambs)}_{name}_nr_runs_{len(runs_analysis)}_{perc}.npy', N_k)
else:
    np.save(f'{base}/{exp}/{run}/{name}/analysis/u_kn_{len(lambs)}_{total_number_of_samples:.0f}_{name}_nr_runs_{len(runs_analysis)}_{perc}.npy', u_kn)
    np.save(f'{base}/{exp}/{run}/{name}/analysis/N_k_{total_number_of_samples:.0f}_{len(lambs)}_{name}_nr_runs_{len(runs_analysis)}_{perc}.npy', N_k)

# ######################################################################################## debugging
# discard_frames=int((n_samples / 100) * 20) # discard first 20%

# Nk=int(np.ceil((n_samples-discard_frames)*(1/every_nth_frame))*len(lambs))

# u_kn = np.load(f'{base}/{exp}/{run}/{name}/analysis/u_kn_{len(lambs)}_{Nk}_{name}_nr_runs_1.npy') 
# N_k = np.load(f'{base}/{exp}/{run}/{name}/analysis/N_k_{Nk}_{len(lambs)}_{name}_nr_runs_1.npy')

# print(u_kn)
# print(N_k)
# ######################################################################################## debugging

# initialize the MBAR maximum likelihood estimate
from pymbar import MBAR
mbar = MBAR(u_kn, N_k)
#mbar = MBAR(u_kn, N_k, solver_protocol='robust')
r = mbar.compute_free_energy_differences()["Delta_f"][0][-1]

from taut_diff.constant import kBT
from openmm import unit
kBT_kcal = kBT.value_in_unit(unit.kilocalories_per_mole)
dG = r * kBT_kcal
print("##################################################")
print(f"Computed dG: {dG:.2f} kcal/mol") # convert from kBT to kcal/mol
print(f"Experimental dG: {experiment:.2f} kcal/mol")
print(f"Error to experiment: {(experiment - dG):.2f} kcal/mol")
print("##################################################")

if reweighting:

    # dG = dG(enol_no_restr --> enol_restr) +   A
    #      dG(enol_restr --> keto_restr) +      B
    #      dG(keto_restr --> keto_no_restr)     C

    dG_A, dE_A = compute_dG_flat_bottom_restraint(base=base,
                                    exp=exp,
                                    name=name,
                                    run=run,
                                    n_samples=n_samples,
                                    n_steps_per_sample=n_steps_per_sample,
                                    lambda_val=0,
                                    environment=environment,
                                    nnp=nnp,
                                    implementation=implementation,
                                    device=device,
                                    every_nth_frame=every_nth_frame,
                                    platform=platform,
                                    bond_restraints=bond_restraints,
                                    flat_bottom_restraints=flat_bottom_restraints,
                                    angle_restraints=angle_restraints,
                                    bond_restr_constant=bond_restr_constant,
                                    angle_restr_constant=angle_restr_constant,
                                    ensemble=ensemble,
                                    control_param=control_param)
                                    #device_index=device_index) 

    dG_C, dE_C = compute_dG_flat_bottom_restraint(base=base,
                                    exp=exp,
                                    name=name,
                                    run=run,
                                    n_samples=n_samples,
                                    n_steps_per_sample=n_steps_per_sample,
                                    lambda_val=1,
                                    environment=environment,
                                    nnp=nnp,
                                    implementation=implementation,
                                    device=device,
                                    every_nth_frame=every_nth_frame,
                                    platform=platform,
                                    bond_restraints=bond_restraints,
                                    flat_bottom_restraints=flat_bottom_restraints,
                                    angle_restraints=angle_restraints,
                                    bond_restr_constant=bond_restr_constant,
                                    angle_restr_constant=angle_restr_constant,
                                    ensemble=ensemble,
                                    control_param=control_param,)
                                   #device_index=device_index,) 

    reweighted_dG = dG_A + dG + dG_C

    # save as a compressed NumPy file
    # print(f"Saving dG corrections to {base}/{exp}/{run}/{name}/analysis/dG_restraints.npz")
    # np.savez(f'{base}/{exp}/{run}/{name}/analysis/dG_restraints.npz', dG_A=dG_A, dG_C=dG_C)
    print(f"Saving dG and dE corrections to {base}/{exp}/{run}/{name}/analysis/dG_restraints_{perc}.npz")
    np.savez(f'{base}/{exp}/{run}/{name}/analysis/dG_restraints_{perc}.npz', 
         dG_A=dG_A, dG_C=dG_C, 
         dE_A=dE_A, dE_C=dE_C)

    print("##################################################")
    print("AFTER REWEIGHTING")
    print(f"dG_A: {dG_A:.2f} kcal/mol")
    print(f"dG: {dG:.2f} kcal/mol")
    print(f"dG_C: {dG_C:.2f} kcal/mol")
    print("--------------------------------------------")
    print(f"Computed dG: {reweighted_dG:.2f} kcal/mol") # convert from kBT to kcal/mol
    print(f"Difference to previous dG: {(reweighted_dG - dG):.2f} kcal/mol")
    print(f"Experimental dG: {experiment:.2f} kcal/mol")
    print(f"Error to experiment: {(experiment - reweighted_dG):.2f} kcal/mol")
    print("##################################################")

    import seaborn as sns
    import matplotlib.pyplot as plt

    # Plot KDE distributions
    fontsize=18
    plt.figure(figsize=(10, 6))
    sns.kdeplot(dE_A, label=r'$\Delta$ E(enol_no_restr,enol_restr)', color='blue', fill=True, alpha=0.5)
    sns.kdeplot(dE_C, label=r'$\Delta$ E(keto_restr,keto_no_restr)', color='red', fill=True, alpha=0.5)

    plt.xlabel(r"$\Delta$ E[kcal/mol]", fontsize=fontsize)
    plt.ylabel("")
    plt.yticks([])
    plt.title(f"{run}", fontsize=fontsize)
    plt.legend(fontsize=fontsize)
    #plt.grid()
    plt.xticks(fontsize=fontsize)
    plt.yticks(fontsize=fontsize)

    path_save = f"{base}/{exp}/analysis/{name}_dE_reweighting_kde_{run}_{perc}.png"
    print(f"Saving KDE plot to {path_save}")
    plt.savefig(path_save)
    plt.close()
