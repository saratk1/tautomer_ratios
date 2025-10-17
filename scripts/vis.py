# first sys.argv is device index, second specifies the path to the config.yaml file
import numpy as np
import yaml
import sys
from taut_diff.visualization import (
     plot_accumulated_free_energy,
     plot_overlap_for_equilibrium_free_energy,
     plot_stddev_of_free_energy,
     plot_weight_matrix
)

config_path = sys.argv[1] # path to config.yaml file
with open(config_path, 'r') as file:
        config = yaml.safe_load(file)

if sys.argv[2] == "runs_all" or sys.argv[2] == "runs_individual": # if all the samples from all available runs should be analyzed together or individually, but in one plot
    runs_analysis = config['analysis']['runs_analysis']
    run = runs_analysis[0]
else:
    runs_analysis = [sys.argv[2]] # analyze only one run
    run = sys.argv[2]
print("runs analysis", runs_analysis)
print(len(runs_analysis))
exp = config['exp']
name = config["tautomer_systems"]["name"]
nnp = config['sim_control_params']['nnp']
n_samples = config['sim_control_params']['n_samples']
n_steps_per_sample = config['sim_control_params']['n_steps_per_sample']
base = config['base']
experiment = config['tautomer_systems']['dG']
lambs = config['analysis']['lambda_scheme']
every_nth_frame = config['analysis']['every_nth_frame']

#discard_frames=int((n_samples / 100) * 20) # discard first 20%
discard_frames= 200

Nk=int(np.ceil((n_samples-discard_frames)*(1/every_nth_frame))*len(lambs)) # total number of analyzed samples, only for getting the correct file
#Nk=Nk[::every_nth_frame]# # number of samples
print(n_samples-discard_frames)
#print(f"Number of samples used for analysis: {n_samples}")
print(f"Nk: {Nk}")
print("len(lambs)", len(lambs))
print("nr runs", len(runs_analysis))

u_kn = []
N_k = []

if sys.argv[2] == "runs_all": # if all the samples from all available runs should be analyzed
    for run in runs_analysis: 
        u_run = np.load(f'{base}/{exp}/analysis/u_kn_{len(lambs)}_{Nk}_{name}_nr_runs_{len(runs_analysis)}.npy')
        N_run = np.load(f'{base}/{exp}/analysis/N_k_{Nk}_{len(lambs)}_{name}_nr_runs_{len(runs_analysis)}.npy')
        u_kn.append(u_run)
        N_k.append(N_run)

    nr_runs_save = len(runs_analysis)

elif sys.argv[2] == "runs_individual": #  or all runs ("runs_individual") should be analyzed individually, but in one plot
    for run in runs_analysis: 
        u_run = np.load(f'{base}/{exp}/{name}/{run}/analysis/u_kn_{len(lambs)}_{Nk}_{name}_nr_runs_{len(runs_analysis)}.npy')
        N_run = np.load(f'{base}/{exp}/{name}/{run}/analysis/N_k_{Nk}_{len(lambs)}_{name}_nr_runs_{len(runs_analysis)}.npy')
        u_kn.append(u_run)
        N_k.append(N_run)

    nr_runs_save = len(runs_analysis)

else: # if one run (eg. "run_01") should be analyzed
    u_run = np.load(f'{base}/{exp}/{name}/{run}/analysis/u_kn_{len(lambs)}_{Nk}_{name}_nr_runs_1_100.npy')
    N_run = np.load(f'{base}/{exp}/{name}/{run}/analysis/N_k_{Nk}_{len(lambs)}_{name}_nr_runs_1_100.npy')
    u_kn.append(u_run)
    N_k.append(N_run)
    #N_k.extend(N_run) ########################################################################################## debugging

    nr_runs_save = 1
# print(f"u_kn shape: {u_kn.shape}")  # Should be (K, N)
# print(f"N_k shape: {len(N_k)}")    # Should match the number of states

#u_kn = np.vstack(u_kn)  ########################################################################################## debugging
# print(f"u_kn shape: {u_kn.shape}")  # Should be (K, N)
# print(f"N_k shape: {len(N_k)}")    # Should match the number of states

# for run in runs_analysis: 
#     u_run = np.load(f'{base_load}/analysis/u_kn_{len(lambs)}_{Nk}_{name}_nr_runs_{nr_runs_load}.npy')
#     N_run = np.load(f'{base_load}/analysis/N_k_{Nk}_{len(lambs)}_{name}_nr_runs_{nr_runs_load}.npy')
#     u_kn.append(u_run)
#     N_k.append(N_run)

base_save = f"{base}/{exp}/"
print("len of ukn and nk", len(u_kn), len(N_k))
################################################################################# debugging
#print(N_k)
#print(u_kn)
# from pymbar import MBAR
# mbar = MBAR(u_kn, N_k)
# print(mbar)
#r = mbar.compute_free_energy_differences()["Delta_f"][0][-1]
#################################################################################

plot_overlap_for_equilibrium_free_energy(N_k=N_k, 
                                         u_kn=u_kn, 
                                         name=name, 
                                         base=base_save, 
                                         lambs=lambs, 
                                         runs_analysis=runs_analysis,
                                         exp=exp,
                                         Nk=Nk,
                                         run=run
                                         )
plot_accumulated_free_energy(N_k=N_k, 
                             u_kn=u_kn, 
                             nr_runs=nr_runs_save, 
                             name=name,
                             base=base_save,
                             exp=exp,
                             Nk=Nk,
                             lambs=lambs,
                             run=run,
                             experiment=experiment
                             )
if len(N_k)>1:
    plot_stddev_of_free_energy(N_k=N_k, 
                            u_kn=u_kn, 
                            name=name,
                            nr_runs=nr_runs_save, 
                            base=base_save,
                            exp=exp,
                            Nk=Nk,
                            lambs=lambs,
                            )

plot_weight_matrix(N_k=N_k, 
                           u_kn=u_kn, 
                           name=name,
                           nr_runs=nr_runs_save, 
                           base=base_save,
                           exp=exp,
                           Nk=Nk,
                           lambs=lambs,
                           run=run
                           )