import numpy as np
import matplotlib.pyplot as plt
from pymbar import MBAR
import seaborn as sns
import os

def plot_overlap_for_equilibrium_free_energy(
    N_k: list[np.array], 
    u_kn: list[np.ndarray], 
    name: str, 
    base: str, 
    lambs: list, 
    runs_analysis: list, 
    exp: str,
    Nk: int, # total number of analyzed samples
    run: str
):
    """
        Calculate the overlap for each state with each other state. The overlap is normalized to be 1 for each row.

        Args:
            N_k (np.array): number of samples for each state k
            u_kn (np.ndarray): each of the potential energy functions `u` describing a state `k` are applied to each sample `n` from each of the states `k`
            name (str): name of the system in the plot
    """

    ncols = len(N_k)

    if ncols > 1:
        fig, axes = plt.subplots(1, ncols, figsize=(8*ncols, 8), dpi=300)
        path_save = f"{base}/analysis/{name}_overlap_u_kn_{len(lambs)}_N_k_{Nk}.png"
    else:
        fig, axes = plt.subplots(figsize=(8, 8), dpi=300)
        path_save = f"{base}/{name}/{run}/analysis/{name}_overlap_u_kn_{len(lambs)}_N_k_{Nk}.png"

    for figure, (N_k, u_kn) in enumerate(zip(N_k, u_kn)):
        mbar = MBAR(u_kn, N_k)
        overlap = mbar.compute_overlap()["matrix"]

        if ncols > 1:
            ax = axes[figure]
        else:
            ax = axes
        sns.heatmap(
            overlap,
            cmap="Blues",
            linewidth=0.5,
            annot=True,
            fmt="0.3f",
            annot_kws={"size": 12},
            ax=ax,
            cbar = False,
        ) 
        tick_pos = np.arange(len(lambs)) + 0.5  # add 0.5 for centering

        ax.set_xticks(tick_pos, ["{:.2f}".format(label) for label in lambs], fontsize=12)
        ax.set_yticks(tick_pos, ["{:.2f}".format(label) for label in lambs], fontsize=12) 
        ax.set_title(f"{runs_analysis[figure]}", fontsize=12, weight="bold")

    plt.suptitle(f"Overlap matrix for {name}\n{exp}", fontsize=15)

    if not os.path.exists(f"{base}/analysis"):
        print("Creating directory:", f"{base}/analysis")
        os.makedirs(f"{base}/analysis")
    
    
    print(f"Saving overlap figure to {path_save}")
    plt.savefig(path_save)
    #plt.show()
    plt.close()


def plot_accumulated_free_energy(N_k, 
                                 u_kn, 
                                 nr_runs: int, 
                                 name: str,
                                 base: str,
                                 exp: str,
                                 Nk,
                                 lambs,
                                 run:str,
                                 experiment
                                 ):
    """
    Calculate the accumulated free energy along the mutation progress.

    Args:
        N_k (list of np.array): number of samples for each state k for each dataset
        u_kn (list of np.ndarray): potential energy functions for each dataset
        name (str): name of the system in the plot
    """

    fig, ax = plt.subplots(figsize=(8, 8), dpi=300)
    #plt.figure(figsize=[8, 8], dpi=300)
    results = []
    for i in range(nr_runs):
        mbar = MBAR(u_kn[i], N_k[i])
        free_energy_diff = mbar.compute_free_energy_differences()

        #ddG = free_energy_diff["Delta_f"][0][-1] * 0.5922
        #####
        from tautomer_ratios.constant import kBT
        from openmm import unit
        kBT_kcal = kBT.value_in_unit(unit.kilocalories_per_mole)
        ddG = free_energy_diff["Delta_f"][0][-1] * kBT_kcal
        ddG_error = free_energy_diff["dDelta_f"][0][-1] * kBT_kcal
        ####
        #print(f'Dataset {i+1}: dG = {ddG} +- {ddG_error}')
        r=f'run {i+1}: {ddG:.2f} +- {ddG_error:.2f} kcal/mol, error to exp: {(experiment - ddG):.2f} kcal/mol'
        results.append(r)
        
        print("##################################################")
        print(f"Run: {i+1}")
        print(f"Computed dG: {ddG} kcal/mol") # convert from kBT to kcal/mol
        print(f"Experimental dG: {experiment:.2f} kcal/mol")
        print(f"Error to experiment: {(experiment - ddG):.2f} kcal/mol")
        print("##################################################")

        r = free_energy_diff["Delta_f"]
        
        #r_kcal_per_mol = r * 0.5922
        r_kcal_per_mol = r * kBT_kcal
        #print(f"r result----------------------------------- {r_kcal_per_mol[0]}")
        y = r_kcal_per_mol[0]
        #y_error = free_energy_diff["dDelta_f"][0] * 0.5922
        y_error = free_energy_diff["dDelta_f"][0] * kBT_kcal
        x = [0.0, 0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 0.8, 0.9, 0.95, 1.0]
        if nr_runs == 1:
            run = run
        else:
            run = i+1
        plt.errorbar(x, y, yerr=y_error, label=f"run {run} dG +- stddev [kcal/mol]", marker='o', linestyle='-', color=f'C{i}')
        
    text_dG = ""
    for i, result in enumerate(results):
        # Add newline character only if there are more results
        if i < len(results) - 1:
            text_dG += f"{result}\n"
        else:
            text_dG += f"{result}"
    ax.text(
        0.02,
        0.98,
        text_dG,
        transform=ax.transAxes,
        fontsize=15,
        verticalalignment="top",
        horizontalalignment="left",  
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.5),
        linespacing=1.5,
        )

    plt.legend(loc='lower center')
    plt.suptitle(f"{exp}", fontsize=17)
    plt.title(f"Accumulated free energy along the mutation progress", fontsize=15)
    plt.ylabel("Free energy estimate [kcal/mol]", fontsize=15)
    plt.xlabel("lambda state (0 to 1)", fontsize=15)
    plt.xticks(x, rotation=45)
    plt.grid(True)

    if not os.path.exists(f"{base}/analysis"):
        print("Creating directory:", f"{base}/analysis")
        os.makedirs(f"{base}/analysis")
    
    if len(N_k) > 1:
        path_save = f"{base}/analysis/{name}_accumulated_dG_u_kn_{len(lambs)}_N_k_{Nk}.png"
    else:
        path_save = f"{base}/{name}/{run}/analysis/{name}_accumulated_dG_u_kn_{len(lambs)}_N_k_{Nk}.png"
    print(f"Saving overlap figure to {path_save}")
    plt.savefig(path_save)
    #plt.show()
    plt.close()


def plot_stddev_of_free_energy(N_k, 
                               u_kn, 
                               name: str, 
                               nr_runs:int,
                               base: str,
                               Nk,
                               exp: str,
                               lambs
                               ):
    """
    Plot the standard deviation of dG(lamb+1) - dG(lamb) throughout all the repetitions.

    Args:
        N_k (list of np.array): number of samples for each state k for each dataset
        u_kn (list of np.ndarray): potential energy functions for each dataset
        name (str): name of the system in the plot
        nr_runs (int): number of runs 
    """
    results = []

    plt.figure(figsize=[8, 9], dpi=300)

    for i in range(len(N_k)):
        print(results)
        dddG = []
        mbar = MBAR(u_kn[i], N_k[i])
        free_energy_diff = mbar.compute_free_energy_differences()

        for lamb in range(10):
            diff = free_energy_diff["Delta_f"][0][lamb+1]*0.5922 - free_energy_diff["Delta_f"][0][lamb]*0.5922
            dddG.append(diff)
        results.append(dddG)

    stddev = [np.std(col) for col in zip(*results)]
    y = stddev
    x = [r"0.00$\rightarrow$0.05", 
         r"0.05$\rightarrow$0.10", 
         r"0.10$\rightarrow$0.20", 
         r"0.20$\rightarrow$0.30", 
         r"0.30$\rightarrow$0.50", 
         r"0.50$\rightarrow$0.70", 
         r"0.70$\rightarrow$0.80", 
         r"0.80$\rightarrow$0.90", 
         r"0.90$\rightarrow$0.95", 
         r"0.95$\rightarrow$1.00"]

    plt.plot(x, y, '-o')
    #plt.legend()
    plt.axhline(y=0.1, linestyle='--', color="red")
    plt.title(f"Standard deviation of dG throughout {nr_runs} repeats ({name}) \n {exp}", fontsize=15)
    plt.ylabel("stddev(dG) [kcal/mol]", fontsize=15)
    plt.xlabel("lambda state", fontsize=15)
    plt.xticks(x, rotation=45)
    plt.grid(True)

    # if len(N_k) > 1:
    #     path_save = f"{base}/analysis/{name}_stddev_dG_u_kn_{len(lambs)}_N_k_{Nk}.png"
    # else:
    path_save = f"{base}/analysis/{name}_stddev_dG_u_kn_{len(lambs)}_N_k_{Nk}.png"
    print(f"Saving overlap figure to {path_save}")
    plt.savefig(path_save)
    #plt.show()
    plt.close()

        
def plot_weight_matrix(N_k, u_kn, name: str, nr_runs: int, base: str, Nk, exp: str, lambs, run:str):
    # import numpy as np
    # import matplotlib.pyplot as plt
    # from pymbar import MBAR
    
    ncols = len(N_k)
    height = 8  # Fixed height for the figure
    width = 12 * ncols  # dynamically adjust width to stretch the matrices

    # Create a single figure with subplots
    fig, axes = plt.subplots(1, ncols, figsize=(width, height), dpi=300)

    if ncols == 1:  # If there's only one run, ensure axes is treated as a list
        axes = [axes]

    for i, ax in enumerate(axes):

        mbar = MBAR(u_kn[i], N_k[i])
        W_nk = mbar.W_nk

        im = ax.imshow(W_nk, aspect='auto', interpolation='nearest', cmap='viridis')
        ax.set_title(f"Weight Matrix (Run {i+1}) \n {exp}", fontsize=15)
        ax.set_ylabel("Samples (n)", fontsize=12)
        ax.set_xlabel("Lambda States (k)", fontsize=12)
        ax.set_xticks(np.arange(len(lambs)))
        ax.set_xticklabels([f"{l:.2f}" for l in lambs], rotation=45)
        ax.tick_params(axis='both', which='major', labelsize=10)

    cbar = fig.colorbar(im, ax=axes, orientation='vertical', shrink=0.8)
    cbar.set_label("Weight (W_nk)", fontsize=12)

    path_save = f"{base}/{name}/{run}/analysis/{name}_weight_matrix_combined.png"
    print(f"Saving combined weight matrix figure to {path_save}")
    plt.savefig(path_save)
    # plt.show()
    plt.close()

