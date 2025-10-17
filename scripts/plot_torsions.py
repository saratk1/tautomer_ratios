import matplotlib.pyplot as plt
import numpy as np
import mdtraj as md
import yaml
import os
import sys
from scipy.stats import gaussian_kde


config_path = sys.argv[1]  # path to config.yaml file
with open(config_path, "r") as file:
    config = yaml.safe_load(file)

# determine which runs to analyze
if sys.argv[2] in ["runs_all", "runs_individual"]:
    runs_analysis = config["analysis"]["runs_analysis"]
    run = runs_analysis[0]
else:
    runs_analysis = [sys.argv[2]]  # analyze only one run
    run = sys.argv[2]

print("Runs to analyze:", runs_analysis)

name = config["tautomer_systems"]["name"]
exp = config["exp"]
base = config["base"]
n_samples = config["sim_control_params"]["n_samples"]
n_steps_per_sample = config["sim_control_params"]["n_steps_per_sample"]
indices = config["analysis"]["torsion_indices"]
lambs = config["analysis"]["lambda_scheme"]

if lambs is None:
    lambs = [float(config["analysis"]["lambda_val"])]

if isinstance(lambs, (float, int)):
    lambs = [lambs]
elif isinstance(lambs, str):
    lambs = [float(x.strip()) for x in lambs.strip("[]").split(",")]
lambs = [float(l) for l in lambs]

print(f"Lambdas to plot: {lambs}")

colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]
suptitle_fontsize = 20
title_fontsize = 20

# Ensure analysis directory exists
global_analysis_dir = os.path.join(base, exp, "analysis")
os.makedirs(global_analysis_dir, exist_ok=True)

def plot_torsion(lambs, name, base, n_samples, n_steps_per_sample, indices, runs_analysis):
    adjusted_indices = [idx + 1 for idx in indices]

    n_lambs = len(lambs)
    n_cols = 3
    n_rows = int(np.ceil(n_lambs / n_cols))
    fig, axs = plt.subplots(n_rows, n_cols, figsize=(n_cols * 6, n_rows * 4))
    axs = axs.flatten()

    for i, lambda_val in enumerate(lambs):
        for run_idx, run in enumerate(runs_analysis):
            print(f"Loading trajectory for lambda = {lambda_val:.4f}, run = {run}...")

            traj_path = f"{base}/{exp}/{name}/{run}/{name}_samples_{n_samples}_steps_{n_steps_per_sample}_lamb_{lambda_val:.4f}.dcd"
            top_path = f"{base}/{exp}/{name}/{run}/{name}_hybrid_solv.pdb"

            if not os.path.exists(traj_path):
                print(f"Skipping lambda ={lambda_val:.4f} -- trajectory not found: {traj_path}")
                continue

            traj = md.load_dcd(traj_path, top=top_path)

            # discard first frames if desired
            discard_frames = 0
            traj = traj[discard_frames:]

            # compute torsion angle in degrees
            torsions = md.compute_dihedrals(traj, indices=[indices], periodic=True, opt=True)
            torsions_degrees = np.degrees(torsions).flatten()

            if len(torsions_degrees) < 2:
                print(f"Not enough data points for labmda ={lambda_val:.4f}, skipping KDE.")
                continue

            kde = gaussian_kde(torsions_degrees, bw_method=0.2)
            x = np.linspace(-180, 180, 1000)
            y = kde(x)

            color = colors[run_idx % len(colors)]
            axs[i].plot(x, y, label=f"{run}", color=color, linewidth=3)
            axs[i].fill_between(x, y, color=color, alpha=0.3)

        axs[i].set_title(f"labmda = {lambda_val:.4f}", fontsize=suptitle_fontsize)
        axs[i].set_xlim([-180, 180])
        axs[i].set_ylim(bottom=0)
        axs[i].set_yticklabels([])
        axs[i].legend(fontsize=suptitle_fontsize)
        axs[i].tick_params(axis='x', labelsize=20)  


    # Hide unused subplots if any
    for j in range(len(lambs), len(axs)):
        axs[j].axis("off")

    fig.text(0.5, 0.04, "Torsion Angle [degrees]", ha="center", fontsize=suptitle_fontsize)
    fig.text(0.04, 0.5, "Density", va="center", rotation="vertical", fontsize=suptitle_fontsize)
    fig.suptitle(f"{name} torsion profile ({adjusted_indices})", fontsize=title_fontsize, y=0.98)

    plt.tight_layout(rect=[0.05, 0.05, 1, 0.95])

    # Save figure
    save_dir = os.path.join(base, exp, name, run, "analysis")
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(
        save_dir,
        f"{name}_samples_{n_samples}_steps_{n_steps_per_sample}_torsion_{indices}.png"
    )
    print(f"Saving torsion profile plot to {save_path}")
    plt.savefig(save_path, dpi=300)
    plt.close(fig)


# -----------------------------
# Run plotting
# -----------------------------
plot_torsion(
    lambs=lambs,
    name=name,
    base=base,
    n_samples=n_samples,
    n_steps_per_sample=n_steps_per_sample,
    indices=indices,
    runs_analysis=runs_analysis,
)
