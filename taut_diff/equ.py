import numpy as np
from tqdm import tqdm
from typing import Tuple
from taut_diff.tautanipotential import create_system
from openmm.app import Simulation
from openmm import LangevinIntegrator
from openmm import unit
from openmm import MonteCarloBarostat
from openmm import HarmonicBondForce
from simtk.openmm import HarmonicAngleForce
from taut_diff.tautomers import get_indices, get_atoms_for_restraint

def get_sim(solv_system, 
            name: str,
            base: str,
            nnp:str, 
            lambda_val: float, 
            device, 
            platform, 
            bond_restraints: bool = False, 
            angle_restraints: bool = False,
            control_param: float = 4, # parameter for controling the restraint strength. 1 corresponds to the lambda scheme
            device_index: int = 0):
    
    from taut_diff.constant import (temperature, 
                                    pressure, 
                                    collision_rate, 
                                    stepsize)
    
    system_topology = solv_system.topology
    # get indices for t1 and t2 that should be masked (with -1)
    t1_idx_mask = get_indices(tautomer="t1", ligand_topology=system_topology, device=device)
    t2_idx_mask = get_indices(tautomer="t2", ligand_topology=system_topology, device=device)
    # print(f"Mask indices for t1: {t1_idx_mask}")
    # print(f"Mask indices for t2: {t2_idx_mask}")

    # create the modified MLPotential (from openmm-ml-taut)
    # potential = MLPotential(name=nnp, 
    #                         lambda_val = lambda_val, 
    #                         t1_idx_mask=t1_idx_mask, 
    #                         t2_idx_mask=t2_idx_mask)
    
    # system = potential.createSystem(
    #     system_topology,
    #     implementation = "torchani"
    # )

    system = create_system(nnp_name=nnp,
                           topology=system_topology,
                           implementation="torchani",
                           lambda_val=lambda_val,
                           t1_idx_mask=t1_idx_mask,
                           t2_idx_mask=t2_idx_mask)

    integrator = LangevinIntegrator(temperature, 1 / collision_rate, stepsize)
    barostate = MonteCarloBarostat(pressure, temperature) 
    system.addForce(barostate) 

    # add restraints
    if bond_restraints:
            
        bond_force_t1 = get_bond_restraint(name=name, base=base, tautomer="t1", lambda_val=lambda_val, control_param=control_param)
        bond_force_t2 = get_bond_restraint(name=name, base=base, tautomer="t2", lambda_val=lambda_val, control_param=control_param)
        #system.addForce(restraints)
        system.addForce(bond_force_t1)
        system.addForce(bond_force_t2)

    if angle_restraints:
        angle_force_1 = get_angle_restraint(name=name, base=base, tautomer="t1", lambda_val=lambda_val, control_param=control_param)
        angle_force_2 = get_angle_restraint(name=name, base=base, tautomer="t2", lambda_val=lambda_val, control_param=control_param)

        system.addForce(angle_force_1) 
        system.addForce(angle_force_2)
    
    sim = Simulation(
    system_topology, 
    system, 
    integrator, 
    platform=platform,
    platformProperties={
                    "Precision": "mixed",
                    "DeviceIndex": str(device_index),
                },)
    
    return sim
    
def get_bond_restraint(name:str, base:str, tautomer: str, lambda_val: float, control_param: float):
    atom_1, atom_2, _, _ = get_atoms_for_restraint(name=name,  base=base, tautomer=tautomer)
    if tautomer == "t1":
        spring_constant = 100 * (1 - np.min([(1-lambda_val)*control_param,1]))
    elif tautomer == "t2":
        spring_constant = 100 * (1 - np.min([lambda_val*control_param,1]))
    # restraint_force = FlatBottomRestraintBondForce(spring_constant= spring_constant  * unit.kilocalories_per_mole / unit.angstrom**2,
    #                                             well_radius= 1.5 * unit.angstrom,
    #                                             restrained_atom_index1 = atom_1,  
    #                                             restrained_atom_index2 = atom_2)
    # restraint_force = HarmonicRestraintBondForce(spring_constant= spring_constant  * unit.kilocalories_per_mole / unit.angstrom**2,
    #                                             restrained_atom_index1 = atom_1,
    #                                             restrained_atom_index2 = atom_2)
    
    restraint = HarmonicBondForce()
    equilibrium_bond_length=1
    restraint.addBond(particle1=atom_1, 
                      particle2=atom_2, 
                      length=equilibrium_bond_length * unit.angstroms, 
                      k=spring_constant * unit.kilocalories_per_mole/unit.angstroms**2)
    restraint.setUsesPeriodicBoundaryConditions(True)
    print(f"Restraining the bond between atoms {atom_1+1} and {atom_2+1} with a harmonic restraint (with a spring constant of {spring_constant} kcal/mol/A^2 and an equilibrium bond length of {equilibrium_bond_length} A)")
    return restraint

def get_angle_restraint(name:str, base:str, tautomer:str, lambda_val: float, control_param:float):
    if tautomer == "t1":
        k = 10 * (1 - np.min([(1-lambda_val)*control_param,1]))
    elif tautomer == "t2":
        k = 10 * (1 - np.min([lambda_val*control_param,1]))
    atom_1, atom_2, atom_3, angle = get_atoms_for_restraint(name=name, base=base, tautomer=tautomer)
    angle_force = HarmonicAngleForce()
    angle_force.addAngle(particle1=atom_1, 
                            particle2=atom_2, 
                            particle3=atom_3, 
                            angle=angle * unit.radian, 
                            k=k * unit.kilocalories_per_mole / unit.radian**2) 
    angle_force.setUsesPeriodicBoundaryConditions(True)
    print(f"Restraining the angle defined by atoms {atom_1+1}, {atom_2+1} and {atom_3+1} ({np.degrees(angle):.2f} degrees) with a harmonic angle restraint (with a harmonic force constant of {k} kcal/mole/rad^2)")
    return angle_force


# adapted from https://github.com/wiederm/endstate_correction/blob/63b92ab2b25bd4272fa11c956663f7f70f81a11c/endstate_correction/equ.py
def _collect_equ_samples(
    trajs: list, every_nth_frame: int = 10, discard_frames:int = 0
) -> Tuple[list, np.array]:
    """Generate a dictionary with the number of samples per trajektory and 
    a list with all samples [n_1, n_2, ...] given a list of k trajectories with n samples.

    Args:
        trajs (list): list of trajectories
        every_nth_frame (int, optional): prune samples by taking only every nth sample. Defaults to 10.

    Returns:
        Tuple[list, np.array]: coordinates, N_k
    """
    
    coordinates = []
    N_k = np.zeros(len(trajs))
    print(f"Will discard {discard_frames} samples from the beginning of the trajectory (20%) and take only every {every_nth_frame}th frame...")
    # loop over lambda scheme and collect samples in nanometer
    for idx, traj in enumerate(trajs):
        print(f"Loading trajectory {idx+1}/{len(trajs)}")     
        xyz=traj.xyz
        xyz = xyz[discard_frames:]  # remove first 20%
        xyz = xyz[::every_nth_frame]  # take only every nth sample
        N_k[idx] = len(xyz)
        coordinates.extend([c_*unit.nanometer for c_ in xyz])
    number_of_samples = len(coordinates)
    print(f"Number of samples loaded: {number_of_samples}")
    return coordinates * unit.nanometer, N_k

def calculate_u_kn(
    trajs: list,  # list of trajectories
    solv_system,
    nnp: str,
    name: str,
    base: str,
    lambda_scheme,
    platform,
    device,
    device_index,
    discard_frames: int,
    every_nth_frame: int = 1,  # prune the samples further by taking only every nth sample
) -> np.ndarray:
    """
    Calculate the u_kn matrix to be used by the mbar estimator

    Args:
        trajs (list): list of trajectories
        sim (Simulation): simulation object
        every_nth_frame (int, optional): prune the samples further by taking only every nth sample. Defaults to 1.
        
    Returns:
        np.ndarray: u_kn matrix
    """
    from taut_diff.constant import kBT

    lambda_scheme = np.array(lambda_scheme)
    samples, N_k = _collect_equ_samples(trajs=trajs, every_nth_frame=every_nth_frame, discard_frames=discard_frames)  # collect samples

    u_kn = np.zeros(
        (len(N_k), int(N_k[0] * len(N_k))), dtype=np.float64
    )  # NOTE: assuming that N_k[0] is the maximum number of samples drawn from any state k
    samples = np.array(samples.value_in_unit(unit.nanometer))  # positions in nanometer
    
    for k, lamb in enumerate(lambda_scheme):
        print("Calculate Us for lambda = {:.2f}".format(lamb))
        sim = get_sim(solv_system=solv_system, 
                      name=name, 
                      base=base,
                      nnp=nnp, 
                      lambda_val=lamb, 
                      device=device,
                      platform=platform,
                      bond_restraints=True,
                      angle_restraints=True,
                      device_index=device_index)
        us = []
        for x in tqdm(range(len(samples))):
            sim.context.setPositions(samples[x])
            u_ = sim.context.getState(getEnergy=True).getPotentialEnergy()
            us.append(u_)
        us = np.array([u / kBT for u in us], dtype=np.float64)
        u_kn[k] = us

    # total number of samples
    total_nr_of_samples = 0
   
    for n in N_k:
        total_nr_of_samples += n

    assert total_nr_of_samples > 20  # make sure that there are samples present

    return (N_k, u_kn, total_nr_of_samples)

def plot_overlap_for_equilibrium_free_energy(
    N_k: np.array, u_kn: np.ndarray, n_samples: int, n_steps_per_sample: int, name: str, base: str
):
    """
    Calculate the overlap for each state with each other state. The overlap is normalized to be 1 for each row.

    Args:
        N_k (np.array): number of samples for each state k
        u_kn (np.ndarray): each of the potential energy functions `u` describing a state `k` are applied to each sample `n` from each of the states `k`
        name (str): name of the system in the plot
    """
    from pymbar import MBAR
    import matplotlib.pyplot as plt
    import seaborn as sns

    # initialize the MBAR maximum likelihood estimate

    mbar = MBAR(u_kn, N_k)
    plt.figure(figsize=[8, 8], dpi=300)
    overlap = mbar.compute_overlap()["matrix"]
    sns.heatmap(
        overlap,
        cmap="Blues",
        linewidth=0.5,
        annot=True,
        fmt="0.2f",
        annot_kws={"size": "small"},
    )
    plt.title(f"Overlap matrix for {name}", fontsize=15)
    plt.savefig(f"{base}/{name}/analysis/{name}_overlap_samples_{n_samples}_steps_{n_steps_per_sample}.png")
    plt.show()
    plt.close()