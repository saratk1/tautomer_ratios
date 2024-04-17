import numpy as np
from tqdm import tqdm
from typing import Tuple
from openmm.app import Simulation
from openmmml import MLPotential
from openmm import LangevinIntegrator
from openmm import unit
from openmm import MonteCarloBarostat
from openmmtools.forces import FlatBottomRestraintBondForce
from taut_diff.tautomers import get_indices

def get_sim(system_topology, 
            nnp:str, 
            lambda_val: float, 
            device, 
            platform, 
            restraints: bool = False, 
            device_index: int = 0):
    
    from taut_diff.constant import (temperature, 
                                    pressure, 
                                    collision_rate, 
                                    stepsize)
    
    # get indices for t1 and t2 that should be masked (with -1)
    t1_idx_mask = get_indices(tautomer="t1", ligand_topology=system_topology, device=device)
    t2_idx_mask = get_indices(tautomer="t2", ligand_topology=system_topology, device=device)
    # print(f"Mask indices for t1: {t1_idx_mask}")
    # print(f"Mask indices for t2: {t2_idx_mask}")

    # create the modified MLPotential (from openmm-ml-taut)
    potential = MLPotential(name=nnp, 
                            lambda_val = lambda_val, 
                            t1_idx_mask=t1_idx_mask, 
                            t2_idx_mask=t2_idx_mask)
    
    system = potential.createSystem(
        system_topology,
        implementation = "torchani"
    )

    integrator = LangevinIntegrator(temperature, 1 / collision_rate, stepsize)
    barostate = MonteCarloBarostat(pressure, temperature) 
    system.addForce(barostate) 

    # add restraints
    if restraints:
        # get indices of heavy atom-H for tautomer 1 and tautomer 2
        acceptor_t1 = next((idx for idx, atom in enumerate(system_topology.atoms()) if atom.name == "HET1"), None) 
        acceptor_t2 = next((idx for idx, atom in enumerate(system_topology.atoms()) if atom.name == "HET2"), None)
        dummy_t1 = next((idx for idx, atom in enumerate(system_topology.atoms()) if atom.name == "D1"), None)
        dummy_t2 = next((idx for idx, atom in enumerate(system_topology.atoms()) if atom.name == "D2"), None)
        #print(f"Restraint atom indices: acceptor_t1={acceptor_t1}, dummy_t1={dummy_t1}, acceptor_t2={acceptor_t2}, dummy_t2={dummy_t2}")

        # add C-H dummy atom restraint
        restraint_force_t1 = FlatBottomRestraintBondForce(spring_constant= 50  * unit.kilocalories_per_mole / unit.angstrom**2,
                                                    well_radius= 1.5 * unit.angstrom,
                                                    restrained_atom_index1 = acceptor_t1,  
                                                    restrained_atom_index2 = dummy_t1)
        restraint_force_t2 = FlatBottomRestraintBondForce(spring_constant= 50  * unit.kilocalories_per_mole / unit.angstrom**2,
                                                    well_radius= 1.5 * unit.angstrom,
                                                    restrained_atom_index1 = acceptor_t2, 
                                                    restrained_atom_index2 = dummy_t2) 
        # restraint_force_t1.setUsesPeriodicBoundaryConditions = True
        # restraint_force_t2.setUsesPeriodicBoundaryConditions = True

        system.addForce(restraint_force_t1)
        system.addForce(restraint_force_t2)
    
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
    system_topology,
    nnp: str,
    nr_lambda_states,
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

    lambda_scheme = np.linspace(0, 1, nr_lambda_states)  # equilibrium lambda scheme
    samples, N_k = _collect_equ_samples(trajs=trajs, every_nth_frame=every_nth_frame, discard_frames=discard_frames)  # collect samples

    u_kn = np.zeros(
        (len(N_k), int(N_k[0] * len(N_k))), dtype=np.float64
    )  # NOTE: assuming that N_k[0] is the maximum number of samples drawn from any state k
    samples = np.array(samples.value_in_unit(unit.nanometer))  # positions in nanometer
    
    for k, lamb in enumerate(lambda_scheme):
        print("Calculate Us for lambda = {:.1f}".format(lamb))
        sim = get_sim(system_topology=system_topology, 
                      nnp=nnp, 
                      lambda_val=lamb, 
                      device=device,
                      platform=platform,
                      restraints=False,
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

    return (N_k, u_kn)