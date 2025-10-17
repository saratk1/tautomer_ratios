import numpy as np
from tqdm import tqdm
from typing import Tuple
from taut_diff.alchemical_potential import create_system_ani, create_system_mace
from openmm.app import Simulation
from openmm import LangevinIntegrator, unit, MonteCarloBarostat, HarmonicBondForce, CustomBondForce
from simtk.openmm import HarmonicAngleForce
from openmm import app
import mdtraj as md
from taut_diff.tautomers import get_indices, get_atoms_for_restraint
from pymbar.other_estimators import exp as exponential_averaging

def get_sim(solv_system, 
            environment: str,
            nnp:str, 
            lambda_val: float, 
            device, 
            platform, 
            pdb_path_vacuum:str, 
            ensemble: str,
            bond_restraints: bool = False, 
            flat_bottom_restraints: bool = False,
            angle_restraints: bool = False,
            bond_restr_constant: float = 100,
            angle_restr_constant: float = 50,
            precision: str = "single"):
    """
    Generates a simulation object for the given system, environment, and simulation parameters.

    Args:
        solv_system (object): The solvated system object.
        environment (str): The environment of the simulation (e.g. "waterbox").
        nnp (str): The name of the neural network potential.
        lambda_val (float): The lambda value for the simulation.
        device (object): The device to use for the simulation.
        platform (object): The platform to use for the simulation.
        pdb_path (str): The path to the PDB file (vacuum -- connectivity needed).
        ensemble (str): The ensemble to use for the simulation (e.g. "npt").
        bond_restraints (bool, optional): Whether to add bond restraints.
        flat_bottom_restraints (bool, optional): Whether to add flat-bottom restraints.
        angle_restraints (bool, optional): Whether to add angle restraints.
        bond_restr_constant (float, optional): The bond restraint constant.
        angle_restr_constant (float, optional): The angle restraint constant.
        device_index (int, optional): The index of the device to use.
        precision (str, optional): The precision to use for the simulation.

    Returns:
        object: The simulation object.
    """
    
    print("Ensemble was set to: ", ensemble)
    from taut_diff.constant import (temperature, 
                                    pressure, 
                                    collision_rate, 
                                    stepsize)
    
    system_topology = solv_system.topology
    # get indices for t1 and t2 that should be masked (with -1)
    t1_idx_mask = get_indices(tautomer="t1", ligand_topology=system_topology, device=device) #bool
    t2_idx_mask = get_indices(tautomer="t2", ligand_topology=system_topology, device=device)
    #print("DEVICE", device)
    print(f"Indices that will be masked in t1: {t1_idx_mask}")
    print(f"Indices that will be masked in t2: {t2_idx_mask}")
    d1, _, _, _ = get_atoms_for_restraint(tautomer="t1", pdb_path=pdb_path_vacuum)
    d2, _, _, _ = get_atoms_for_restraint(tautomer="t2", pdb_path=pdb_path_vacuum)

    print("\n")
    print("############################################################")
    print("Creating system....")
    print("############################################################ \n")
    if nnp.startswith("ani"):
        system = create_system_ani(nnp_name=nnp,
                            topology=system_topology,
                            lambda_val=lambda_val,
                            t1_idx_mask=t1_idx_mask,
                            t2_idx_mask=t2_idx_mask,
                            )
    elif nnp.startswith("mace"):
        system = create_system_mace(nnp_name=nnp,
                            topology=system_topology,
                            lambda_val=lambda_val,
                            d1 = d1,
                            d2 = d2,
                            precision = precision)

    print("Stepsize is set to: ", stepsize)
    integrator = LangevinIntegrator(temperature, 1 / collision_rate, stepsize)
    if environment == "waterbox" and ensemble == "npt":
        barostat = MonteCarloBarostat(pressure, temperature)
        system.addForce(barostat) 

    # add restraints
    if flat_bottom_restraints:
        print("\n")
        print("############################################################")
        print("Adding flat bottom restraints....")
        print("############################################################ \n")
        flatt_bottom_bond_force_t1 = get_flat_bottom_restraint(
                                           tautomer="t1", 
                                           bond_restr_constant=bond_restr_constant,
                                           pdb_path=pdb_path_vacuum
                                           )
        flatt_bottom_bond_force_t2 = get_flat_bottom_restraint(
                                           tautomer="t2", 
                                           bond_restr_constant=bond_restr_constant,
                                           pdb_path=pdb_path_vacuum
                                           )
        # add the flat bottom restraint to a different force group so that it can be retrieved more easily
        flatt_bottom_bond_force_t1.setForceGroup(1) 
        flatt_bottom_bond_force_t2.setForceGroup(1)
        
        system.addForce(flatt_bottom_bond_force_t1)
        system.addForce(flatt_bottom_bond_force_t2)
        
    if bond_restraints:
        print("\n")
        print("############################################################")
        print("Adding restraints....")
        print("############################################################ \n")
        bond_force_t1 = get_bond_restraint(tautomer="t1", 
                                           lambda_val=lambda_val, 
                                           environment=environment, 
                                           bond_restr_constant=bond_restr_constant,
                                           pdb_path=pdb_path_vacuum
                                           )
        bond_force_t2 = get_bond_restraint(tautomer="t2", 
                                           lambda_val=lambda_val, 
                                           environment=environment,
                                           bond_restr_constant=bond_restr_constant,
                                           pdb_path=pdb_path_vacuum
                                           )
        system.addForce(bond_force_t1)
        system.addForce(bond_force_t2)

    if angle_restraints:
        angle_force_1 = get_angle_restraint(tautomer="t1", 
                                            lambda_val=lambda_val, 
                                            environment=environment,
                                            angle_restr_constant=angle_restr_constant,
                                            pdb_path=pdb_path_vacuum
                                            )
        angle_force_2 = get_angle_restraint(tautomer="t2", 
                                            lambda_val=lambda_val, 
                                            environment=environment,
                                            angle_restr_constant=angle_restr_constant,
                                            pdb_path=pdb_path_vacuum
                                            )

        system.addForce(angle_force_1) 
        system.addForce(angle_force_2)
    print(f"System uses periodic boundary conditions: {system.usesPeriodicBoundaryConditions()}")
    print(f"Number of constraints: {system.getNumConstraints()}")
    print(f"Forces acting on the system: {system.getForces()}")
    
    print("\n")
    print("############################################################")
    print("Generating simulation object....")
    print("############################################################ \n")
    sim = Simulation(
    system_topology, 
    system, 
    integrator, 
    platform=platform,
    platformProperties={
                    "Precision": "mixed",
                    #"DeviceIndex": str(device_index),
                },)
    print("CHECK if simulation object was created: ", sim)
    print("CHECK if sim.context is available: ", sim.context)
    return sim
    
def get_bond_restraint(tautomer: str, 
                       lambda_val: float, 
                       environment: str, 
                       bond_restr_constant: float, # TODO change name to max_bond_restr_constant
                       pdb_path: str
                       ):
    # get atom indices for bond restraint
    atom_1, atom_2, _, _ = get_atoms_for_restraint(tautomer=tautomer, pdb_path=pdb_path)
    if tautomer == "t1":
        k_start, k_end = 1, bond_restr_constant
        lambda_start, lambda_end = 0.05, 1.0

        if lambda_val == 0.0:  
            spring_constant = 0
        else:
            # creates the following scheme: 1.0 6.210526315789474 16.631578947368425 27.05263157894737 47.89473684210527 68.73684210526315 79.15789473684211 89.57894736842105 94.78947368421052 100.0
            spring_constant = k_start + (k_end - k_start) * (lambda_val - lambda_start) / (lambda_end - lambda_start)
            print(spring_constant)
    elif tautomer == "t2":
        k_start, k_end = bond_restr_constant, 1
        lambda_start, lambda_end = 0.0, 0.95

        if lambda_val == 1.0: 
            spring_constant = 0
        else:
            # creates the following scheme: 1.0 6.210526315789474 16.631578947368425 27.05263157894737 47.89473684210527 68.73684210526315 79.15789473684211 89.57894736842105 94.78947368421052 100.0
            spring_constant = k_start + (k_end - k_start) * (lambda_val - lambda_start) / (lambda_end - lambda_start)
    
    restraint = HarmonicBondForce()
    equilibrium_bond_length=1
    restraint.addBond(particle1=atom_1, 
                      particle2=atom_2, 
                      length=equilibrium_bond_length * unit.angstroms, 
                      k=spring_constant * unit.kilocalories_per_mole/unit.angstroms**2)
    if environment == "waterbox":
        restraint.setUsesPeriodicBoundaryConditions(True)
    #print(f"Restraining the bond between atoms {atom_1+1} and {atom_2+1} with a harmonic restraint (with a spring constant of {spring_constant} kcal/mole/A^2 and an equilibrium bond length of {equilibrium_bond_length} A)")
    return restraint

def get_flat_bottom_restraint(tautomer: str,  
                       bond_restr_constant: float,
                       pdb_path: str
                       ):
    # get atom indices for bond restraint
    atom_1, atom_2, _, _ = get_atoms_for_restraint(tautomer=tautomer, pdb_path=pdb_path)
    
    # create a CustomBondForce for the flat-bottom restraint
    flat_bottom_force = CustomBondForce("step(r - r0) * 0.5 * K * (r - r0)^2")

    # add parameters for the flat-bottom potential
    flat_bottom_force.addPerBondParameter("r0")  # flat-bottom threshold
    flat_bottom_force.addPerBondParameter("K")   

    # add a bond between two atoms (indices of atom_1 and atom_2) with specific parameters
    equilibrium_bond_length=1.5
    flat_bottom_force.addBond(atom_1, atom_2, [equilibrium_bond_length * unit.angstroms, bond_restr_constant* unit.kilocalories_per_mole/unit.angstroms**2])

    return flat_bottom_force

def get_angle_restraint(tautomer:str, 
                        lambda_val: float, 
                        environment: str, 
                        angle_restr_constant: float,
                        pdb_path:str):
    if tautomer == "t1":
        k_start, k_end = 0.1, angle_restr_constant
        lambda_start, lambda_end = 0.05, 1.0

        if lambda_val == 0.0:  # Special case for lambda = 1
            k = 0
        else:
            # Interpolate linearly
            k = k_start + (k_end - k_start) * (lambda_val - lambda_start) / (lambda_end - lambda_start)
    elif tautomer == "t2":
        k_start, k_end = angle_restr_constant, 1
        lambda_start, lambda_end = 0.0, 0.95

        if lambda_val == 1.0:  # Special case for lambda = 1
            k = 0
        else:
            # Interpolate linearly
            k = k_start + (k_end - k_start) * (lambda_val - lambda_start) / (lambda_end - lambda_start)
    # get atom indices and angle for angle restraint
    atom_1, atom_2, atom_3, angle = get_atoms_for_restraint(tautomer=tautomer, pdb_path=pdb_path)
    
    angle_force = HarmonicAngleForce()
    angle_force.addAngle(particle1=atom_1, 
                            particle2=atom_2, 
                            particle3=atom_3, 
                            angle=angle * unit.radian, 
                            k=k * unit.kilocalories_per_mole / unit.radian**2) 
    if environment == "waterbox":
        angle_force.setUsesPeriodicBoundaryConditions(True)
    print(f"Restraining the angle defined by atoms {atom_1+1}, {atom_2+1} and {atom_3+1} ({np.degrees(angle):.2f} degrees) with a harmonic angle restraint (with a harmonic force constant of {k} kcal/mole/rad^2)")
    return angle_force

# adapted from https://github.com/wiederm/endstate_correction/blob/63b92ab2b25bd4272fa11c956663f7f70f81a11c/endstate_correction/equ.py
def _collect_equ_samples(
    trajs: list,  environment: str, every_nth_frame: int = 10,
) -> Tuple[list, np.array, list]:
    """Generate a dictionary with the number of samples per trajektory and 
    a list with all samples [n_1, n_2, ...] given a list of k trajectories with n samples.

    Args:
        trajs (list): list of samples (consecutively from all lambda states)
        every_nth_frame (int, optional): prune samples by taking only every nth sample. Defaults to 10.

    Returns:
        Tuple[list, np.array, list]: coordinates, N_k, box_info
    """

    print(f"##################### collect_equ_samples function #####################")
    # list should have the same length as the final number of samples used (i.e. (initial number of samples - 20%) / {every_nth_frame})
    coordinates = []
    box_info = []
    # array of length: lambda states, each entry length: number of samples used for MBAR
    N_k = np.zeros(len(trajs))
    print(f"Initializing N_k as {N_k}, the length is the number of lambda states, i.e. {len(N_k)}. Eventually, each entry of N_k should be the length of the used samples for each lambda state.")
    print(f"Will take only every {every_nth_frame}th frame of the provided trajectory...")
    # loop over lambda scheme and collect samples in nanometer
    print("iterating over the cumulated trajectory (i.e. each entry holds the trajectory of one lambda state and gathering the coordinates.....)")
    for idx, traj in enumerate(trajs):
        print(f"Loading trajectory {idx+1}/{len(trajs)}")  
        print("length of the currently processed trajectory (should be the number of the originally collected frames - 20%): ", len(traj))
        print("Retrieving the xyz coordinates from the current frame")
        #xyz=traj.xyz
        #print(f"len(xyz) before discarding: {len(xyz)}")
        # Calculate pruned frame indices
        pruned_frame_indices = list(range(0, len(traj), every_nth_frame))
        #print(f"Pruned frame indices: {pruned_frame_indices}")
        
        # Prune xyz and box info using the pruned indices
        xyz = [traj.xyz[frame_idx] for frame_idx in pruned_frame_indices]
        if environment == "waterbox":
            boxes = [traj.openmm_boxes(frame_idx) for frame_idx in pruned_frame_indices]
        elif environment == "vacuum":
            boxes = None
        
        #xyz = xyz[::every_nth_frame]  # take only every nth sample
        print("-------------------------------------------------------------------------------------------------------------------------------------")
        print(f"len(xzy) after discarding samples in between: {len(xyz)}")
        print(f"Setting the {idx}th entry of N_k to the length of xyz....")
        print(f"N_k[{idx}] before: {N_k[idx]}")
        N_k[idx] = len(xyz)
        print(f"N_k[{idx}] after: {N_k[idx]}")
        print("-------------------------------------------------------------------------------------------------------------------------------------")
        #print(f"Formatting coordinates...")
        #coordinates.extend([c_ for c_ in xyz])
        coordinates.extend(xyz)
        if environment == "waterbox":
            box_info.extend(boxes)
    
    print(f"Pruning of samples completed. Performing some checks:") 
    print("-------------------------------------------------------------------------------------------------------------------------------------")
    number_of_samples = len(coordinates)
    #print(f"Box information: {box_info}")
    print(f"Length of box information (should be the same as the number of total samples used): {len(box_info)}")
    #print(f"Returned coordinates (samples) look like this (first entry as an example): {coordinates[0]}")
    print(f"Final number of samples loaded: {number_of_samples}")
    print(f"Final length of N_k: {len(N_k)}")
    #print("-------------------------------------------------------------------------------------------------------------------------------------")
    
    # Assertion for expected length of coordinates
    # initial_number_of_samples = len(trajs[0]) / 0.8  # Deduce the original number of samples
    # expected_number_of_samples = (initial_number_of_samples * 0.8) / every_nth_frame
    # print(f"Initial number of samples (deduced): {initial_number_of_samples}")
    # print(f"Expected number of samples after pruning: {expected_number_of_samples}")
    print("Returning samples with unit nanometer")
    #return coordinates, N_k, box_info
    return coordinates * unit.nanometer, N_k, box_info

def calculate_u_kn(
    trajs: list,  # list of trajectories
    solv_system,
    environment: str,
    bond_restraints: bool,
    angle_restraints: bool,
    flat_bottom_restraints: bool,
    bond_restr_constant: float,
    angle_restr_constant: float,
    nnp: str,
    lambda_scheme,
    platform,
    device,
    ensemble: str,
    pdb_path_vacuum: str,
    precision: str,
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
    #print("calculate ", len(trajs[0]))
    print("##################### calculate_u_kn function ###########################")

    lambda_scheme = np.array(lambda_scheme)
    print(f"The u_kn matrix will be calculated for the following lambda scheme {lambda_scheme}")
    print("\n")
    print("############################################################")
    print("Converting loaded samples to an appropriate format...")
    print("############################################################ \n")
    samples, N_k, box_info = _collect_equ_samples(trajs=trajs, environment=environment, every_nth_frame=every_nth_frame)  # collect samples
    #print("-------------------------------------------------------------------------------------------------------------------------------------")
    #print("Performing some checks on the loaded samples...")
    expected_nr_of_samples = (len(trajs[0])/every_nth_frame)*len(lambda_scheme) 
    print(f"Expected number of samples: {expected_nr_of_samples}, returned number of samples: {len(samples)}")
    assert len(samples) == expected_nr_of_samples, f"Expected {expected_nr_of_samples} samples, but got {len(samples)} samples."
    print(f"Length of first sample should be the number of molecules: {len(samples[0])}")
    print(f"Length of first entry of first sample should be 3 (x,y and z component): {len(samples[0][0])}")
    print("-------------------------------------------------------------------------------------------------------------------------------------")

    u_kn = np.zeros(
        (len(N_k), int(N_k[0] * len(N_k))), dtype=np.float64
    )  # NOTE: assuming that N_k[0] is the maximum number of samples drawn from any state k
    print(f"initializing u_kn matrix as (length of Nk, (first entry of N_k, ie number of samples, * length of Nk)) {u_kn}, dimensions: {u_kn.shape}")
    print(f"Converting the loaded samples to an array of just values (instead of an openmm Quantity)")
    print(f"Example: first loaded sample BEFORE conversion to the correct format: {samples[0][0]}")
    #samples = np.array(samples.value_in_unit(unit.nanometer))  # positions in nanometer
    print(f"Example: first loaded sample AFTER conversion to the correct format: {samples[0][0]}")
    
    print("\n")
    print("############################################################")
    print("Calculating the u_kn matrix...")
    print("############################################################ \n")
    
    for k, lamb in enumerate(lambda_scheme):
        print("Getting energies for every sample with potential energy function of lambda = {:.2f}".format(lamb))
        sim = get_sim(solv_system=solv_system, 
                      environment=environment,
                      nnp=nnp, 
                      lambda_val=lamb, 
                      device=device,
                      platform=platform,
                      ensemble=ensemble,
                      bond_restraints=bond_restraints,
                      flat_bottom_restraints=flat_bottom_restraints,
                      angle_restraints=angle_restraints,
                      bond_restr_constant=bond_restr_constant,
                      angle_restr_constant=angle_restr_constant,
                      pdb_path_vacuum=pdb_path_vacuum,
                      precision=precision,
                      )
        us = []
        #print("len samples in calculate u_s:", len(samples))
        for i in tqdm(range(len(samples))):
            #print(f"check units of sample: {samples[i]}")
            #print(f"Getting the coordianes for sample {i}")
            sim.context.setPositions(samples[i]) # TODO: check if this is in nm !!!
            
            if environment == "waterbox":
                #print(f"box info that goes into calculating u: {box_info[i]}")
                sim.context.setPeriodicBoxVectors(*box_info[i]) # index == sample; gives box info in as: Vec3(x=1.6, y=0.0, z=0.0) nm Vec3(x=0.0, y=1.6, z=0.0) nm Vec3(x=0.0, y=0.0, z=1.6) nm
            #print(f"Getting the energy of current sample")
            u_ = sim.context.getState(getEnergy=True).getPotentialEnergy()
            us.append(u_)
        #print(f"Dividing all accumulated energies by kBT")
        us = np.array([u / kBT for u in us], dtype=np.float64)
        print(f"Adding the calculated energies (with potential energy function lambda state {lamb}) to the final u_kn matrix")
        u_kn[k] = us

    # total number of samples
    total_nr_of_samples = 0
   
    for n in N_k:
        total_nr_of_samples += n

    assert total_nr_of_samples > 20  # make sure that there are samples present

    return (N_k, u_kn, total_nr_of_samples)

def compute_dG_flat_bottom_restraint(base:str,
                                     exp:str,
                                     name:str,
                                     run:str,
                                     n_samples:int,
                                     n_steps_per_sample:int,
                                     lambda_val:str,
                                     environment:str,
                                     nnp:str,
                                     device:str,
                                     every_nth_frame:int,
                                     platform:str,
                                     bond_restraints:bool,
                                     flat_bottom_restraints:bool,
                                     angle_restraints:bool,
                                     bond_restr_constant:int,
                                    angle_restr_constant:int,
                                    ensemble:str,
                                    ):

    if lambda_val == 0:
        tautomer_form = "enol"
    elif lambda_val == 1:
        tautomer_form = "keto"
    
    pdb_file_name = f'{base}/{exp}/{name}/{run}/{name}_hybrid_solv.pdb'
    pdb_file_name_vac = f'{base}/{exp}/{name}/{run}/{name}_hybrid.pdb'

    solv_system = app.PDBFile(pdb_file_name)
    discard_frames = 200
    trajs = []
    traj = md.load_dcd(
        f"{base}/{exp}/{name}/{run}/{name}_samples_{n_samples}_steps_{n_steps_per_sample}_lamb_{lambda_val:.4f}.dcd",
        top=pdb_file_name
    )
    traj=traj[discard_frames:]
    trajs.append(traj)

    # collect all samples of the corresponding endstate
    # _collect_equ_samples takes a list of trajectories, usually all lambda states; but here we're only interested in the endstate
    # _collect_equ_samples returns coordinates of each sample, N_k and the box vectors for each sample
    samples, N_k, box_info = _collect_equ_samples(trajs=trajs, environment=environment, every_nth_frame=every_nth_frame)  
    endstate_samples = np.array(samples.value_in_unit(unit.nanometer)) 
    print("number of collected endstate samples: ", len(endstate_samples))


    print(f"Creating simulation object for {tautomer_form} form with flat-bottom restraint")
    #simulation object with flat bottom restraint
    sim_endstate_flat_bottom = get_sim(solv_system=solv_system, 
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
                pdb_path_vacuum=pdb_file_name_vac,)

    print(f"Creating simulation object for {tautomer_form} form without flat-bottom restraint")
    # simulation object without flat bottom restraint
    sim_endstate_no_flat_bottom = get_sim(solv_system=solv_system, 
            environment=environment,
            nnp=nnp, 
            lambda_val=lambda_val, 
            device=device,
            platform=platform,
            bond_restraints=bond_restraints,
            flat_bottom_restraints=False, #set flat_bottom restraint to False!
            angle_restraints=angle_restraints,
            bond_restr_constant=bond_restr_constant,
            angle_restr_constant=angle_restr_constant,
            ensemble=ensemble,
            pdb_path_vacuum=pdb_file_name_vac,)

    w = 0.0
    ws = []

    for i in tqdm(range(len(endstate_samples))):
        sim_endstate_flat_bottom.context.setPositions(endstate_samples[i])
        sim_endstate_no_flat_bottom.context.setPositions(endstate_samples[i])
        if environment == "waterbox":
            sim_endstate_flat_bottom.context.setPeriodicBoxVectors(*box_info[i]) # first index == lambda state, second index == sample; gives box info in as: Vec3(x=1.6, y=0.0, z=0.0) nm Vec3(x=0.0, y=1.6, z=0.0) nm Vec3(x=0.0, y=0.0, z=1.6) nm
            sim_endstate_no_flat_bottom.context.setPeriodicBoxVectors(*box_info[i])

        u_endstate_flat_bottom = sim_endstate_flat_bottom.context.getState(getEnergy=True).getPotentialEnergy()
        u_endstate_no_flat_bottom = sim_endstate_no_flat_bottom.context.getState(getEnergy=True).getPotentialEnergy()
        
        if lambda_val == 0:
            #w += (u_endstate_flat_bottom - u_endstate_no_flat_bottom).value_in_unit(unit.kilojoule_per_mole)
            w = (u_endstate_flat_bottom - u_endstate_no_flat_bottom).value_in_unit(unit.kilojoule_per_mole)
        elif lambda_val == 1:
            #w += (u_endstate_no_flat_bottom - u_endstate_flat_bottom).value_in_unit(unit.kilojoule_per_mole)
            w = (u_endstate_no_flat_bottom - u_endstate_flat_bottom).value_in_unit(unit.kilojoule_per_mole) 
            
        ws.append(w)

    from taut_diff.constant import kBT
    kBT_kcal = kBT.value_in_unit(unit.kilocalories_per_mole)

    #print(type(ws))
    ws_array = np.array(ws)
    #print(ws_array)

    est = exponential_averaging(ws_array)
    if lambda_val == 0:
        print(f"dG({tautomer_form}_flatbottom_restraint, {tautomer_form}_no_flatbottom_restraint){est["Delta_f"]*kBT_kcal}")
    elif lambda_val == 1:
        print(f"dG({tautomer_form}_no_flatbottom_restraint, {tautomer_form}_flatbottom_restraint){est["Delta_f"]*kBT_kcal}")
    
    return est["Delta_f"]*kBT_kcal, np.array(ws) * kBT_kcal