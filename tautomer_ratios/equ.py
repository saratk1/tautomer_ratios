import numpy as np
from openmm.app import (
    Simulation,
    DCDReporter,
    StateDataReporter,
    PDBFile,
    Topology,
)
from openmm import System
from openmmml import MLPotential
from openmm import app
from openmm import Platform
from openmm import LangevinIntegrator
from openmm import unit
from openmm import MonteCarloBarostat
from openmmtools.constants import kB
from openmmtools.forces import FlatBottomRestraintBondForce
import os
#from sys import stdout
import torch 
#import sys
#import openmm
import yaml

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# define units and constants
distance_unit = unit.angstrom
time_unit = unit.femto * unit.seconds
speed_unit = distance_unit / time_unit

stepsize = 1 * time_unit
collision_rate = unit.pico * unit.second
temperature = 300 * unit.kelvin
pressure = 1 * unit.atmosphere

def run_mfes(name: str, lambda_val: float, n_samples: float, n_steps_per_sample: int, nnp:str):

    ################################################## set up system ###################################################################

    solv_system = app.PDBFile(f'../testing/{name}/{name}_hybrid_solv.pdb')
    ligand_topology = solv_system.getTopology()
    atoms = ligand_topology.atoms()

    def get_indices(tautomer: str, ligand_topology,device) :
        # get indices of tautomer 1 and tautomer 2
        indices = torch.zeros(ligand_topology.getNumAtoms(), device=device)
        # mask the hydrogen defining the respective other tautomer topology with a -1
        indices = torch.tensor([1 if atom.name == {"t1": "D2", "t2": "D1"}.get(tautomer) else 0 for atom in ligand_topology.atoms()])
        indices = indices.bool()
        return indices

    t1_idx_mask = get_indices(tautomer="t1", ligand_topology=ligand_topology, device=device)
    t2_idx_mask = get_indices(tautomer="t2", ligand_topology=ligand_topology, device=device)

    ################################################ set up simulation ###################################################################

    torch._C._jit_set_nvfuser_enabled(False) # to prevent decrease of performance

    integrator = LangevinIntegrator(temperature, 1 / collision_rate, stepsize)
    platform = Platform.getPlatformByName("CUDA")
    potential = MLPotential(name=nnp, lambda_val = lambda_val, t1_idx_mask=t1_idx_mask, t2_idx_mask=t2_idx_mask)

    system = potential.createSystem(
        solv_system.getTopology(),
        implementation = "torchani"
    )

    barostate = MonteCarloBarostat(pressure, temperature)
    system.addForce(barostate)

    # add restraints
    # get indices of heavy atom-H for tautomer 1 and tautomer 2
    acceptor_t1 = next((idx for idx, atom in enumerate(ligand_topology.atoms()) if atom.name == "HET1"), None) 
    acceptor_t2 = next((idx for idx, atom in enumerate(ligand_topology.atoms()) if atom.name == "HET2"), None)
    dummy_t1 = next((idx for idx, atom in enumerate(ligand_topology.atoms()) if atom.name == "D1"), None)
    dummy_t2 = next((idx for idx, atom in enumerate(ligand_topology.atoms()) if atom.name == "D2"), None)

    # add C-H dummy atom restraint
    restraint_force_t1 = FlatBottomRestraintBondForce(spring_constant= 100  * unit.kilocalories_per_mole / unit.angstrom**2,
                                                well_radius= 2 * unit.angstrom,
                                                restrained_atom_index1 = acceptor_t1,  
                                                restrained_atom_index2 = dummy_t1)
    restraint_force_t2 = FlatBottomRestraintBondForce(spring_constant= 100  * unit.kilocalories_per_mole / unit.angstrom**2,
                                                well_radius= 2 * unit.angstrom,
                                                restrained_atom_index1 = acceptor_t2,  
                                                restrained_atom_index2 = dummy_t2)
    system.addForce(restraint_force_t1)
    system.addForce(restraint_force_t2)

    sim = Simulation(
        solv_system.getTopology(), 
        system, 
        integrator, 
        platform=platform,
        platformProperties={
                        "Precision": "mixed",
                        "DeviceIndex": str(1),
                    },)

    ################################################ data collection setup ###############################################################
    # define base path for storing data
    base = f"../testing/{name}/"

    # define where to store simulation info
    statereport_file = f"{base}/{name}_samples_{n_samples}_steps_{n_steps_per_sample}_lamb_{lambda_val:.4f}_report.csv"
    print(f"State report saved to: {statereport_file}")

    sim.reporters.append(
        StateDataReporter(
            statereport_file,
            reportInterval=n_steps_per_sample,
            step=True,
            time=True,
            potentialEnergy=True,
            totalEnergy=True,
            temperature=True,
            density=True,
            speed=True,
            elapsedTime=True,
            separator="\t",
        )
    )
    # define where to store samples
    trajectory_file = f"{base}/{name}_samples_{n_samples}_steps_{n_steps_per_sample}_lamb_{lambda_val:.4f}.dcd"
    print(f"Trajectory saved to: {trajectory_file}")

    sim.reporters.append(
        DCDReporter(
            trajectory_file,
            reportInterval=n_steps_per_sample,
        )
    )
    ################################################### sampling #####################################################################
    
    # set coordinates
    sim.context.setPositions(solv_system.getPositions())

    # perform sampling
    sim.step(n_samples * n_steps_per_sample)
    sim.reporters.clear()

######################################################################################################################################
    
if __name__ == "__main__":

    with open('config.yaml', 'r') as file:
        config = yaml.safe_load(file)

    name = config["tautomer_systems"]["name"]
    lambda_val = config['sim_control_params']['lambda_val']
    nnp = config['sim_control_params']['nnp']
    n_samples = config['sim_control_params']['n_samples']
    n_steps_per_sample = config['sim_control_params']['n_steps_per_sample']

    if not os.path.exists(f"../testing/{name}"):
        os.makedirs(f"../testing/{name}")

    run_mfes(name=name, 
            lambda_val=lambda_val, 
            nnp=nnp,
            n_samples=n_samples, 
            n_steps_per_sample=n_steps_per_sample)
    