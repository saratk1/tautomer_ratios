import pytest
import torch
from openmm import app, unit, Platform
from tautomer_ratios.equ import get_sim
from pathlib import Path
import tautomer_ratios
import yaml
from tautomer_ratios.tautomers import save_pdb
from tautomer_ratios.constant import temperature

REFERENCE_ENERGIES = {
    "ani2x": -952307.6884740161,
    "mace-off23-small": -6834.163159105894,
}

@pytest.mark.parametrize("nnp", ["ani2x", "mace-off23-small"])
def test_single_point_energy(nnp):
    """
    Check if energy of tautomer at lambda = 0 (== masking of keto hydrogen in the hybrid form) 
    corresponds to the energy of the pure enol form. 
    """

    package_base = Path(tautomer_ratios.__file__).resolve().parent
    base = package_base / "data"
    name = "tp_558"
    pdb_path_vacuum = f"{base}/test_simulation/{name}/run01/{name}_hybrid_TEST.pdb"
    environment = "vacuum"
    lambda_val = 0.0

    pdb = app.PDBFile(pdb_path_vacuum)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    platform_name = "CUDA" if torch.cuda.is_available() else "CPU"
    platform = Platform.getPlatformByName(platform_name)

    sim = get_sim(
        solv_system=pdb,
        environment=environment,
        nnp=nnp,
        lambda_val=lambda_val,
        device=device,
        platform=platform,
        bond_restraints=False,
        flat_bottom_restraints=False,
        angle_restraints=False,
        bond_restr_constant=0.0,
        angle_restr_constant=0.0,
        ensemble="npt",
        pdb_path_vacuum=pdb_path_vacuum,
        precision="single",
    )

    sim.context.setPositions(pdb.getPositions())
    state = sim.context.getState(getEnergy=True)
    energy = state.getPotentialEnergy().value_in_unit(unit.kilojoules_per_mole)

    print(f"{nnp} lambda={lambda_val:.1f} computed energy = {energy:.6f} kJ/mol")

    ref = REFERENCE_ENERGIES[nnp]
    diff = abs(energy - ref)
    print(f"reference = {ref:.6f} kJ/mol, diff = {diff:.6f} kJ/mol")

    assert isinstance(energy, float)
    assert diff < 0.001, f"{nnp} lambda={lambda_val} energy mismatch: {diff:.6f} kJ/mol"

@pytest.mark.slow
@pytest.mark.parametrize("nnp", ["ani2x", "mace-off23-small"])
def test_simulation(nnp):
    package_base = Path(tautomer_ratios.__file__).resolve().parent
    base = package_base / "data"
    config_path = f"{base}/test_simulation/config_tp_558.yaml" 
    with open(config_path, 'r') as file:
            config = yaml.safe_load(file)
            
    lambda_val = 0.5
    run = "run01"

    name = config["tautomer_systems"]["name"]
    smiles_t1 = config["tautomer_systems"]["smiles_t1"]
    smiles_t2 = config["tautomer_systems"]["smiles_t2"]
    exp = config["exp"]
    precision = config['sim_control_params']["precision"]
    n_samples = config['sim_control_params']['n_samples']
    n_steps_per_sample = config['sim_control_params']['n_steps_per_sample']
    nnp = config['sim_control_params']['nnp']
    ensemble = config['sim_control_params']['ensemble']
    bond_restraints = config['sim_control_params']['restraints']['bond']
    flat_bottom_restraints = config['sim_control_params']['restraints']['flat_bottom_bond']
    angle_restraints = config['sim_control_params']['restraints']['angle']
    bond_restr_constant = config['sim_control_params']['restraints']['bond_constant']
    angle_restr_constant = config['sim_control_params']['restraints']['angle_constant']
    environment = config['sim_control_params']['environment']
    box_length = config['sim_control_params']['box_length']
    minimize = config['sim_control_params']['minimize']
    
    pdb_file_name = f"{base}/{exp}/{name}/{run}/{name}_hybrid_solv_13A_TEST.pdb"
    pdb_file_name_vac = f"{base}/{exp}/{name}/{run}/{name}_hybrid_TEST.pdb"
    
    _ = save_pdb(name=name, 
                    smiles_t1=smiles_t1, 
                    smiles_t2=smiles_t2, 
                    base = f"{base}/{exp}/{name}/{run}/",
                    environment=environment,
                    box_length=box_length,
                    minimize=minimize)
    
    solv_system = app.PDBFile(pdb_file_name) 
    system_topology = solv_system.getTopology()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    platform = Platform.getPlatformByName("CUDA")  
        
    sim = get_sim(solv_system=solv_system, 
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
                pdb_path_vacuum=pdb_file_name_vac,
                precision=precision)
    
    sim.context.setPositions(solv_system.getPositions())
    sim.context.setVelocitiesToTemperature(temperature)
    sim.step(n_samples * n_steps_per_sample)
    
