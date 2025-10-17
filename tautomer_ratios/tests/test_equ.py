import numpy as np
import pytest
import torch 
from openmm.app import Simulation
from openmm import app, Platform, HarmonicBondForce, HarmonicAngleForce,CustomBondForce
from pathlib import Path
import mdtraj as md
import tautomer_ratios
from simtk import unit
from tautomer_ratios.equ import (get_sim,
                           get_bond_restraint, 
                           get_angle_restraint, 
                           get_flat_bottom_restraint,
                           calculate_u_kn,
                           _collect_equ_samples
                           )
from openmm.unit import Quantity, nanometer

@pytest.mark.parametrize(
    "nnp",
    [
        ("ani2x"),        
        ("mace-off23-small"),   
    ]
)
def test_simulation_object_creation(nnp):
    package_base = Path(tautomer_ratios.__file__).resolve().parent
    base = package_base / "data"

    sim = get_sim(
        solv_system=app.PDBFile(f"{base}/test_simulation/tp_558/run01/tp_558_hybrid_solv_13A_TEST.pdb"),
        environment="waterbox",
        nnp=nnp,
        lambda_val=0.5,
        device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
        platform=Platform.getPlatformByName("CUDA"),
        pdb_path_vacuum=f"{base}/test_simulation/tp_558/run01/tp_558_hybrid_TEST.pdb",
        ensemble="npt",
    )

    assert isinstance(sim, Simulation)
    assert hasattr(sim, "context")
    
##############################################################################################################
    
lambda_scheme = [0.0, 0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 0.8, 0.9, 0.95, 1.0]

expected_constants_t1 = [0.0, 1.0, 6.210526315789474, 16.631578947368425, 27.05263157894737,
                         47.89473684210527, 68.73684210526315, 79.15789473684211, 89.57894736842105,
                         94.78947368421052, 100.0]

expected_constants_t2 = [100.0, 94.78947368421052, 89.57894736842105, 79.15789473684211, 68.73684210526315,
                         47.89473684210527, 27.05263157894737, 16.631578947368425, 6.210526315789474,
                         1.0, 0.0]

@pytest.mark.parametrize("tautomer, expected_constants", [
    ("t1", expected_constants_t1),
    ("t2", expected_constants_t2),
])
def test_bond_restraint_spring_constants(tautomer, expected_constants):
    package_base = Path(tautomer_ratios.__file__).resolve().parent
    base = package_base / "data"
    pdb_path = f"{base}/test_simulation/tp_558/run01/tp_558_hybrid_TEST.pdb"
    environment = "waterbox"
    bond_restr_constant = 100.0

    actual_constants = []

    for lambda_val in lambda_scheme:
        restraint = get_bond_restraint(
            tautomer=tautomer,
            lambda_val=lambda_val,
            environment=environment,
            bond_restr_constant=bond_restr_constant,
            pdb_path=pdb_path,
        )

        assert isinstance(restraint, HarmonicBondForce)
        assert restraint.getNumBonds() == 1
        _, _, _, k = restraint.getBondParameters(0)

        assert unit.is_quantity(k)
        actual_constants.append(k.value_in_unit(unit.kilocalories_per_mole/unit.angstroms**2))

        assert restraint.usesPeriodicBoundaryConditions()

    for actual, expected in zip(actual_constants, expected_constants):
        assert pytest.approx(actual, abs=1e-4) == expected

#################################################################################################################

expected_constants_angle_t1 = [
    0,
    0.1,
    0.6210526316,
    1.6631578947,
    2.7052631579,
    4.7894736842,
    6.8736842105,
    7.9157894737,
    8.9578947368,
    9.4789473684,
    10
]


expected_constants_angle_t2 = [
    10.0,
    9.5263157895,
    9.0526315789,
    8.1052631579,
    7.1578947368,
    5.2631578947,
    3.3684210526,
    2.4210526316,
    1.4736842105,
    1.0,
    0
]

@pytest.mark.parametrize("tautomer, expected_constants", [
    ("t1", expected_constants_angle_t1),
    ("t2", expected_constants_angle_t2),
])
def test_angle_restraint_spring_constants(tautomer, expected_constants):
    package_base = Path(tautomer_ratios.__file__).resolve().parent
    base = package_base / "data"
    pdb_path = f"{base}/test_simulation/tp_558/run01/tp_558_hybrid_TEST.pdb"
    environment = "waterbox"
    angle_restr_constant = 10.0

    actual_constants = []

    for lambda_val in lambda_scheme:
        angle_force = get_angle_restraint(
            tautomer=tautomer,
            lambda_val=lambda_val,
            environment=environment,
            angle_restr_constant=angle_restr_constant,
            pdb_path=pdb_path,
        )

        assert isinstance(angle_force, HarmonicAngleForce)
        assert angle_force.getNumAngles() == 1
        _, _, _, _, k_val = angle_force.getAngleParameters(0)

        actual_constants.append(k_val.value_in_unit(unit.kilocalories_per_mole / unit.radian**2))
        assert angle_force.usesPeriodicBoundaryConditions()

    for actual, expected in zip(actual_constants, expected_constants):
        assert pytest.approx(actual, abs=1e-4) == expected

#################################################################################################################

@pytest.mark.parametrize("tautomer", ["t1", "t2"])
def test_flat_bottom_restraint(tautomer):
    package_base = Path(tautomer_ratios.__file__).resolve().parent
    base = package_base / "data"
    pdb_path = f"{base}/test_simulation/tp_558/run01/tp_558_hybrid_TEST.pdb"
    bond_restr_constant = 100.0 

    flat_bottom_force = get_flat_bottom_restraint(
        tautomer=tautomer,
        bond_restr_constant=bond_restr_constant,
        pdb_path=pdb_path,
    )

    assert isinstance(flat_bottom_force, CustomBondForce)

    assert flat_bottom_force.getNumBonds() == 1

    atom1, atom2, parameters = flat_bottom_force.getBondParameters(0)
    r0, K = parameters

    # bond length at which flat bottom harmonic restraint should be applied is at 1.5 A == 0.15 nm
    assert abs(r0 - 0.15) < 1e-6
    # check also if the constant was set correctly; convert kcal/mol/A^2 to kJ/mol/nm^2
    assert abs(K - bond_restr_constant * 4.184 * 100) < 1e-6
    
#################################################################################################################

@pytest.mark.parametrize("environment", ["vacuum", "waterbox"])
def test_collect_equ_samples(capsys, environment):
    package_base = Path(tautomer_ratios.__file__).resolve().parent
    base = package_base / "data"

    traj = md.load_dcd(f"{base}/test_simulation/tp_558/run01/tp_558_short_traj_TEST.dcd", 
                       top=f"{base}/test_simulation/tp_558/run01/tp_558_hybrid_solv_21A_TEST.pdb")
    trajs = [traj]

    coordinates, N_k, box_info = _collect_equ_samples(
        trajs=trajs,
        environment=environment,
        every_nth_frame=2,  
    )

    expected_n_samples = len(traj) // 2
    assert len(coordinates) == expected_n_samples
    assert hasattr(coordinates, "unit")
    assert coordinates.unit == unit.nanometer

    assert N_k.shape == (1,)
    assert N_k[0] == expected_n_samples

    if environment == "waterbox":
        assert len(box_info) == expected_n_samples
        assert isinstance(box_info[0], Quantity)
        assert box_info[0].unit == nanometer

    else:
        assert box_info == []

    captured = capsys.readouterr()
    assert "collect_equ_samples function" in captured.out
    assert "Setting the 0th entry of N_k" in captured.out

#################################################################################################################

@pytest.mark.slow
@pytest.mark.parametrize(
    "nnp",
    [
        ("ani2x"),
        ("mace-off23-small"),
    ],
)
@pytest.mark.parametrize("environment", ["waterbox"])
def test_calculate_u_kn_with_real_trajs(nnp, environment):
    package_base = Path(tautomer_ratios.__file__).resolve().parent
    base = package_base / "data"
    # Load a single trajectory and reuse it for 3 states
    traj = md.load_dcd(f"{base}/test_simulation/tp_558/run01/tp_558_short_traj_TEST.dcd",
                       top=f"{base}/test_simulation/tp_558/run01/tp_558_hybrid_solv_21A_TEST.pdb")
    trajs = [traj,traj,traj]

    lambda_scheme = [0.0, 0.5, 1.0]
    every_nth_frame = 1
    total_input_samples = sum(len(t) for t in trajs) // every_nth_frame
    expected_samples = total_input_samples 

    N_k, u_kn, total_samples = calculate_u_kn(
        trajs=trajs,
        solv_system=app.PDBFile(f"{base}/test_simulation/tp_558/run01/tp_558_hybrid_solv_21A_TEST.pdb"),
        environment=environment,
        bond_restraints=False,
        angle_restraints=False,
        flat_bottom_restraints=False,
        bond_restr_constant=0.0,
        angle_restr_constant=0.0,
        nnp=nnp,
        lambda_scheme=lambda_scheme,
        platform=Platform.getPlatformByName("CUDA"),
        device="cpu",
        ensemble="NVT",
        pdb_path_vacuum=f"{base}/test_simulation/tp_558/run01/tp_558_hybrid_TEST.pdb",
        precision="single",
        every_nth_frame=every_nth_frame,
    )

    assert isinstance(N_k, np.ndarray)
    assert isinstance(u_kn, np.ndarray)

    # Each row = lambda, each column = energy for sample
    assert u_kn.shape == (len(lambda_scheme), expected_samples)
    assert total_samples == expected_samples

    # N_k should show number of samples per lambda
    assert N_k.shape == (len(lambda_scheme),)
    assert N_k.sum() == total_input_samples

    # Check u_kn has finite values (i.e. no NaNs or infs)
    assert np.all(np.isfinite(u_kn))

