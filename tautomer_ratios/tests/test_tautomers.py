import numpy as np
import pytest
import os
from rdkit import Chem
from rdkit.Chem import AllChem
from openmm import unit
import mdtraj as md
import math
import tempfile
from openmm.app import PDBFile
from pathlib import Path
import tautomer_ratios
from tautomer_ratios.tautomers import (
    get_coordinates,
    get_atoms_for_restraint,
    get_hybrid_atom_numbers,
    get_topology,
    get_hybrid_topology,
    sample_spherical,
    find_idx,
    add_connect,
    get_indices
)
def test_get_coordinates():
    smiles = "Cc1nccc(O)c1" 
    mol = Chem.MolFromSmiles(smiles)
    mol = Chem.AddHs(mol)  

    coords = get_coordinates(mol)

    assert hasattr(coords, "unit")
    assert coords.unit == unit.angstrom

    coords_array = coords.value_in_unit(unit.angstrom)
    n_atoms = mol.GetNumAtoms()
    assert coords_array.shape == (1, n_atoms, 3)
    
    assert np.all(np.isfinite(coords_array))
    
########################################################################################################

@pytest.mark.parametrize("smiles_t1, smiles_t2", [
    ("Cc1nccc(O)c1", "CC(NC=C1)=CC1=O")  # tp_558 enol/keto
])
def test_find_idx(smiles_t1, smiles_t2):
    # Generate RDKit molecules
    m1 = Chem.MolFromSmiles(smiles_t1)
    m1 = Chem.AddHs(m1)
    AllChem.EmbedMolecule(m1, enforceChirality=True)
    AllChem.UFFOptimizeMolecule(m1)
    
    m2 = Chem.MolFromSmiles(smiles_t2)
    m2 = Chem.AddHs(m2)
    AllChem.EmbedMolecule(m2, enforceChirality=True)
    AllChem.UFFOptimizeMolecule(m2)

    donor_idx, acceptor_idx, hydrogen_idx = find_idx(m1, m2)

    print(f"Donor atom index: {donor_idx}")
    print(f"Acceptor atom index: {acceptor_idx}")
    print(f"Hydrogen atom index: {hydrogen_idx}")

    assert isinstance(donor_idx, int)
    assert isinstance(acceptor_idx, int)
    assert isinstance(hydrogen_idx, int)
    
    assert m1.GetAtomWithIdx(donor_idx).GetSymbol() == "O"       # donor is oxygen
    assert m1.GetAtomWithIdx(acceptor_idx).GetSymbol() == "N"    # acceptor is nitrogen
    assert m1.GetAtomWithIdx(acceptor_idx).GetSymbol() != "H"
    assert m1.GetAtomWithIdx(hydrogen_idx).GetSymbol() == "H"    # hydrogen that moves

    # check if the identified tautomer hydrogen is bound to the donor atom 
    assert donor_idx in [nbr.GetIdx() for nbr in m1.GetAtomWithIdx(hydrogen_idx).GetNeighbors()]
    
########################################################################################################

@pytest.fixture
def enol_mol():
    smiles = "Cc1nccc(O)c1" 
    mol = Chem.MolFromSmiles(smiles)
    mol = Chem.AddHs(mol)
    return mol

########################################################################################################

def test_get_hybrid_atom_numbers(enol_mol):
    nums = get_hybrid_atom_numbers(enol_mol)
    # should have one more than original number of atoms
    assert len(nums) == enol_mol.GetNumAtoms() + 1
    # last one should be hydrogen
    assert nums[-1] == 1
    # original atomic numbers should match
    for i, atom in enumerate(enol_mol.GetAtoms()):
        assert nums[i] == atom.GetAtomicNum()

########################################################################################################

def test_get_topology(enol_mol):
    top = get_topology(enol_mol)
    assert isinstance(top, md.Topology)
    assert top.n_atoms == enol_mol.GetNumAtoms()
    
########################################################################################################

def test_get_hybrid_topology(enol_mol):
    top = get_topology(enol_mol)
    donor_idx = 6
    acceptor_idx = 2
    hydrogen_idx = 13

    hybrid_top = get_hybrid_topology(top, acceptor_idx, hydrogen_idx, donor_idx)

    # check that dummy atom was added
    assert hybrid_top.n_atoms == top.n_atoms + 1
    # check atom names were modified correctly
    assert hybrid_top.atom(hydrogen_idx).name == "D1"
    assert hybrid_top.atom(donor_idx).name == "HET1"
    assert hybrid_top.atom(acceptor_idx).name == "HET2"

    # check that a bond exists between dummy atom and acceptor
    dummy_idx = hybrid_top.n_atoms - 1
    bonded_to_acceptor = [
        b.atom2.index if b.atom1.index == acceptor_idx else b.atom1.index
        for b in hybrid_top.bonds
        if acceptor_idx in (b.atom1.index, b.atom2.index)
    ]
    assert dummy_idx in bonded_to_acceptor

########################################################################################################

def test_sample_spherical():
    vec = sample_spherical()
    # check shape
    assert vec.shape == (3,)
    # check length is roughly around 1 Angstrom
    length = np.linalg.norm(vec.value_in_unit(unit.angstrom))
    assert 0.5 < length < 1.5  # some reasonable range
    
########################################################################################################

def test_add_connect():
    pdb_content = """\
ATOM      1  N   LIG     1       0.000   0.000   0.000
ATOM      2  C   LIG     1       1.000   0.000   0.000
ATOM      3  O   LIG     1       0.000   1.000   0.000
CONECT    1    2
CONECT    2    3
END
"""
    with tempfile.NamedTemporaryFile(mode="w+", delete=False, suffix=".pdb") as tmp:
        tmp.write(pdb_content)
        tmp_path = tmp.name

    traj = md.load(tmp_path)
    top = traj.topology

    add_connect(tmp_path, top)

    with open(tmp_path, "r") as f:
        lines = f.readlines()

    conect_lines = [line for line in lines if line.startswith("CONECT")]

    # there should be as many CONECT records as bonds in the topology
    expected_bonds = {(b[0].index + 1, b[1].index + 1) for b in top.bonds}
    written_bonds = {
        tuple(map(int, line.split()[1:])) for line in conect_lines
    }

    assert written_bonds == expected_bonds

    assert lines[-1].strip() == "END"
    
    os.remove(tmp_path)

###########################################################################################

@pytest.mark.parametrize("tautomer", ["t1", "t2"])
def test_get_indices(tautomer):
    package_base = Path(tautomer_ratios.__file__).resolve().parent
    pdb = PDBFile(f"{package_base}/data/test_simulation/tp_558/run01/tp_558_hybrid_solv_13A_TEST.pdb")
    top = pdb.topology
    device = "cpu"

    mask = get_indices(tautomer, top, device)

    # Mask length should match number of atoms
    assert mask.shape[0] == top.getNumAtoms()

    # Check the correct dummy atom is masked
    if tautomer == "t1":
        # D2 should be True
        d2_idx = next(a.index for a in top.atoms() if a.name == "D2")
        assert mask[d2_idx]
        # D1 should not be masked
        d1_idx = next(a.index for a in top.atoms() if a.name == "D1")
        assert not mask[d1_idx]
    else:
        # D1 should be True
        d1_idx = next(a.index for a in top.atoms() if a.name == "D1")
        assert mask[d1_idx]
        # D2 should not be masked
        d2_idx = next(a.index for a in top.atoms() if a.name == "D2")
        assert not mask[d2_idx]

##########################################################################################

@pytest.mark.parametrize("tautomer", ["t1", "t2"])
def test_get_atoms_for_restraint(tautomer):
    package_base = Path(tautomer_ratios.__file__).resolve().parent
    base = package_base / "data"
    pdb_path = f"{base}/test_simulation/tp_558/run01/tp_558_hybrid_TEST.pdb"
    name = "tp_558"

    atom_1, atom_2, atom_3, angle = get_atoms_for_restraint(tautomer, pdb_path)

    # atom indices should be integers
    assert isinstance(atom_1, int)
    assert isinstance(atom_2, int)
    assert isinstance(atom_3, int)

    # atom indices should be within number of atoms in the topology
    top = md.load(pdb_path).topology
    n_atoms = top.n_atoms
    assert 0 <= atom_1 < n_atoms
    assert 0 <= atom_2 < n_atoms
    assert 0 <= atom_3 < n_atoms

    # check the angle is scalar and within [0, pi] radians
    assert np.isscalar(angle)
    assert 0.0 <= angle <= math.pi

    # check that atom_1 is the dummy atom
    dummy_name = "D1" if tautomer == "t1" else "D2"
    assert top.atom(atom_1).name == dummy_name

    # check that atom_2 is the heavy atom
    het_name = "HET1" if tautomer == "t1" else "HET2"
    assert top.atom(atom_2).name == het_name

    # check that atom_3 is bonded to atom_2 and is not a hydrogen
    bonded_indices = []
    for bond in top.bonds:
        if bond[0].index == atom_2:
            bonded_indices.append(bond[1].index)
        elif bond[1].index == atom_2:
            bonded_indices.append(bond[0].index)
    
    assert atom_3 in bonded_indices
    assert top.atom(atom_3).element.symbol != "H"


