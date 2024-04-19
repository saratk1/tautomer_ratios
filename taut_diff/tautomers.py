# this module contains functions, which generate a PDB file with the 
# solvated hybrid tautomer structure

# NOTE: all atom indices are modified to start at 1 (only for printing and visualization)
# so that they match indexing in the PDB file

import numpy as np
import sys
import random
import copy
import torch
import os
import yaml
from typing import List

from rdkit import Chem
from rdkit.Chem import AllChem, rdFMCS
from rdkit.Chem import Draw
from rdkit.Chem.Draw import rdMolDraw2D

from openmm import unit
from openmm.app import PDBFile
from openmm.vec3 import Vec3
from openmmtools.constants import kB

import torchani
import mdtraj as md
from pdbfixer import PDBFixer
import matplotlib.pyplot as plt

from PIL import Image
from io import BytesIO

#visualize both tautomers
#indexing starts at 1, as in the corresponding PDB file written by save_solv_pdb()
def visualize_tautomers(m1, m2, name, base):

    AllChem.Compute2DCoords(m1)
    AllChem.Compute2DCoords(m2)

    # start indexin with 1
    def adjust_atom_indices(mol):
        for atom in mol.GetAtoms():
            atom.SetProp("atomNote", str(atom.GetIdx() + 1))  

    adjust_atom_indices(m1)
    adjust_atom_indices(m2)

    def get_image(m):
        # formatting
        d = rdMolDraw2D.MolDraw2DCairo(1500, 1000)
        d.drawOptions().fixedFontSize = 90
        d.drawOptions().fixedBondLength = 110
        d.drawOptions().annotationFontScale = 0.7
        #d.drawOptions().addAtomIndices = True
        d.DrawMolecule(m)
        d.FinishDrawing()
        
        drawing = d.GetDrawingText()
        image = Image.open(BytesIO(drawing))
        return image
    
    img1 = get_image(m1)
    img2 = get_image(m2)

    fig, (ax1,ax2) = plt.subplots(1,2) 
    ax1.axis('off')
    ax2.axis('off')
    ax1.imshow(img1)
    ax2.imshow(img2)
    fig.suptitle(f"{name} tautomerization", fontsize = 15)
    fig.text(0.5, 0.9, 'Indexing the same as in PDB file (starting with 1)', ha='center', va='center', fontsize=12)
    fig.tight_layout(pad=0)

    composed_image = Image.new('RGB', (img1.width + img2.width, max(img1.height, img2.height)))
    composed_image.paste(img1, (0, 0))
    composed_image.paste(img2, (img1.width, 0))
    
    if not os.path.exists(f"{base}/{name}"):
        os.makedirs(f"{base}/{name}")

    plt.savefig(f"{base}/{name}/{name}.png")
    plt.close()  
    
def get_coordinates(m):

    AllChem.EmbedMolecule(m, enforceChirality=True)
    AllChem.UFFOptimizeMolecule(m)
    conf = m.GetConformer()
    coordinates = []

    for atom_idx in range(m.GetNumAtoms()):
        atom_pos = conf.GetAtomPosition(atom_idx)
        coordinates.append([atom_pos.x, atom_pos.y, atom_pos.z])

    coordinates = np.array([coordinates]) * unit.angstrom
    # check if shape is (number of molecules, number of atoms, xyz)
    assert coordinates.shape == (1, m.GetNumAtoms(), 3)
    return coordinates

# NOTE: large parts of the following code are taken from https://github.com/choderalab/neutromeratio and adjusted to fit our needs
# find indices of heavy atom acceptor, donor and idx of hydrogen that moves (idx of H in t1)
def find_idx(m1: int, m2: int):
    # for clarity: tautomer 2 is used only for finding MCS
    # NOTE: indices are modified, so that they start with 1 (& match the generated PDB file and the illustration)
    
    # find substructure and generate mol from substructure
    sub_m = rdFMCS.FindMCS(
        [m1, m2],
        bondCompare=Chem.rdFMCS.BondCompare.CompareOrder.CompareAny,
        maximizeBonds=True,
    )
    mcsp = Chem.MolFromSmarts(sub_m.smartsString, False)

    # the order of the substructure lists are the same for both
    # substructure matches => substructure_idx_m1[i] = substructure_idx_m2[i]
    substructure_idx_m1 = m1.GetSubstructMatch(mcsp)
    substructure_idx_m2 = m2.GetSubstructMatch(mcsp)

    # get idx of hydrogen that moves to new position
    hydrogen_idx_that_moves = -1
    atoms = ""  # atom element string
    for a in m1.GetAtoms():
        atoms += str(a.GetSymbol())

        if a.GetIdx() not in substructure_idx_m1:
            print("Atom index not in MCS:", a.GetIdx()+1)
            print("Index of atom that moves: {}.".format(a.GetIdx()+1))
            hydrogen_idx_that_moves = a.GetIdx()
            print("Check element that moves:" ,a.GetSymbol())

    # get idx of connected heavy atom which is the donor atom
    # there can only be one neighbor, therefor it is valid to take the first neighbor of the hydrogen
    donor = int(
        m1.GetAtomWithIdx(hydrogen_idx_that_moves).GetNeighbors()[0].GetIdx()
    )
    print("Index of atom (tautomer 1) that donates hydrogen: {}".format(donor+1))
    print("Element: ", m1.GetAtomWithIdx(hydrogen_idx_that_moves).GetNeighbors()[0].GetSymbol())

    # get idx of acceptor atom
    for i in range(len(substructure_idx_m1)):
        a1 = m1.GetAtomWithIdx(substructure_idx_m1[i])
        if a1.GetSymbol() != "H":
            a2 = m2.GetAtomWithIdx(substructure_idx_m2[i])
            # get acceptor - there are two heavy atoms that have
            # not the same number of neighbors
            a1_neighbors = a1.GetNeighbors()
            a2_neighbors = a2.GetNeighbors()
            acceptor_count = 0
            if (len(a1_neighbors)) != (len(a2_neighbors)):
                # we are only interested in the one that is not already the donor
                if substructure_idx_m1[i] == donor:
                    continue
                acceptor = substructure_idx_m1[i]
                print(
                    "Index of atom that accepts hydrogen (tautomer 1): {}".format(acceptor+1)
                )
                acceptor_count += 1
                if acceptor_count > 1:
                    raise RuntimeError(
                        "There are too many potential acceptor atoms."
                    )
    # summary of indices
    heavy_atom_hydrogen_donor_idx = donor
    hydrogen_idx = hydrogen_idx_that_moves
    heavy_atom_hydrogen_acceptor_idx = acceptor
    print("donor idx: ", heavy_atom_hydrogen_donor_idx+1, ", acceptor: ", heavy_atom_hydrogen_acceptor_idx+1, ", hydrogen: ", hydrogen_idx+1)

    return(heavy_atom_hydrogen_donor_idx, heavy_atom_hydrogen_acceptor_idx, hydrogen_idx)

# we need atom numbers of tautomer 1 (+ H -> hybrid structure) for ANI
def get_hybrid_atom_numbers(m): 
    atomic_nums = [atom.GetAtomicNum() for atom in m.GetAtoms()]
    #hybrid_atom_numbers = ligand_atom_numbers + "H"
    atomic_nums.append(1)
    return atomic_nums

def get_topology(m):
    n = random.randint(1, 10000000)
    # TODO: use tmpfile for this https://stackabuse.com/the-python-tempfile-module/ or io.StringIO
    _ = Chem.MolToPDBFile(m, f"tmp{n}.pdb")
    # get mdtraj topology
    ligand_topology = md.load(f"tmp{n}.pdb").topology
    os.remove(f"tmp{n}.pdb")
    return ligand_topology

def get_hybrid_topology(ligand_topology, heavy_atom_hydrogen_acceptor_idx, hydrogen_idx, heavy_atom_hydrogen_donor_idx):
    # for clarity: 
    # D1 = hydrogen defining topology of tautomer 1, but turns into a dummy atom, when defining topology of tautomer 2
    # D2 = hydrogen defining topology of tautomer 2, but turns into a dummy atom, when defining topology of tautomer 1
    # HET1 = heteroatom connected to D1 in tautomer 1
    # HET2 = heteroatom connected to D2 in tautomer 2

    # add a new hydrogen (dummy atom 2) to mdtraj ligand topology 
    hybrid_topology = copy.deepcopy(ligand_topology)
    dummy_atom = hybrid_topology.add_atom(
        "D2", md.element.hydrogen, hybrid_topology.residue(-1)
    )
    hybrid_topology.add_bond(
        hybrid_topology.atom(heavy_atom_hydrogen_acceptor_idx), dummy_atom
    ) 
    # modify name of dummy atom 1
    hybrid_topology.atom(hydrogen_idx).name = "D1"

    # modify name of donor heavy atom (eg hydrogen attached to carbon on tautomer 1 would be HET1)
    hybrid_topology.atom(heavy_atom_hydrogen_donor_idx).name = "HET1"
    hybrid_topology.atom(heavy_atom_hydrogen_acceptor_idx).name = "HET2"
    
    return hybrid_topology

# sample on a sphere to find the position of the dummy atom
def sample_spherical(acceptor:str, ndim=3):
    # sample a random direction
    unit_vector = np.random.randn(ndim)
    unit_vector /= np.linalg.norm(unit_vector, axis=0)
    # sample a random length
    #https://github.com/choderalab/neutromeratio/blob/4e3077acbab687d83a49fec7012bc88ed7ee8d76/neutromeratio/constants.py#L55-L60
    # for now, take these values
    acceptor_hydrogen_stddev_bond_length = 0.10 * unit.angstrom
    acceptor_hydrogen_equilibrium_bond_length = 1.02 * unit.angstrom

    effective_bond_length = (np.random.randn() * acceptor_hydrogen_stddev_bond_length +
                            acceptor_hydrogen_equilibrium_bond_length) 
    return (unit_vector * effective_bond_length)

def add_connect(pdb_filepath: str, hybrid_topology):
    # write correct CONECT record in pdb

    with open(pdb_filepath, 'r') as f:
        lines = f.readlines()

    new_lines = []

    for line in lines:
        if line.startswith('CONECT'):
            continue
        new_lines.append(line)

    sorted_bonds = []
    for bond in hybrid_topology.bonds:
        atom1_index = bond[0].index + 1  # PDB atom indices start from 1
        atom2_index = bond[1].index + 1  
        sorted_bonds.append((atom1_index, atom2_index))

    for atom1_index, atom2_index in sorted_bonds:
        new_lines.append(f'CONECT {atom1_index:>4}{atom2_index:>4}\n')

    # remove old END
    new_lines = [line for line in new_lines if not line.strip() == 'END']
    # add new END
    new_lines.append('END\n')

    with open(f"{pdb_filepath}", 'w') as f:
        f.writelines(new_lines)

def save_solv_pdb(name: str, smiles_t1: str, smiles_t2: str, base: str):

    # generate rdkit molecule objects
    mol1 = Chem.MolFromSmiles(smiles_t1)
    mol2 = Chem.MolFromSmiles(smiles_t2)
    # add hydrogens
    m1 = Chem.AddHs(mol1)
    m2 = Chem.AddHs(mol2)

    # visualize tautomers with atom indices
    print("Generating representation of tautomer 1 and tautomer 2...")
    visualize_tautomers(m1, m2, name, base)

    # get indices 
    print("Finding indices of donor, acceptor and hydrogen that moves...")
    heavy_atom_hydrogen_donor_idx, heavy_atom_hydrogen_acceptor_idx, hydrogen_idx = find_idx(m1, m2)

    # get hybrid atom numbers as a string
    hybrid_atom_numbers = get_hybrid_atom_numbers(m1)

    # get toplogy of tautomer 1
    ligand_topology = get_topology(m1)

    # define ANI model
    print("Define ANI model...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = torchani.models.ANI2x(periodic_table_index=True).to(device)

    # define species for ANI
    species = torch.tensor([hybrid_atom_numbers], device = device)

    # sample new dummy atom (H) coordinates on acceptor atom 
    min_e = 100.0 * unit.kilojoule_per_mole
    min_coordinates = []

    print("Find coordinates of dummy atom...")
    for _ in range(100):

        # get coordinates of acceptor atom and hydrogen that moves
        # get coordinates of tautomer 1
        coordinates = get_coordinates(m1)
        acceptor_coordinate = coordinates[0][heavy_atom_hydrogen_acceptor_idx]

        # generate new hydrogen atom position and define new hybrid coordinates
        ## TODO equilibrium bond lengths !!!! ####

        new_hydrogen_coordinate = acceptor_coordinate + sample_spherical(acceptor = m1.GetAtomWithIdx(heavy_atom_hydrogen_acceptor_idx).GetSymbol())
        new_hydrogen_coordinate = np.array([[new_hydrogen_coordinate]]) 
        hybrid_coord = (np.append(coordinates, new_hydrogen_coordinate, axis=1)) * unit.angstrom # changed from axis = 0

        # prepare hybrid coordinates for ANI
        coordinates_tensor = torch.tensor(
            hybrid_coord.value_in_unit(unit.angstrom),
            requires_grad=True,
            device=device,
            dtype=torch.float32,
        )

        # from _calculate_energy
        nr_of_mols = len(coordinates_tensor)
        batch_species = torch.stack(
            [species[0]] * nr_of_mols
        )  # species is a [1][1] tensor, afterwards it's a [1][nr_of_mols]

        if batch_species.size()[:2] != coordinates_tensor.size()[:2]:
            raise RuntimeError(
                f"Dimensions of coordinates: {coordinates_tensor.size()} and batch_species: {batch_species.size()} are not the same."
            )

        # energy evaluation
        energy_in_hartree = model(
            (
                batch_species, # batch_species
                coordinates_tensor) # ANI wants coordinates in angstrom
            ).energies

        #  convert energy from hartree to kT
        temperature = 300 * unit.kelvin
        kT = kB * temperature
        hartree_to_kJ_mol = 2625.499638
        kJ_mol_to_kT = (1.0 * unit.kilojoule_per_mole) / kT
        hartree_to_kT = hartree_to_kJ_mol * kJ_mol_to_kT

        energy_in_kT = energy_in_hartree * hartree_to_kT
        energy = np.array([e.item() for e in energy_in_kT]) * kT

        if energy < min_e:
            min_e = energy
            min_coordinates = hybrid_coord

    # define hybrid topology
    hybrid_topology = get_hybrid_topology(ligand_topology=ligand_topology, 
                                          heavy_atom_hydrogen_acceptor_idx=heavy_atom_hydrogen_acceptor_idx, 
                                          hydrogen_idx=hydrogen_idx, 
                                          heavy_atom_hydrogen_donor_idx=heavy_atom_hydrogen_donor_idx)

    # update hybrid toplogy
    pdb_filepath = f"{base}/{name}/{name}_hybrid.pdb"
    print("writing hybrid topology...")
    traj = md.Trajectory(min_coordinates.value_in_unit(unit.nanometer), hybrid_topology)
    modified_traj = traj.atom_slice(hybrid_topology.select("all"))

    #Save the modified trajectory with updated topology
    modified_traj.save_pdb(pdb_filepath)

    add_connect(pdb_filepath=pdb_filepath, hybrid_topology=hybrid_topology)

    # solvate
    print("solvating...")
    pdb = PDBFixer(filename=pdb_filepath)
    pdb.addSolvent(boxSize=Vec3(30,30,30)*unit.angstrom)

    PDBFile.writeFile(pdb.topology, pdb.positions, open(f'{base}/{name}/{name}_hybrid_solv.pdb', 'w'))
    print("writing solvated hybrid topology...")

def get_indices(tautomer: str, ligand_topology,device) :
    # get indices of tautomer 1 and tautomer 2
    indices = torch.zeros(ligand_topology.getNumAtoms(), device=device)
    # mask the hydrogen defining the respective other tautomer topology with a -1
    indices = torch.tensor([1 if atom.name == {"t1": "D2", "t2": "D1"}.get(tautomer) else 0 for atom in ligand_topology.atoms()])
    indices = indices.bool()
    return indices

def get_atoms_for_restraint(name:str, base:str, tautomer:str):
    if tautomer == "t1":
        het_atom = "HET1"
        dummy_atom = "D1"
    elif tautomer == "t2":
        het_atom = "HET2"
        dummy_atom = "D2"

    tautomer_system = md.load(f"{base}/{name}/{name}_hybrid.pdb")
    # get topology
    system_topology = tautomer_system.topology
    # get indices of dummy atom, atom bound to dummy atom
    atom_1=next(system_topology.atoms_by_name(dummy_atom)).index
    atom_2=next(system_topology.atoms_by_name(het_atom)).index
    # find index of the third atom
    for bond in system_topology.bonds:
        # check if atom1 of the bond is the het_atom
        if bond.atom1.name == het_atom:
            # check if it is not bound to the corresponding dummy atom and it is not a hydrogen atom
            if not bond.atom2.name == dummy_atom and bond.atom2.element.symbol != "H":
                atom_3=bond.atom2.index
                break
        # same for if atom2 of the bond is the het_atom
        elif bond.atom2.name == het_atom:
            if not bond.atom1.name == dummy_atom and bond.atom1.element.symbol != "H":
                atom_3 = bond.atom1.index
                break

    # compute angle between those three atoms
    angle = md.compute_angles(tautomer_system, [[atom_1, atom_2, atom_3]]) # in radians
    
    return atom_1, atom_2, atom_3, angle[0][0]

if __name__ == "__main__":

    with open(sys.argv[1], 'r') as file:
        config = yaml.safe_load(file)

    name = config["tautomer_systems"]["name"]
    smiles_t1 = config["tautomer_systems"]["smiles_t1"]
    smiles_t2 = config["tautomer_systems"]["smiles_t2"]
    base = config["base"]

    if not os.path.exists(f"{base}/{name}"):
        os.makedirs(f"{base}/{name}")
    # generate a hybrid structure and solvate (eg. for acetylacetone and the corresponding enol form):
    save_solv_pdb(name=name, smiles_t1=smiles_t1, smiles_t2=smiles_t2, base=base)