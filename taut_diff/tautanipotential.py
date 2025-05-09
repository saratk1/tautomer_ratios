import torch
import openmm
from typing import Iterable, Optional
import torchani
import openmmtorch
from typing import Tuple
from NNPOps.neighbors import getNeighborPairs
print("imported neighborpairs form nnpops------------------------------------------------------------------")

class ANIForce(torch.nn.Module):

    def __init__(self, model, species, periodic, implementation, lambda_val, t1_idx_mask, t2_idx_mask):
        
        super(ANIForce, self).__init__()
        self.model = model
        self.species = torch.nn.Parameter(species, requires_grad=False)
        #########################################################
        # check which species should be masked
        mask_t1 = torch.nn.Parameter(t1_idx_mask, requires_grad=False)
        # first, get all species
        self.t1_species = torch.nn.Parameter(
            species.clone().detach(), requires_grad=False
        )
        # mask dummy atom (defining topology of tautomer 2) with -1
        self.t1_species[:, mask_t1] = -1 # CHANGED
        #print(self.t1_species)
        # do the same for tautomer 2
        mask_t2 = torch.nn.Parameter(t2_idx_mask, requires_grad=False)
        self.t2_species = torch.nn.Parameter(
            species.clone().detach(), requires_grad=False
        )
        self.t2_species[:, mask_t2] = -1 # CHANGED
        #print(self.t2_species)
        #########################################################
        self.energyScale = torchani.units.hartree2kjoulemol(1)
        self.lambda_val = lambda_val 
        self.implementation = implementation 

        # if atoms is None:
        #     self.indices = None
        # else:
        #     self.indices = torch.tensor(sorted(atoms), dtype=torch.int64)
        if periodic:
            self.pbc = torch.nn.Parameter(torch.tensor([True, True, True], dtype=torch.bool), requires_grad=False)
        else:
            self.pbc = None

    def forward(self, positions, boxvectors: Optional[torch.Tensor] = None):

        positions = positions.to(torch.float32)
        
        # if self.indices is not None:
        #     positions = positions[self.indices]
        
        ##########################################################
        # combine species and positions of tautomer 1 and tautomer 2
        # stack concatenates a sequence of tensors along a new dimension.
        species_positions_tuple = (
            torch.stack((self.t1_species, self.t2_species)).squeeze(),
            10.0 * positions.repeat(2, 1).reshape(2, -1, 3),
        )
        # 
        # int(f"t2 species: {self.t2_species}")
        # print("positions", positions)
        # print("species positions tuple", species_positions_tuple)
        
        ##########################################################
        
        if boxvectors is None:
            #print("in vacuum")
            #print(species_positions_tuple)
            # if self.implementation== "nnpops":
            #     _, t1 = self.model((self.t1_species, 10.0*positions.unsqueeze(0))) 
            #     _, t2 = self.model((self.t2_species, 10.0*positions.unsqueeze(0))) 
            #     # print(f"t1 energy: {t1}")
            #     # print(f"t2 energy: {t2}")
            #     return self.energyScale * (
            #         (self.lambda_val * t2) + ((1 - self.lambda_val) * t1)
            #    )
            #elif self.implementation == "torchani":
            # _, t1 = self.model((self.t1_species, 10.0*positions.unsqueeze(0))) 
            # _, t2 = self.model((self.t2_species, 10.0*positions.unsqueeze(0))) 
            _, energy = self.model(species_positions_tuple)
            # # print("energy", energy)
            # # print(f"lambda val:", self.lambda_val)
            t1 = energy[0]
            t2 = energy[1]
            # # print("t1 energy:", t1)
            # # print("t2 energy:", t2)
            # # print("----------------------------------------")
            return self.energyScale * (
                (self.lambda_val * t2) + ((1 - self.lambda_val) * t1)
            )
           
        else:
            #print("periodic system-----------------------")
            boxvectors = boxvectors.to(torch.float32)
            
            ######################################################### 
            if self.implementation == "torchani":
                #print("here torchani----------------------------")
                positions = positions - torch.outer(torch.floor(positions[:,2]/boxvectors[2,2]), boxvectors[2])
                positions = positions - torch.outer(torch.floor(positions[:,1]/boxvectors[1,1]), boxvectors[1])
                positions = positions - torch.outer(torch.floor(positions[:,0]/boxvectors[0,0]), boxvectors[0])
            #########################################################

                ##########################################################
                # combine species and wrapped postions of tautomer 1 and tautomer 2
                species_positions_tuple = (
                    torch.stack((self.t1_species, self.t2_species)).squeeze(),
                    10.0 * positions.repeat(2, 1).reshape(2, -1, 3),
                )
                #print("species positions tuple: ", species_positions_tuple)
                ##########################################################
                
                #_, energy_full = self.model((self.species, 10.0*positions.unsqueeze(0)), cell=10.0*boxvectors, pbc=self.pbc)
                _, energy = self.model(species_positions_tuple, cell=10.0*boxvectors, pbc=self.pbc) 
                #print("energy-----------------", energy)
                
                #########################################################################
                # get the energy of tautomer 1 and tautomer 2
                #print(energy[0])
                t1 = energy[0]
                t2 = energy[1]
                # print(f"t1: {float(t1.detach())}")
                # print(f"t2: {float(t2.detach())}")
                # print(f"t1 and t2 added: {float(t1.detach())+float(t2.detach())}")
                
                return self.energyScale * (
                    (self.lambda_val * t2) + ((1 - self.lambda_val) * t1)
                )
            
            elif self.implementation == "nnpops":
                #print("here nnpops----------------------------")
                # _, t1 = self.model((self.t1_species, 10.0*positions.unsqueeze(0)), cell=10.0*boxvectors, pbc=self.pbc)
                # _, t2 = self.model((self.t2_species, 10.0*positions.unsqueeze(0)), cell=10.0*boxvectors, pbc=self.pbc)
                #print(t1, t2)
                species_positions_tuple = (
                    torch.stack((self.t1_species, self.t2_species)).squeeze(),
                    10.0 * positions.repeat(2, 1).reshape(2, -1, 3),
                )
                _, energy = self.model(species_positions_tuple, cell=10.0*boxvectors, pbc=self.pbc) 
                t1 = energy[0]
                t2 = energy[1]

                return self.energyScale * (
                    (self.lambda_val * t2) + ((1 - self.lambda_val) * t1)
                )

        # print(self.lambda_val)
        # return_value = self.energyScale * (
        #     torch.abs((t2) -  t1)
        # )
        # print(f"return: {return_value}")
        # interpolate between potential of tautomer 1 and tautomer 2 with lambda
        # return self.energyScale * (
        #     (self.lambda_val * t2) + ((1 - self.lambda_val) * t1)
        # )
        ##########################################################################
        #return self.energyScale*energy

class MACEForce(torch.nn.Module):
    """
    MACEForce class to be used with TorchForce.

    Parameters
    ----------
    model : torch.jit._script.RecursiveScriptModule
        The compiled MACE model.
    dtype : torch.dtype
        The precision with which the model will be used.
    energyScale : float
        Conversion factor for the energy, viz. eV to kJ/mol.
    lengthScale : float
        Conversion factor for the length, viz. nm to Angstrom.
    indices : torch.Tensor
        The indices of the atoms to calculate the energy for.
    returnEnergyType : str
        Whether to return the interaction energy or the energy including the self-energy.
    inputDict : dict
        The input dictionary passed to the model.
    """

    def __init__(
        self,
        model: torch.jit._script.RecursiveScriptModule,
        nodeAttrs: torch.Tensor,
        atoms: Optional[Iterable[int]],
        periodic: bool,
        dtype: torch.dtype,
        returnEnergyType: str,
        d1: int, # NEW
        d2: int,
        lambda_val
    ) -> None:
        """
        Initialize the MACEForce.

        Parameters
        ----------
        model : torch.jit._script.RecursiveScriptModule
            The MACE model.
        nodeAttrs : torch.Tensor
            The one-hot encoded atomic numbers.
        atoms : iterable of int
            The indices of the atoms. If ``None``, all atoms are included.
        periodic : bool
            Whether the system is periodic.
        dtype : torch.dtype
            The precision of the model.
        returnEnergyType : str
            Whether to return the interaction energy or the energy including the self-energy.
        """
        super(MACEForce, self).__init__()

        self.dtype = dtype
        self.model = model.to(self.dtype)
        self.energyScale = 96.4853
        self.lengthScale = 10.0
        self.returnEnergyType = returnEnergyType

        if atoms is None:
            self.indices = None
        else:
            self.indices = torch.tensor(sorted(atoms), dtype=torch.int64)

        ######################################################################################################################
        self.mask_d1 = d1 # NEW
        self.mask_d2 = d2
        #print("self mask d1", self.mask_d1)
        #print("self mask d2", self.mask_d2)
        self.lambda_val = lambda_val
        ######################################################################################################################
        
        # Create the default input dict.
        self.register_buffer("ptr", torch.tensor([0, nodeAttrs.shape[0]], dtype=torch.long, requires_grad=False))
        self.register_buffer("node_attrs", nodeAttrs.to(self.dtype))
        self.register_buffer("batch", torch.zeros(nodeAttrs.shape[0], dtype=torch.long, requires_grad=False))
        self.register_buffer("pbc", torch.tensor([periodic, periodic, periodic], dtype=torch.bool, requires_grad=False))


    def _getNeighborPairs(
        self, positions: torch.Tensor, cell: Optional[torch.Tensor], mask_atom_idx:int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get the shifts and edge indices.

        Notes
        -----
        This method calculates the shifts and edge indices by determining neighbor pairs (``neighbors``)
        and respective wrapped distances (``wrappedDeltas``) using ``NNPOps.neighbors.getNeighborPairs``.
        After obtaining the ``neighbors`` and ``wrappedDeltas``, the pairs with negative indices (r>cutoff)
        are filtered out, and the edge indices and shifts are finally calculated.

        Parameters
        ----------
        positions : torch.Tensor
            The positions of the atoms.
        cell : torch.Tensor
            The cell vectors.

        Returns
        -------
        edgeIndex : torch.Tensor
            The edge indices.
        shifts : torch.Tensor
            The shifts.
        """
        
        ######################################################################################################################
        # print(f"positions {positions}")
        # print(f"self model r_max {self.model.r_max}")
        # print(f"cell {cell}")
        ######################################################################################################################
        
        # Get the neighbor pairs, shifts and edge indices.

        
        neighbors, wrappedDeltas, _, _ = getNeighborPairs(
            positions, self.model.r_max, -1, cell
        )
        #print(f"neigbors: {neighbors}")
        mask = (neighbors >= 0)
        ###################################################################################################################### # NEW
        #print(f"idx of atom that should be masked: {mask_atom_idx}")
        #print(f"mask BEFORE considering dummy atom: {mask}")
        
        # if an atom index to mask is provided, exclude pairs involving that atom index
        if mask_atom_idx is not None:
            mask &= (neighbors[0] != mask_atom_idx) & (neighbors[1] != mask_atom_idx)
            #print(f"mask AFTER considering dummy atom: {mask}")
        ######################################################################################################################
        neighbors = neighbors[mask].view(2, -1)
        wrappedDeltas = wrappedDeltas[mask[0], :]

        edgeIndex = torch.hstack((neighbors, neighbors.flip(0))).to(torch.int64)
        if cell is not None:
            deltas = positions[edgeIndex[0]] - positions[edgeIndex[1]]
            wrappedDeltas = torch.vstack((wrappedDeltas, -wrappedDeltas))
            shiftsIdx = torch.mm(deltas - wrappedDeltas, torch.linalg.inv(cell))
            shifts = torch.mm(shiftsIdx, cell)
        else:
            shifts = torch.zeros((edgeIndex.shape[1], 3), dtype=self.dtype, device=positions.device)

        return edgeIndex, shifts

    def forward(
        self, positions: torch.Tensor, boxvectors: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass of the model.

        Parameters
        ----------
        positions : torch.Tensor
            The positions of the atoms.
        box_vectors : torch.Tensor
            The box vectors.

        Returns
        -------
        energy : torch.Tensor
            The predicted energy in kJ/mol.
        """
        # Setup positions and cell.
        if self.indices is not None:
            positions = positions[self.indices]

        positions = positions.to(self.dtype) * self.lengthScale

        if boxvectors is not None:
            cell = boxvectors.to(self.dtype) * self.lengthScale
        else:
            cell = None
        
        #print(f"self mask d1: {self.mask_d1}")
        #print(f"self mask d2: {self.mask_d2}")
        
        ######################################################################################################################
        # Get the shifts and edge indices.
        edgeIndex_1, shifts_1 = self._getNeighborPairs(positions, cell, mask_atom_idx=self.mask_d1) # NEW
        edgeIndex_2, shifts_2 = self._getNeighborPairs(positions, cell, mask_atom_idx=self.mask_d2)

        # Update input dictionary.
        inputDict_1 = {
            "ptr": self.ptr,
            "node_attrs": self.node_attrs,
            "batch": self.batch,
            "pbc": self.pbc,
            "positions": positions,
            "edge_index": edgeIndex_1,
            "shifts": shifts_1,
            "cell": cell if cell is not None else torch.zeros(3, 3, dtype=self.dtype),
        }    
        inputDict_2 = {
            "ptr": self.ptr,
            "node_attrs": self.node_attrs,
            "batch": self.batch,
            "pbc": self.pbc,
            "positions": positions,
            "edge_index": edgeIndex_2,
            "shifts": shifts_2,
            "cell": cell if cell is not None else torch.zeros(3, 3, dtype=self.dtype),
        }   #NEW                

        # Predict the energy.
        energy_1 = self.model(inputDict_1, compute_force=False)[
            self.returnEnergyType
        ]
        energy_2 = self.model(inputDict_2, compute_force=False)[
            self.returnEnergyType
        ]

        assert (
            energy_1 is not None
        ), "The model did not return any energy. Please check the input."
        assert (
            energy_2 is not None
        ), "The model did not return any energy. Please check the input."

        #return energy * self.energyScale
        return self.energyScale * (
                    (self.lambda_val * energy_1) + ((1 - self.lambda_val) * energy_2) 
                )
        ######################################################################################################################

def create_system(nnp_name, topology, implementation, lambda_val, t1_idx_mask, t2_idx_mask, modelPath: str = None, removeCMMotion: bool = True):
# topology: openmm.app.Topology,
#         system: openmm.System,
#         atoms: Optional[Iterable[int]],
#         forceGroup: int,
#         precision: Optional[str] = None,
#         returnEnergyType: str = "interaction_energy",
    #if nnp_name.startswith("ani"):
    print(f"Loading ANI model: {nnp_name}......")
    # Create the TorchANI model.
    # `nnpops` throws error if `periodic_table_index`=False if one passes `species` as `species_to_tensor` from `element`
    
    # FROM anipotential.py addForces() ----------------
    _kwarg_dict = {'periodic_table_index': True}
    if nnp_name == 'ani2x':
        model = torchani.models.ANI2x(**_kwarg_dict)
    else:
        raise ValueError(f'Unsupported ANI model: {nnp_name}')
    # ---------------------------------------------------

    # Create the PyTorch model that will be invoked by OpenMM.
    atoms = list(topology.atoms())
    #print("atoms", atoms)
    # if atoms is not None: 
    #     includedAtoms = [includedAtoms[i] for i in atoms]
    species = torch.tensor([[atom.element.atomic_number for atom in atoms]])
    #print("species", species)

    if implementation == 'nnpops':
        #print("implementation: nnpops")
        try:
            from NNPOps import OptimizedTorchANI
            model = OptimizedTorchANI(model, species)
        except Exception as e:
            print(f"failed to equip `nnpops` with error: {e}")
    elif implementation == "torchani":
        pass # do nothing
    else:
        raise NotImplementedError(f"implementation {implementation} is not supported")

    #create openmm system
    system = openmm.System()
    print("CHECK if system was created: ", system)
    if topology.getPeriodicBoxVectors() is not None:
        system.setDefaultPeriodicBoxVectors(*topology.getPeriodicBoxVectors())
    for atom in topology.atoms():
        #print(atom, atom.element.mass)
        if atom.element is None:
            system.addParticle(0)
        else:
            system.addParticle(atom.element.mass)
    # NEW !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    if removeCMMotion:
        system.addForce(openmm.CMMotionRemover())

    # is_periodic...
    is_periodic = (topology.getPeriodicBoxVectors() is not None) or system.usesPeriodicBoundaryConditions()
    aniForce = ANIForce(model, species, is_periodic, implementation, lambda_val, t1_idx_mask, t2_idx_mask)
    # Convert it to TorchScript.
    module = torch.jit.script(aniForce)

    # Create the TorchForce and add it to the System.
    force = openmmtorch.TorchForce(module)
    force.setForceGroup(0) ############################################################# NEEDED?
    force.setUsesPeriodicBoundaryConditions(is_periodic)
    system.addForce(force)  
    print("CHECK if system still exists before returning it : ", system)

    return system
    
    
def create_system_mace(nnp_name, topology, implementation, lambda_val, d1, d2, modelPath: str = None, removeCMMotion: bool = True, precision: str = "single"):
    #elif nnp_name.startswith("mace"):
    ######################################################## from addforces
    import torch
    import openmmtorch
    
    
    ################################ NEW
    
    returnEnergyType = "interaction_energy"
    modelPath = None
    atoms = None # if None, all atoms are included
    precision = precision
    print(f"Simulation will be run in {precision} precision")
    forceGroup = 0
    
    
    ##################################
    

    try:
        from mace.tools import utils, to_one_hot, atomic_numbers_to_indices
        from mace.calculators.foundations_models import mace_off
    except ImportError as e:
        raise ImportError(
            f"Failed to import mace with error: {e}. "
            "Install mace with 'pip install mace-torch'."
        )
    try:
        from e3nn.util import jit
    except ImportError as e:
        raise ImportError(
            f"Failed to import e3nn with error: {e}. "
            "Install e3nn with 'pip install e3nn'."
        )
    try:
        from NNPOps.neighbors import getNeighborPairs
        #print("imported neighborpairs form nnpops------------------------------------------------------------------")
    except ImportError as e:
        raise ImportError(
            f"Failed to import NNPOps with error: {e}. "
            "Install NNPOps with 'conda install -c conda-forge nnpops'."
        )

    assert returnEnergyType in [
        "interaction_energy",
        "energy",
    ], f"Unsupported returnEnergyType: '{returnEnergyType}'. Supported options are 'interaction_energy' or 'energy'."

    # Load the model to the CPU (OpenMM-Torch takes care of loading to the right devices)
    if nnp_name.startswith("mace-off23"):
        size = nnp_name.split("-")[-1]
        assert (
            size in ["small", "medium", "large"]
        ), f"Unsupported MACE model: '{nnp_name}'. Available MACE-OFF23 models are 'mace-off23-small', 'mace-off23-medium', 'mace-off23-large'"
        model = mace_off(model=size, device="cpu", return_raw_model=True)
    elif nnp_name == "mace":
        if modelPath is not None:
            model = torch.load(modelPath, map_location="cpu")
        else:
            raise ValueError("No modelPath provided for local MACE model.")
    else:
        raise ValueError(f"Unsupported MACE model: {nnp_name}")

    # Compile the model.
    model = jit.compile(model)  

    # Get the atomic numbers of the ML region.
    includedAtoms = list(topology.atoms())
    if atoms is not None:
        includedAtoms = [includedAtoms[i] for i in atoms]
    atomicNumbers = [atom.element.atomic_number for atom in includedAtoms]

    # Set the precision that the model will be used with.
    modelDefaultDtype = next(model.parameters()).dtype
    if precision is None:
        dtype = modelDefaultDtype
    elif precision == "single":
        dtype = torch.float32
    elif precision == "double":
        dtype = torch.float64
    else:
        raise ValueError(
            f"Unsupported precision {precision} for the model. "
            "Supported values are 'single' and 'double'."
        )
    if dtype != modelDefaultDtype:
        print(
            f"Model dtype is {modelDefaultDtype} "
            f"and requested dtype is {dtype}. "
            "The model will be converted to the requested dtype."
        )

    # One hot encoding of atomic numbers
    zTable = utils.AtomicNumberTable([int(z) for z in model.atomic_numbers])
    nodeAttrs = to_one_hot(
        torch.tensor(
            atomic_numbers_to_indices(atomicNumbers, z_table=zTable),
            dtype=torch.long,
        ).unsqueeze(-1),
        num_classes=len(zTable),
    )
    ########################################################
    
    ##################################################################

    # FROM mlpotential.py

    #create openmm system
    system = openmm.System()
    if topology.getPeriodicBoxVectors() is not None:
        system.setDefaultPeriodicBoxVectors(*topology.getPeriodicBoxVectors())
    for atom in topology.atoms():
        if atom.element is None:
            system.addParticle(0)
        else:
            system.addParticle(atom.element.mass)
    # NEW !!!!!!!!!!!!!!!!!!!!!!!!! CMM
    if removeCMMotion:
        system.addForce(openmm.CMMotionRemover())


    #########################################################

    isPeriodic = (
        topology.getPeriodicBoxVectors() is not None
    ) or system.usesPeriodicBoundaryConditions()
    
    ##################################
    # d1=13
    # d2=15
    print(f"d1: {d1}")
    print(f"d2: {d2}")
    
    ##################################

    maceForce = MACEForce(
        model,
        nodeAttrs,
        atoms,
        isPeriodic,
        dtype,
        returnEnergyType,
        d1,
        d2,
        lambda_val
    )

    # Convert it to TorchScript.
    module = torch.jit.script(maceForce)

    # Create the TorchForce and add it to the System.
    force = openmmtorch.TorchForce(module)
    force.setForceGroup(forceGroup)
    force.setUsesPeriodicBoundaryConditions(isPeriodic)
    system.addForce(force)

    return system