import torch
import openmm
from typing import Iterable, Optional
import torchani
import openmmtorch

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
        self.t1_species[:, mask_t1] = -1

        # do the same for tautomer 2
        mask_t2 = torch.nn.Parameter(t2_idx_mask, requires_grad=False)
        self.t2_species = torch.nn.Parameter(
            species.clone().detach(), requires_grad=False
        )
        self.t2_species[:, mask_t2] = -1
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
        species_positions_tuple = (
            torch.stack((self.t1_species, self.t2_species)).squeeze(),
            10.0 * positions.repeat(2, 1).reshape(2, -1, 3),
        )
        ##########################################################
        
        if boxvectors is None:
            
            #_, energy = self.model((self.species, 10.0*positions.unsqueeze(0))) 
            _, energy = self.model(species_positions_tuple)
        else:
            boxvectors = boxvectors.to(torch.float32)
            
            ######################################################### 
            if self.implementation == "torchani":
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
            ##########################################################
            
            #_, energy = self.model((self.species, 10.0*positions.unsqueeze(0)), cell=10.0*boxvectors, pbc=self.pbc)
            _, energy = self.model(species_positions_tuple, cell=10.0*boxvectors, pbc=self.pbc) 
            
        #########################################################################
        # get the energy of tautomer 1 and tautomer 2
        t1 = energy[0]
        t2 = energy[1]

        # interpolate between potential of tautomer 1 and tautomer 2 with lambda
        return self.energyScale * (
            (self.lambda_val * t2) + ((1 - self.lambda_val) * t1)
        )
        ##########################################################################
        #return self.energyScale*energy

def create_system(nnp_name, topology, implementation, lambda_val, t1_idx_mask, t2_idx_mask):

    # Create the TorchANI model.
    # `nnpops` throws error if `periodic_table_index`=False if one passes `species` as `species_to_tensor` from `element`
    _kwarg_dict = {'periodic_table_index': True}
    if nnp_name == 'ani2x':
        model = torchani.models.ANI2x(**_kwarg_dict)
    else:
        raise ValueError(f'Unsupported ANI model: {nnp_name}')

    # Create the PyTorch model that will be invoked by OpenMM.
    atoms = list(topology.atoms())
    # if atoms is not None: 
    #     includedAtoms = [includedAtoms[i] for i in atoms]
    species = torch.tensor([[atom.element.atomic_number for atom in atoms]])

    if implementation == 'nnpops':
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
    if topology.getPeriodicBoxVectors() is not None:
        system.setDefaultPeriodicBoxVectors(*topology.getPeriodicBoxVectors())
    for atom in topology.atoms():
        if atom.element is None:
            system.addParticle(0)
        else:
            system.addParticle(atom.element.mass)

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

    return system