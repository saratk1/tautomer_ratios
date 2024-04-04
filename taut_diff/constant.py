from openmm import unit
from openmmtools.constants import kB

# define units and constants
distance_unit = unit.angstrom
time_unit = unit.femto * unit.seconds
speed_unit = distance_unit / time_unit

stepsize = 1 * time_unit
collision_rate = unit.pico * unit.second
temperature = 300 * unit.kelvin
pressure = 1 * unit.atmosphere

kBT = temperature * kB