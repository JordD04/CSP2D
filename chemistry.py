import copy

import numpy as np
import math

# 30 pixels = 1 angstroms

ANG2PIX = 30
ke = 14.3996

ATOM_TYPING = \
    {'LJ': {'epsilon'   : 1,
            'sigma'     : 1 * ANG2PIX,
            'colour'    : 'crimson',
            'radius'    : 0.5 * ANG2PIX,
            'mass'      : 1,
            'charge'    : 1},
     'H': {'epsilon'    : 0,
            'sigma'     : 1 * ANG2PIX,
            'colour'    : 'white',
            'radius'    : 0.3 * ANG2PIX,
            'mass'      : 1.008,
            'charge'    : 0.417},
     'O': {'epsilon'    : 0.6363864,
            'sigma'     : 3.1507 * ANG2PIX,
            'colour'    : 'red',
            'radius'    : 0.5 * ANG2PIX,
            'mass'      : 15.9994,
            'charge'    : -0.834}
    }


def calc_r(coords0, coords1):

    if len(coords1.shape) == 1:
        return np.linalg.norm(coords0 - coords1)
    else:
        return np.linalg.norm(coords0-coords1, axis=1)

def calc_atom_atom_vector(coords0, coords1):

    return coords1 - coords0

def calc_force_vector(force, vector):
    if isinstance(force, np.ndarray):
        force = np.column_stack((force, force))

    component_forces = (force * vector) / (abs(vector[0]) + abs(vector[1]))
    return component_forces

def calc_lj_force(r, epsilon, sigma):
    coefficient = 4 * epsilon
    repulsive = -12 * (sigma ** 12) * (r ** (-13))
    attractive = 6 * (sigma ** 6) * (r ** (-7))

    return 0.5 * coefficient * (repulsive + attractive)

def calc_lj_energy(r, epsilon, sigma):
    coefficient = 4 * epsilon
    repulsive = (sigma ** 12) * (r ** (-12))
    attractive = (sigma ** 6) * (r ** (-6))

    return 0.5 * coefficient * (repulsive - attractive)

def calc_coulomb_force(r, qprod):

    return - (ke * qprod) / (r ** 2)

def calc_coulomb_energy(r, qprod):

    return (ke * qprod) / r

def calc_total_force(r, qprod, epsilon, sigma):
    lj_force = calc_lj_force(r, epsilon, sigma)
    coulomb_force = calc_coulomb_force(r, qprod)

    return lj_force + coulomb_force

    # return calc_coulomb_force(r, qprod)
    # return calc_lj_force(r, epsilon, sigma)

def calc_total_energy(r, qprod, epsilon, sigma):
    lj_energy = calc_lj_energy(r, epsilon, sigma)
    coulomb_energy = calc_coulomb_energy(r, qprod)

    return lj_energy + coulomb_energy

    # return calc_coulomb_energy(r, qprod)
    # return calc_lj_energy(r, epsilon, sigma)

def limit_vector_mag(vector, max_step_size=3):
    mag = np.linalg.norm(vector)
    if mag > max_step_size:
        vector = vector * (max_step_size / mag)

    return vector

def get_rotation_matrix(theta):
    return np.array([[math.cos(theta), math.sin(theta)], [-math.sin(theta), math.cos(theta)]])


class Atom:
    def __init__(self, coords, type, colourless=False):
        self.coords = coords
        typing = ATOM_TYPING[type]
        self.epsilon = typing['epsilon']
        self.sigma = typing['sigma']
        self.q = typing['charge']
        if colourless:
            self.colour = 'black'
        else:
            self.colour = typing['colour']
        self.radius = typing['radius']
        self.type = type

        self.graphic_coords = np.asarray([coords[0] - self.radius, coords[1] - self.radius,
                                          coords[0] + self.radius, coords[1] + self.radius])

    def update_coords(self, vector, lattice_x, lattice_y, lattice_offset, ignore_wrapping=False):
        self.coords += vector

        if not ignore_wrapping:
            # check PBC
            wrapping_vector = np.asarray([0, 0])
            if self.coords[0] > (lattice_offset[0] + lattice_x):
                wrapping_vector[0] -= lattice_x
            if self.coords[0] < lattice_offset[0]:
                wrapping_vector[0] += lattice_x

            if self.coords[1] > (lattice_offset[1] + lattice_y):
                wrapping_vector[1] -= lattice_y
            if self.coords[1] < lattice_offset[1]:
                wrapping_vector[1] += lattice_y

            self.coords += wrapping_vector
        self.graphic_coords = np.asarray([self.coords[0] - self.radius, self.coords[1] - self.radius,
                                          self.coords[0] + self.radius, self.coords[1] + self.radius])


class Molecule:
    def __init__(self, atom_coords, atom_types):
        self.atoms = []
        self.num_atoms = len(atom_coords)
        self.total_mass = 0
        mass_coord_sum = np.array([0, 0])
        for atom_ind in range(self.num_atoms):
            coords = atom_coords[atom_ind]
            type = atom_types[atom_ind]
            mass = ATOM_TYPING[type]['mass']
            self.atoms.append(Atom(coords, type))
            self.total_mass += mass
            mass_coord_sum = np.sum(np.vstack((mass_coord_sum, (coords * mass))), axis=0)
        self.CoM = mass_coord_sum / self.total_mass

        self.get_atom_CoM_vectors(skip_CoM_update=True)
        self.gyration_radius = np.max(np.asarray([np.linalg.norm(vector) for vector in self.atom_CoM_vectors]))
        self.get_rotation_for_translation()

    @classmethod
    def init_water(cls, x=0, y=0, theta=0):
        O_coords = np.array([0, 0])
        H_coords0 = np.array([0.9572 * ANG2PIX, 0])
        H_coords1 = np.array([-0.2400 * ANG2PIX, 0.9266 * ANG2PIX])
        O_mass = ATOM_TYPING['O']['mass']
        H_mass = ATOM_TYPING['H']['mass']
        total_mass = O_mass + (2 * H_mass)
        CoM = ((O_coords * O_mass) + (H_coords0 * H_mass) + (H_coords1 * H_mass)) / total_mass

        all_coords = np.asarray([O_coords, H_coords0, H_coords1])
        # bring CoM to 0
        all_coords -= CoM

        rotation_matrix = get_rotation_matrix(theta)
        all_coords_T = all_coords.T
        new_coords = np.matmul(rotation_matrix, all_coords_T)
        all_coords = new_coords.T

        # apply offset
        all_coords += np.array([x, y])

        return cls(all_coords, ['O', 'H', 'H'])

    def translate_molecule(self, vector, lattice_x, lattice_y, lattice_offset):
        old_CoM = copy.deepcopy(self.CoM)
        self.CoM += vector

        wrapping_vector = np.asarray([0, 0])
        if self.CoM[0] > (lattice_offset[0] + lattice_x):
            wrapping_vector[0] -= lattice_x
        if self.CoM[0] < lattice_offset[0]:
            wrapping_vector[0] += lattice_x

        if self.CoM[1] > (lattice_offset[1] + lattice_y):
            wrapping_vector[1] -= lattice_y
        if self.CoM[1] < lattice_offset[1]:
            wrapping_vector[1] += lattice_y

        self.CoM += wrapping_vector

        resultant_vector = self.CoM - old_CoM
        for atom in self.atoms:
            atom.update_coords(resultant_vector, lattice_x, lattice_y, lattice_offset, ignore_wrapping=True)

    def rotate_molecule(self, rotation_matrix):
        all_coords = np.asarray([atom.coords for atom in self.atoms])
        all_coords -= self.CoM

        all_coords_T = all_coords.T
        new_coords = np.matmul(rotation_matrix, all_coords_T)
        all_coords = new_coords.T

        # apply offset
        all_coords += self.CoM

        for atom_ind, coords in enumerate(all_coords):
            self.atoms[atom_ind].coords = coords

    def get_CoM(self):
        mass_coord_sum = np.array([0, 0])
        for atom in self.atoms:
            coords = atom.coords
            type = atom.type
            mass = ATOM_TYPING[type]['mass']
            mass_coord_sum = np.sum(np.vstack((mass_coord_sum, (coords * mass))), axis=0)
        self.CoM = mass_coord_sum / self.total_mass

        return self.CoM

    def get_atom_CoM_vectors(self, skip_CoM_update=False):

        if not skip_CoM_update:
            # make sure CoM is up to date
            _ = self.get_CoM()

        all_atom_coords = np.asarray([atom.coords for atom in self.atoms])
        self.atom_CoM_vectors = all_atom_coords - self.CoM


    def get_rotation_for_translation(self):
        # find out what value of theta will deliver a gyration of 1 Angstrom
        # use law of cosines

        self.theta_per_ang = 2  * math.asin(1 / (2 * self.gyration_radius)) # in radians