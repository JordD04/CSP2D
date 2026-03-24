import random
from copy import deepcopy

import numpy as np

from chemistry import *

def get_random_coords():

    x = random.random()
    y = random.random()

    return x, y

class Atom:
    def __init__(self, coords, type, colourless=False):
        self.coords = coords
        typing = ATOM_TYPING[type]
        self.epsilon = typing['epsilon']
        self.sigma = typing['sigma']
        if colourless:
            self.colour = 'white'
        else:
            self.colour = typing['colour']
        self.radius = typing['radius']
        self.type = type

        self.graphic_coords = np.asarray([coords[0] - self.radius, coords[1] - self.radius,
                                          coords[0] + self.radius, coords[1] + self.radius])

    def update_coords(self, vector, lattice_x, lattice_y, lattice_offset):
        self.coords += vector

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


class UnitCell:
    def __init__(self, num_atoms, frame_center):
        # this doesn't do anything yet
        self.wpg = random.choice(['p1', 'p2', 'pmm', 'p4'])
        self.step_size = 30

        self.frame_center = frame_center
        self.lattice_x = (random.random() * 100) + 50
        self.lattice_y = (random.random() * 100) + 50
        self.calc_lattice_offset()

        self.num_atoms = num_atoms
        self.atoms = []

        for ind in range(num_atoms):
            self.create_atoms(type='LJ')

        self.update_images()


    def calc_lattice_offset(self):
        # this offsets puts the center of the unit cell in the center of the frame
        self.lattice_offset = self.frame_center - np.asarray([self.lattice_x / 2, self.lattice_y / 2])


    def create_atoms(self, type):
        frac_x, frac_y = get_random_coords()

        coords = np.asarray([frac_x * self.lattice_x, frac_y * self.lattice_y])
        coords += self.lattice_offset

        self.atoms.append(Atom(coords=coords, type=type))


    def update_images(self):
        latt_x = self.lattice_x
        latt_y = self.lattice_y
        translations = [np.array([-latt_x, latt_y]), np.array([0, latt_y]), np.array([latt_x, latt_y]),
                        np.array([-latt_x, 0]), np.array([latt_x, 0]),
                        np.array([-latt_x, -latt_y]), np.array([0, -latt_y]), np.array([latt_x, -latt_y])]


        self.images = []
        for atom in self.atoms:
            for translation in translations:
                coords = atom.coords + translation
                self.images.append(Atom(coords=coords, type=atom.type, colourless=True))

        self.image_coords = np.asarray([image.coords for image in self.images])

        self.aa_vectors = None


    def random_step(self):
        for atom in self.atoms:
            x = ((random.random() - 0.5) * 2) * self.lattice_x * 0.1 # generate number between -1 and 1 and scale
            y = ((random.random() - 0.5) * 2) * self.lattice_y * 0.1
            atom.update_coords(np.asarray([x, y]), self.lattice_x, self.lattice_y, self.lattice_offset)

        self.update_images()


    def get_atom_atom_vectors(self):
        self.aa_vectors = []
        for ind0, atom0 in enumerate(self.atoms):
            aa_vector_atom0 = []
            for ind1, atom1 in enumerate(self.atoms):
                if not ind0 == ind1:
                    r = calc_r(atom0.coords, atom1.coords)
                else:
                    r = 0
                aa_vector_atom0.append(r)

            pbc_rs = calc_r(atom0.coords, self.image_coords)
            aa_vector_atom0.append(pbc_rs)
            self.aa_vectors.append(aa_vector_atom0)


    def get_energy(self):
        if self.aa_vectors == None:
            self.get_atom_atom_vectors()

        total_energy = 0
        for ind0, atom0 in enumerate(self.atoms):
            for ind1, atom1 in enumerate(self.atoms):
                if not ind0 == ind1:
                    #r = calc_r(atom0.coords, atom1.coords)
                    r = self.aa_vectors[ind0][ind1]
                    energy = calc_lj_energy(r, atom0.epsilon, atom0.sigma)
                    total_energy += energy

            #pbc_rs = calc_r(atom0.coords, self.image_coords)
            pbc_rs = self.aa_vectors[ind0][-1]
            pbc_energies = calc_lj_energy(pbc_rs, atom0.epsilon, atom0.sigma)
            for pbc_energy in pbc_energies:
                total_energy += pbc_energy

        return total_energy


    def get_forces(self):
        if self.aa_vectors == None:
            self.get_atom_atom_vectors()

        forces = {"resultant_forces" : [],
                    "component_forces" : []}

        #pairwise_interactions = itertools.combinations(self.atoms, 2)
        for ind0, atom0 in enumerate(self.atoms):
            component_forces = []
            for ind1, atom1 in enumerate(self.atoms):
                if not ind0 == ind1:
                    #r = calc_r(atom0.coords, atom1.coords)
                    r = self.aa_vectors[ind0][ind1]
                    force = calc_lj_force(r, atom0.epsilon, atom0.sigma)
                    vector = calc_atom_atom_vector(atom0.coords, atom1.coords)
                    component_forces.append(calc_force_vector(force, vector))

            #pbc_rs = calc_r(atom0.coords, self.image_coords)
            pbc_rs = self.aa_vectors[ind0][-1]
            pbc_forces = calc_lj_force(pbc_rs, atom0.epsilon, atom0.sigma)
            pbc_vectors = calc_atom_atom_vector(atom0.coords, self.image_coords)
            pbc_force_vectors = calc_force_vector(pbc_forces, pbc_vectors)
            for force_vector in pbc_force_vectors:
                component_forces.append(force_vector)

            resultant_force = np.asarray([0.0, 0.0])
            for force in component_forces:
                resultant_force += force
            forces["resultant_forces"].append(resultant_force)
            forces["component_forces"].append(component_forces)

        forces["resultant_forces"] = np.asarray(forces["resultant_forces"])
        forces["component_forces"] = np.asarray(forces["component_forces"])

        return forces


    def get_stress(self):
        lattice_shift = 5
        if self.aa_vectors == None:
            self.get_atom_atom_vectors()

        stress_tensor = np.zeros((2,2))  # [ [xx, xy], [yx, yy] ]
        volume = self.lattice_x * self.lattice_y

        frac_coords = []
        for atom in self.atoms:
            coords = atom.coords
            atom_frac_coords = (coords - self.lattice_offset) / np.array([self.lattice_x, self.lattice_y])
            frac_coords.append(atom_frac_coords)

        for axis in ['x', 'y']:
            ext_cell = deepcopy(self)
            comp_cell = deepcopy(self)
            if axis == 'x':
                ext_cell.lattice_x = self.lattice_x + lattice_shift
                comp_cell.lattice_x = self.lattice_x - lattice_shift
            elif axis == 'y':
                ext_cell.lattice_y = self.lattice_y + lattice_shift
                comp_cell.lattice_y = self.lattice_y - lattice_shift

            ext_cell.calc_lattice_offset()
            comp_cell.calc_lattice_offset()

            for ind, atom_frac_coords in enumerate(frac_coords):
                ext_cell.atoms[ind].coords = ext_cell.lattice_offset + (atom_frac_coords * np.array([ext_cell.lattice_x, ext_cell.lattice_y]))
                ext_cell.update_images()

                comp_cell.atoms[ind].coords = comp_cell.lattice_offset + (atom_frac_coords * np.array([comp_cell.lattice_x, comp_cell.lattice_y]))
                comp_cell.update_images()

            energy_change = comp_cell.get_energy() - ext_cell.get_energy()
            if axis == 'x':
                strain = 2 / self.lattice_x
            elif axis == 'y':
                strain = 2 / self.lattice_y

            stress = ((lattice_shift * 2) / volume) * (energy_change / strain)
            if axis == 'x':
                stress_tensor[0][0] = stress
            elif axis == 'y':
                stress_tensor[1][1] = stress

        return stress_tensor

    #move every atom together
    def descend_gradient(self, variable_cell=True):
        if self.aa_vectors == None:
            self.get_atom_atom_vectors()
        #energy = self.get_energy()
        forces = self.get_forces()
        stress = self.get_stress()

        max_force = 0
        for force in forces["resultant_forces"]:
            max_force_local = max(force.min(), force.max(), key=abs)
            if max_force_local > max_force:
                max_force = max_force_local

        for ind, atom in enumerate(self.atoms):
            vector = self.step_size * forces["resultant_forces"][ind]
            vector = limit_vector_mag(vector)
            atom.update_coords(vector, self.lattice_x, self.lattice_y, self.lattice_offset)

        # update lattice parameters
        if variable_cell:
            frac_coords = []
            for atom in self.atoms:
                coords = atom.coords
                atom_frac_coords = (coords - self.lattice_offset) / np.array([self.lattice_x, self.lattice_y])
                frac_coords.append(atom_frac_coords)

            x_step = self.step_size * stress[0][0]
            y_step = self.step_size * stress[1][1]
            max_lattice_shift = 1.5
            if x_step > max_lattice_shift:
                x_step = max_lattice_shift
            elif x_step < -max_lattice_shift:
                x_step = -max_lattice_shift
            if y_step > max_lattice_shift:
                y_step = max_lattice_shift
            elif y_step < -max_lattice_shift:
                y_step = -max_lattice_shift
            self.lattice_x += x_step
            self.lattice_y += y_step
            self.calc_lattice_offset()
            for ind, atom_frac_coords in enumerate(frac_coords):
                self.atoms[ind].coords = self.lattice_offset + (atom_frac_coords * np.array([self.lattice_x, self.lattice_y]))

        self.update_images()

        self.step_size = self.step_size * 0.99