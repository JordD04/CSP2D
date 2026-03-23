import random
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
        corr_vector = vector + wrapping_vector

        # generate vector for graphics
        graphics_vector = np.asarray([corr_vector[0], corr_vector[1],
                                      corr_vector[0], corr_vector[1]])
        self.graphic_coords += graphics_vector


class UnitCell:
    def __init__(self, num_atoms, frame_center):
        # this doesn't do anything yet
        self.wpg = random.choice(['p1', 'p2', 'pmm', 'p4'])

        self.lattice_x = (random.random() * 100) + 50
        self.lattice_y = (random.random() * 100) + 50
        # this offsets puts the center of the unit cell in the center of the frame
        self.lattice_offset = frame_center - np.asarray([self.lattice_x/2, self.lattice_y/2])

        self.num_atoms = num_atoms
        self.atoms = []

        for ind in range(num_atoms):
            self.create_atoms(type='LJ')

        self.update_images()


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


    def random_step(self):
        for atom in self.atoms:
            x = ((random.random() - 0.5) * 2) * self.lattice_x * 0.1 # generate number between -1 and 1 and scale
            y = ((random.random() - 0.5) * 2) * self.lattice_y * 0.1
            atom.update_coords(np.asarray([x, y]), self.lattice_x, self.lattice_y, self.lattice_offset)

        self.update_images()

    def get_forces(self):
        forces = {"resultant_forces" : [],
                    "component_forces" : []}

        #pairwise_interactions = itertools.combinations(self.atoms, 2)
        for ind0, atom0 in enumerate(self.atoms):
            component_forces = []
            for ind1, atom1 in enumerate(self.atoms):
                if not ind0 == ind1:
                    r = calc_r(atom0.coords, atom1.coords)
                    force = calc_lj_force(r, atom0.epsilon, atom0.sigma)
                    vector = calc_atom_atom_vector(atom0.coords, atom1.coords)
                    component_forces.append(calc_force_vector(force, vector))

            pbc_rs = calc_r(atom0.coords, self.image_coords)
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

    #move every atom together
    def descend_gradient(self, step_size=30):
        forces = self.get_forces()

        max_force = 0
        for force in forces["resultant_forces"]:
            max_force_local = max(force.min(), force.max(), key=abs)
            if max_force_local > max_force:
                max_force = max_force_local

        for ind, atom in enumerate(self.atoms):
            vector = step_size * forces["resultant_forces"][ind]
            vector = limit_vector_mag(vector)
            atom.update_coords(vector, self.lattice_x, self.lattice_y, self.lattice_offset)

        self.update_images()