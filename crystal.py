import random
from copy import deepcopy
import math
from chemistry import *

def get_random_coords():

    x = random.random()
    y = random.random()

    return x, y

def get_random_rotation():

    return random.random() * math.pi * 2


class UnitCell:
    def __init__(self, frame_center):
        # this doesn't do anything yet
        self.wpg = random.choice(['p1', 'p2', 'pmm', 'p4'])
        self.step_size = 30

        self.frame_center = frame_center
        self.lattice_x = ((random.random() * 3) + 1.5) * ANG2PIX
        self.lattice_y = ((random.random() * 3) + 1.5) * ANG2PIX
        self.calc_lattice_offset()

        self.num_atoms = 0
        self.atoms = []
        self.molecules = []
        self.all_mol_atoms = []

        # for ind in range(5):
        #     self.create_atoms(type='LJ')

        for ind in range(3):
            frac_x, frac_y = get_random_coords()
            coords = np.asarray([frac_x * self.lattice_x, frac_y * self.lattice_y])
            coords += self.lattice_offset
            x, y = coords
            theta = get_random_rotation()

            # create water Molecule
            molecule = Molecule.init_water(x, y, theta)
            mol_atoms = []
            for atom in molecule.atoms:
                self.atoms.append(atom)
                mol_atoms.append(deepcopy(self.num_atoms))
                self.num_atoms += 1

            self.molecules.append(molecule)
            self.all_mol_atoms.append(mol_atoms)

        self.update_images()

        # calculate atom-atom epsilon and sigma values
        self.epsilon_vector = []
        self.sigma_vector = []
        self.qprod_vector = []
        for ind0, atom0 in enumerate(self.atoms):
            atom0_atomN_epsilon = []
            atom0_atomN_sigma = []
            atom0_atomN_qprod = []
            for ind1, atom1 in enumerate(self.atoms):
                epsilon = (atom0.epsilon + atom1.epsilon) / 2
                sigma = (atom0.sigma + atom1.sigma) / 2
                qprod = atom0.q * atom1.q
                atom0_atomN_epsilon.append(epsilon)
                atom0_atomN_sigma.append(sigma)
                atom0_atomN_qprod.append(qprod)

            atom0_pbc_epsilon = (np.asarray([image.epsilon for image in self.images]) + atom0.epsilon) / 2
            atom0_pbc_sigma = (np.asarray([image.sigma for image in self.images]) + atom0.sigma) / 2
            atom0_pbc_qprod = (np.asarray([image.q for image in self.images]) * atom0.q)

            atom0_atomN_epsilon.append(atom0_pbc_epsilon)
            atom0_atomN_sigma.append(atom0_pbc_sigma)
            atom0_atomN_qprod.append(atom0_pbc_qprod)

            self.epsilon_vector.append(atom0_atomN_epsilon)
            self.sigma_vector.append(atom0_atomN_sigma)
            self.qprod_vector.append(atom0_atomN_qprod)


        # make sure we have no interactions within molecules
        if len(self.molecules) > 0:
            for mol_atoms in self.all_mol_atoms:
                for atom0_ind in mol_atoms:
                    for atom1_ind in mol_atoms:
                        self.epsilon_vector[atom0_ind][atom1_ind] = 0
                        self.sigma_vector[atom0_ind][atom1_ind] = 1


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
                    r = self.aa_vectors[ind0][ind1]
                    #energy = calc_lj_energy(r, self.epsilon_vector[ind0][ind1], self.sigma_vector[ind0][ind1])
                    energy = calc_total_energy(r, self.qprod_vector[ind0][ind1], self.epsilon_vector[ind0][ind1], self.sigma_vector[ind0][ind1])
                    total_energy += energy

            pbc_rs = self.aa_vectors[ind0][-1]
            #pbc_energies = calc_lj_energy(pbc_rs, self.epsilon_vector[ind0][-1], self.sigma_vector[ind0][-1])
            pbc_energies = calc_total_energy(pbc_rs, self.qprod_vector[ind0][-1], self.epsilon_vector[ind0][-1], self.sigma_vector[ind0][-1])
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
                    r = self.aa_vectors[ind0][ind1]
                    #force = calc_lj_force(r, self.epsilon_vector[ind0][ind1], self.sigma_vector[ind0][ind1])
                    force = calc_total_force(r, self.qprod_vector[ind0][ind1], self.epsilon_vector[ind0][ind1], self.sigma_vector[ind0][ind1])
                    vector = calc_atom_atom_vector(atom0.coords, atom1.coords)
                    component_forces.append(calc_force_vector(force, vector))

            pbc_rs = calc_r(atom0.coords, self.image_coords)
            pbc_rs = self.aa_vectors[ind0][-1]
            #pbc_forces = calc_lj_force(pbc_rs, self.epsilon_vector[ind0][-1], self.sigma_vector[ind0][-1])
            pbc_forces = calc_total_force(pbc_rs, self.qprod_vector[ind0][-1], self.epsilon_vector[ind0][-1], self.sigma_vector[ind0][-1])
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
        if len(self.molecules) == 0:
            for atom in self.atoms:
                coords = atom.coords
                atom_frac_coords = (coords - self.lattice_offset) / np.array([self.lattice_x, self.lattice_y])
                frac_coords.append(atom_frac_coords)
        else:
            for molecule in self.molecules:
                mol_CoM = molecule.get_CoM()
                mol_frac_coords = (mol_CoM - self.lattice_offset) / np.array([self.lattice_x, self.lattice_y])
                frac_coords.append(mol_frac_coords)

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

            if len(self.molecules) == 0:
                for ind, atom_frac_coords in enumerate(frac_coords):
                    ext_cell.atoms[ind].coords = ext_cell.lattice_offset + (atom_frac_coords * np.array([ext_cell.lattice_x, ext_cell.lattice_y]))
                    ext_cell.update_images()

                    comp_cell.atoms[ind].coords = comp_cell.lattice_offset + (atom_frac_coords * np.array([comp_cell.lattice_x, comp_cell.lattice_y]))
                    comp_cell.update_images()
            else:
                for mol_ind, molecule in enumerate(ext_cell.molecules):
                    mol_frac_coords = frac_coords[mol_ind]
                    new_mol_CoM = ext_cell.lattice_offset + (mol_frac_coords * np.array([ext_cell.lattice_x, ext_cell.lattice_y]))
                    CoM_vector = new_mol_CoM - molecule.CoM
                    molecule.translate_molecule(CoM_vector, ext_cell.lattice_x, ext_cell.lattice_y, ext_cell.lattice_offset)
                    ext_cell.update_images()

                for mol_ind, molecule in enumerate(comp_cell.molecules):
                    mol_frac_coords = frac_coords[mol_ind]
                    new_mol_CoM = comp_cell.lattice_offset + (mol_frac_coords * np.array([comp_cell.lattice_x, comp_cell.lattice_y]))
                    CoM_vector = new_mol_CoM - molecule.CoM
                    molecule.translate_molecule(CoM_vector, comp_cell.lattice_x, comp_cell.lattice_y, comp_cell.lattice_offset)
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

        # max_force = 0
        # for force in forces["resultant_forces"]:
        #     max_force_local = max(force.min(), force.max(), key=abs)
        #     if max_force_local > max_force:
        #         max_force = max_force_local

        if len(self.molecules) == 0:
            for ind, atom in enumerate(self.atoms):
                vector = self.step_size * forces["resultant_forces"][ind]
                vector = limit_vector_mag(vector)
                atom.update_coords(vector, self.lattice_x, self.lattice_y, self.lattice_offset)
        else:
            for mol_ind, molecule in enumerate(self.molecules):
                molecular_forces = []
                for atom_ind in self.all_mol_atoms[mol_ind]:
                    molecular_forces.append(forces["resultant_forces"][atom_ind])
                net_force = np.sum(np.asarray(molecular_forces), axis=0)
                vector = net_force * self.step_size
                vector = limit_vector_mag(vector)
                molecule.translate_molecule(vector, self.lattice_x, self.lattice_y, self.lattice_offset)

                # do torques now
                molecule.get_atom_CoM_vectors()
                torque = 0
                for atom_ind, atom in enumerate(molecule.atoms):
                    atom_CoM_vector = molecule.atom_CoM_vectors[atom_ind]
                    atom_force = molecular_forces[atom_ind]

                    torque_comp = np.cross(atom_CoM_vector, atom_force)
                    torque += torque_comp

                rotation = torque * self.step_size
                if rotation > (molecule.theta_per_ang * 3):
                    rotation = molecule.theta_per_ang * 3
                elif rotation < (molecule.theta_per_ang * -3):
                    rotation = molecule.theta_per_ang * -3

                rot_mat = get_rotation_matrix(rotation)
                molecule.rotate_molecule(rot_mat)

        # update lattice parameters
        if variable_cell:
            frac_coords = []
            if len(self.molecules) == 0:
                for atom in self.atoms:
                    coords = atom.coords
                    atom_frac_coords = (coords - self.lattice_offset) / np.array([self.lattice_x, self.lattice_y])
                    frac_coords.append(atom_frac_coords)
            else:
                frac_coords = []
                for molecule in self.molecules:
                    mol_CoM = molecule.get_CoM()
                    mol_frac_coords = (mol_CoM - self.lattice_offset) / np.array([self.lattice_x, self.lattice_y])
                    frac_coords.append(mol_frac_coords)

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

            if len(self.molecules) == 0:
                for ind, atom_frac_coords in enumerate(frac_coords):
                    self.atoms[ind].coords = (
                            self.lattice_offset + (atom_frac_coords * np.array([self.lattice_x, self.lattice_y])))
            else:
                for mol_ind, molecule in enumerate(self.molecules):
                    mol_frac_coords = frac_coords[mol_ind]
                    new_mol_CoM = self.lattice_offset + (mol_frac_coords * np.array([self.lattice_x, self.lattice_y]))
                    CoM_vector = new_mol_CoM - molecule.CoM
                    molecule.translate_molecule(CoM_vector, self.lattice_x, self.lattice_y, self.lattice_offset)


        self.update_images()

        self.step_size = self.step_size * 0.99