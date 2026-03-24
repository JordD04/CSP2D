import numpy as np

# 30 pixels = 1 angstroms

ATOM_TYPING = \
    {'LJ': {'epsilon' : 1,
            'sigma' : 30,
            'colour' : 'crimson',
            'radius' : 15}
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
    #print(r)
    coefficient = 4 * epsilon
    repulsive = (sigma ** 12) * (r ** (-12))
    attractive = (sigma ** 6) * (r ** (-6))

    return 0.5 * coefficient * (repulsive - attractive)

def limit_vector_mag(vector, max_step_size=3):
    mag = np.linalg.norm(vector)
    if mag > max_step_size:
        vector = vector * (max_step_size / mag)

    return vector