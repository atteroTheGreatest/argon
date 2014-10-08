from __future__ import division
from math import sqrt

a = 0.23  # nm
b1 = (a, 0, 0)
b2 = (a/2, a * sqrt(3)/ 2, 0)
b3 = (a/2, a * sqrt(3)/ 6, a * sqrt(2/3))


def multiply_vector_by_number(number, vector):
    return [number * x for x in vector]

def add_vectors(a, b, c):
    return [x1 + x2 + x3 for x1, x2, x3 in zip(a, b, c)]

assert(multiply_vector_by_number(3, (2, 3, 0)) == [6, 9, 0])
assert(add_vectors((1, 2, 3), (3, 4, 5), (3, 3, 3)) == [7, 9, 11])


def atoms_coordinates(n, b1, b2, b3):

    atoms = []
    for ix in range(n):
        for iy in range(n):
            for iz in range(n):
                atom = add_vectors(multiply_vector_by_number(ix - (n - 1.0)/2, b1),
                                   multiply_vector_by_number(iy - (n - 1.0)/2, b2),
                                   multiply_vector_by_number(iz - (n - 1.0)/2, b3))
                atoms.append(atom)
    return atoms


def save_to_file(atoms, filename):
    with open(filename, 'w+') as f:
        for atom in atoms:
            f.write('%s\t%s\t%s\n' % (atom[0], atom[1], atom[2]))

p1 = (a, 0, 0)
p2 = (0, a, 0)
p3 = (0, 0, a)

atoms = atoms_coordinates(5, b1, b2, b3)
save_to_file(atoms, 'aus.dat')

atoms = atoms_coordinates(5, p1, p2, p3)
save_to_file(atoms, 'simple.dat')
