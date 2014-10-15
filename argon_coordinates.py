from __future__ import division
from math import sqrt, log
from random import random, choice


k = 0.00831

# Vector operation (don't have numpy ;()

def multiply_vector_by_number(number, vector):
    return [number * x for x in vector]


def add_three_vectors(a, b, c):
    return [x1 + x2 + x3 for x1, x2, x3 in zip(a, b, c)]


def add_two_vectors(a, b):
    return [x1 + x2 for x1, x2 in zip(a, b)]


def vector_difference(a, b):
    return [a[0] - b[0], a[1] - b[1], a[2] - b[2]]


def vector_length(vector):
    return sqrt(vector[0] ** 2 + vector[1] ** 2 + vector[2] ** 2)


def atoms_coordinates(n, b1, b2, b3):

    atoms = []
    for ix in range(n):
        for iy in range(n):
            for iz in range(n):
                atom = add_three_vectors(multiply_vector_by_number(ix - (n - 1.0)/2, b1),
                                         multiply_vector_by_number(iy - (n - 1.0)/2, b2),
                                         multiply_vector_by_number(iz - (n - 1.0)/2, b3))
                atoms.append(atom)
    return atoms


def save_to_file(coordinates, filename):
    with open(filename, 'w+') as f:
        for x, y, z in coordinates:
            f.write('%s\t%s\t%s\n' % (x, y, z))


def read_parameters(filename):
    with open(filename) as f:
        data = f.read()

    numbers = data.split()
    if len(numbers) != 4:
        print('Wrong %s data' % filename)
        return None
    else:
        a = float(numbers[0])
        n = int(numbers[1])
        m = float(numbers[2])
        To = float(numbers[3])
    return a, n, m, To


def start_momentum(atoms, m, To):
    max_energy = - k / 2 * To
    energies = []
    for atom in atoms:
        energies.append((max_energy * log(random()), max_energy * log(random()), max_energy * log(random())))

    # normalization
    average_energy = sum([element for tupl in energies for element in tupl]) / len(energies) / 3

    norm_factor = - max_energy / average_energy
    energies = [multiply_vector_by_number(norm_factor, energy) for energy in energies]

    momentums = []
    
    def compute_momentum(x):
        return choice([-1, 1]) * sqrt(2 * m * x)

    for energy in energies:
        momentum = (compute_momentum(energy[0]), compute_momentum(energy[1]), compute_momentum(energy[2]))
        momentums.append(momentum)
    return momentums


def compute_Fij(atom1, atom2, R, epsilon):
    distance = vector_difference(atom1, atom2)
    rij = vector_length(distance)
    Fij = multiply_vector_by_number(12 * epsilon * ( (R / rij) ** 12 - (R / rij) ** 6) / (rij ** 2), distance)
    return Fij


def compute_Vij(atom1, atom2, R, epsilon):
    distance = vector_difference(atom1, atom2)
    rij = vector_length(distance)
    Vij = epsilon * ( (R / rij) ** 12 - 2 * (R / rij) ** 6) 
    return Vij


def compute_forces_and_potential(atoms, R, f, L, epsilon):
    N = len(atoms)
    Fis = [(0, 0, 0)] * N
    V = 0
    for i in range(N):
        for j in range(i + 1, N):
            # print(i, j)
            Fij = compute_Fij(atoms[i], atoms[j], R, epsilon)
            Vij = compute_Vij(atoms[i], atoms[j], R, epsilon)
            Fis[i] = add_two_vectors(Fis[i], Fij)
            Fis[j] = add_two_vectors(Fis[j], multiply_vector_by_number(-1, Fij))
            V += Vij

    for i, atom in enumerate(atoms):
        ri = vector_length(atom)
        if ri > L:
            fs = multiply_vector_by_number(f * (L - ri) / ri, atom)
            Fis[i] = add_two_vectors(Fis[i], fs)
            Vi = 1 / 2 * f *(L - ri) ** 2
            V += Vi

    return Fis, V
    

def step(atoms, momentums, forces, tau, m, R, f, L, epsilon, number_of_steps=10):
    N = len(atoms)
    new_atoms = [(0, 0, 0)] * N
    momentum_half = [(0, 0, 0)] * N
    new_momentums = [(0, 0, 0)] * N

    forces, _ = compute_forces_and_potential(atoms, R, f, L, epsilon)
    
    for j in range(number_of_steps):
        for i in range(N):
            momentum_half[i] = add_two_vectors(momentums[i], multiply_vector_by_number(tau / 2, forces[i]))
            atoms[i] = add_two_vectors(atoms[i], multiply_vector_by_number(tau / m, momentum_half[i]))

        forces, _ = compute_forces_and_potential(atoms, R, f, L, epsilon)
        for i in range(N):
            momentums[i] = add_two_vectors(momentum_half[i], multiply_vector_by_number(tau / 2, forces[i]))
        print(atoms[0])

    return atoms, momentums 


def main():
    assert(multiply_vector_by_number(3, (2, 3, 0)) == [6, 9, 0])
    assert(add_three_vectors((1, 2, 3), (3, 4, 5), (3, 3, 3)) == [7, 9, 11])
    
    parameters = read_parameters('argon.input')

    if not parameters:
        return 1
    else:
        a, n, m, To = parameters
 
    # argon structure
    b1 = (a, 0, 0)
    b2 = (a/2, a * sqrt(3)/ 2, 0)
    b3 = (a/2, a * sqrt(3)/ 6, a * sqrt(2/3))

    # box structure
    p1 = (a, 0, 0)
    p2 = (0, a, 0)
    p3 = (0, 0, a)

    atoms = atoms_coordinates(n, b1, b2, b3)
    save_to_file(atoms, 'aus.dat')

    # atoms = atoms_coordinates(n, p1, p2, p3)
    # save_to_file(atoms, 'simple.dat')
    
    momentums = start_momentum(atoms, m, To)
    save_to_file(momentums, 'momentums.dat')

    R = a
    f = 10000
    L = 2.3
    epsilon = 1

    forces, V = compute_forces_and_potential(atoms, R, f, L, epsilon)
    print(V)
    # print(forces)
    
    tau = 0.01
    atoms, momentums = step(atoms, momentums, forces, tau, m, R, f, L, epsilon, number_of_steps=10)
    print(atoms)

if __name__ == '__main__':
    main()
