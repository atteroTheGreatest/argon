from __future__ import division
from math import sqrt, log
from random import random, choice
import numpy as np


k = 0.00831


def atoms_coordinates(n, b1, b2, b3):

    atoms = []
    for ix in range(n):
        for iy in range(n):
            for iz in range(n):
                atom = (ix - (n - 1.0) / 2) * b1 + (iy - (n - 1.0)/2) * b2 + (iz - (n - 1.0) / 2) * b3
                atoms.append(atom)
    return atoms


def save_to_file(coordinates, filename):
    with open(filename, 'w+') as f:
        for x in coordinates:
            f.write('%s\t%s\t%s\n' % (x[0], x[1], x[2]))


def append_to_file(coordinates, filename):
    with open(filename, 'a+') as f:
        f.write('\n\n')
        for x in coordinates:
            f.write('%s\t%s\t%s\n' % (x[0], x[1], x[2]))


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
        energies.append(np.array((max_energy * log(random()),
                                  max_energy * log(random()),
                                  max_energy * log(random()))))

    # normalization
    average_energy = sum([element for tupl in energies for element in tupl]) / len(energies) / 3

    norm_factor = - max_energy / average_energy
    energies = [norm_factor * energy for energy in energies]

    momentums = []

    def compute_momentum(x):
        return choice([-1, 1]) * sqrt(2 * m * x)

    for energy in energies:
        momentum = (compute_momentum(energy[0]), compute_momentum(energy[1]), compute_momentum(energy[2]))
        momentums.append(momentum)
    return momentums


def compute_Fij(atom1, atom2, R, epsilon):
    distance = atom1 - atom2
    rij = np.linalg.norm(distance)
    if abs(rij) < 0.2:
        rij = 0.2

    Fij = - 12 * epsilon * ( (R / rij) ** 12 - (R / rij) ** 6) / (rij ** 2) * distance

    return Fij


def compute_Vij(atom1, atom2, R, epsilon):
    distance = atom1 - atom2
    rij = np.linalg.norm(distance)
    if abs(rij) < 0.0001:
        rij = 0.0001

    Vij = epsilon * ( (R / rij) ** 12 - 2 * (R / rij) ** 6)
    return Vij


def compute_forces_and_potential(atoms, R, f, L, epsilon):
    N = len(atoms)
    Fis = [np.array([0, 0, 0])] * N
    V = 0
    for i in range(N):
        for j in range(i + 1, N):
            Fij = compute_Fij(atoms[i], atoms[j], R, epsilon)
            Vij = compute_Vij(atoms[i], atoms[j], R, epsilon)
            Fis[i] = Fis[i] +  Fij
            Fis[j] = Fis[j] - Fij
            V += Vij

    for i, atom in enumerate(atoms):
        ri = np.linalg.norm(atom)
        if ri > L:
            fs = f * (L - ri) / ri * atom
            Fis[i] = Fis[i] + fs
            Vi = 1 / 2 * f *(L - ri) ** 2

            V += Vi
    return Fis, V


def step(atoms, momentums, forces, tau, m, R, f, L, epsilon, number_of_steps=10):
    N = len(atoms)
    new_atoms = [np.array((0, 0, 0))] * N
    momentum_half = [np.array((0, 0, 0))] * N
    new_momentums = [np.array((0,0, 0))] * N

    forces, _ = compute_forces_and_potential(atoms, R, f, L, epsilon)

    save_to_file(atoms, 'avs.dat')

    for j in range(number_of_steps):
        for i in range(N):
            momentum_half[i] = momentums[i] + tau / 2 * forces[i]
            atoms[i] = atoms[i] + tau / m * momentum_half[i]

        forces, _ = compute_forces_and_potential(atoms, R, f, L, epsilon)
        for i in range(N):
            momentums[i] = momentum_half[i] + tau / 2 * forces[i]
        if j % 10 == 0:
            append_to_file(atoms, 'avs.dat')

    return atoms, momentums


def main():
    parameters = read_parameters('argon.input')

    if not parameters:
        return 1
    else:
        a, n, m, To = parameters

    # argon structure
    b1 = np.array((a, 0, 0))
    b2 = np.array((a / 2, a * sqrt(3) / 2, 0))
    b3 = np.array((a / 2, a * sqrt(3) / 6, a * sqrt(2 / 3)))

    atoms = atoms_coordinates(n, b1, b2, b3)
    save_to_file(atoms, 'aus.dat')

    momentums = start_momentum(atoms, m, To)
    save_to_file(momentums, 'momentums.dat')

    R = a
    f = 100000
    L = 3
    epsilon = 1

    forces, V = compute_forces_and_potential(atoms, R, f, L, epsilon)
    print(V)

    tau = 0.003
    atoms, momentums = step(atoms, momentums, forces, tau, m, R, f, L, epsilon, number_of_steps=500)
    print(atoms)

if __name__ == '__main__':
    main()
