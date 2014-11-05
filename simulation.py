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
    if rij < 0.38:
        rij = 0.38

    Fij = - 12 * epsilon * ((R / rij) ** 12 - (R / rij) ** 6) / (rij ** 2) * distance

    return Fij


def compute_Vij(atom1, atom2, R, epsilon):
    distance = atom1 - atom2
    rij = np.linalg.norm(distance)
    if rij < 0.0001:
        rij = 0.0001

    Vij = epsilon * ( (R / rij) ** 12 - 2 * (R / rij) ** 6)
    return Vij


class Simulation(object):

    def read_parameters(self, filename):
        with open(filename) as f:
            data = f.read()

        numbers = data.split()
        if len(numbers) != 4:
            print('Wrong %s data' % filename)
            return None
        else:
            self.a = float(numbers[0])
            self.n = int(numbers[1])
            self.m = float(numbers[2])
            self.To = float(numbers[3])
            self.tau = 0.005
            self.m = 10
            self.R = self.a
            self.f = 10000
            self.L = 2.3
            self.N = self.n ** 3
            self.epsilon = 1
            self.V = 0
            self.P = 0
            self.output_file_created = False

    def __init__(self, configuration_file, output_filename=None, output_prefix=None):
        self.read_parameters(configuration_file)
        a = self.a
        b1 = np.array((a, 0, 0))
        b2 = np.array((a / 2, a * sqrt(3) / 2, 0))
        b3 = np.array((a / 2, a * sqrt(3) / 6, a * sqrt(2 / 3)))
        self.atoms = atoms_coordinates(self.n, b1, b2, b3)
        self.momentums = start_momentum(self.atoms, self.m, self.To)
        self.forces, _ = self.compute_forces_and_potential()

        if output_filename is None:
            if output_prefix:
                self.output_filename = self.generate_from_prefix(output_prefix)
            else:
                self.output_filename = 'out.dat'
        else:
            self.output_filename = output_filename

    def generate_from_prefix(self, output_prefix):
        return "%s-%s-%s-%s-%s.dat" % (output_prefix, self.a, self.tau, self.To, self.n)

    def kinetic_energy(self, momentum):
        return np.linalg.norm(momentum) ** 2 / (2 * self.m)

    def compute_forces_and_potential(self):

        Fis = [np.array([0, 0, 0])] * self.N
        V = 0
        for i in range(self.N):
            for j in range(i + 1, self.N):
                if i == j:
                    print('ups!', j)
                Fij = compute_Fij(self.atoms[i], self.atoms[j],
                                  self.R, self.epsilon)
                Fis[i] = Fis[i] +  Fij
                Fis[j] = Fis[j] - Fij

                Vij = compute_Vij(self.atoms[i], self.atoms[j],
                                  self.R, self.epsilon)
                V += Vij
        P = 0
        for i, atom in enumerate(self.atoms):
            ri = np.linalg.norm(atom)
            if ri > self.L:
                fs = self.f * (self.L - ri) / ri * atom
                P += np.linalg.norm(fs)
                Fis[i] = Fis[i] + fs
                Vi = 1 / 2 * self.f * (self.L - ri) ** 2
                V += Vi
        self.P = P / 4 / np.pi / self.L ** 2
        self.V = V
        return Fis, V

    def compute_temperature(self):
        return sum(map(self.kinetic_energy, self.momentums)) * 2 / (3 * k * self.N)

    def compute_H(self):
        return sum(map(self.kinetic_energy, self.momentums)) + self.V

    def step(self):
        momentum_half = [np.array((0, 0, 0))] * self.N
        for i in range(self.N):
            momentum_half[i] = self.momentums[i] + self.tau / 2 * self.forces[i]
            self.atoms[i] = self.atoms[i] + self.tau / self.m * momentum_half[i]

        self.forces, _ = self.compute_forces_and_potential()
        for i in range(self.N):
            self.momentums[i] = momentum_half[i] + self.tau / 2 * self.forces[i]

    def run(self, s_o=100, s_d=2000, s_out=1, s_xyz=1):

        self.forces, _ = self.compute_forces_and_potential()
        for j in range(s_o):
            self.step()

        self.store_coordinates()
        temperature_sum_in_period = 0
        for j in range(1, s_d):
            self.step()
            temperature = self.compute_temperature()
            print(temperature)
            H = self.compute_H()
            temperature_sum_in_period += temperature
            if j % s_xyz == 0:
                self.store_coordinates()
            if j % s_out == 0:
                self.store_results()
            if j % s_o == 0:
                print('temperature', temperature_sum_in_period / s_o)
                print('pot', self.V)
                print('H', H)
                print('P', self.P)
                temperature_sum_in_period = 0

        return self.atoms, self.momentums

    def store_results(self):
        if not self.output_file_created:
            save_to_file(self.atoms, self.output_filename)
            self.output_file_created = True
        else:
            append_to_file(self.atoms, self.output_filename)

    def store_coordinates(self):
        if not self.output_file_created:
            save_to_file(self.atoms, self.output_filename)
            self.output_file_created = True
        else:
            append_to_file(self.atoms, self.output_filename)


def main():
    import sys
    configuration_file = sys.argv[1]
    output_file = sys.argv[2]

    simulation = Simulation(configuration_file, output_filename=output_file)
    simulation.run(s_o=10, s_d=1000)


if __name__ == '__main__':
    main()