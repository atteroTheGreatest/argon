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


def save_to_file(results, filename):
    with open(filename, 'w+') as f:
        template = "{}\t" * len(results[0]) + "\n"
        for x in results:
            f.write(template.format(*list(x)))


def append_to_file(results, filename, blank_lines=False):
    with open(filename, 'a+') as f:
        if blank_lines:
            f.write('\n\n')
        template = "{}\t" * len(results[0]) + "\n"
        for x in results:
            f.write(template.format(*list(x)))


def start_momentum(atoms, m, To):
    if To == 0:
        return [np.array((0, 0, 0))] * len(atoms)

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


class Configuration(object):

    def __init__(self, filename):
        with open(filename) as f:
            data = f.read()

        numbers = [float(row.split()[0]) for row in data.split('\n')[:13]]

        self.n = int(numbers[0])
        self.m = numbers[1]
        self.epsilon = numbers[2]
        self.R = numbers[3]

        self.f = numbers[4]
        self.L = numbers[5]
        self.a = numbers[6]

        self.To = numbers[7]
        self.tau = numbers[8]
        self.s_o = int(numbers[9])
        self.s_d = int(numbers[10])
        self.s_out = int(numbers[11])
        self.s_xyz = int(numbers[12])

    def __str__(self):
        return "To_%s_tau_%s_n_%s" % (self.To, self.tau, self.n)


class Reporter(object):

    def __init__(self, filename=None, blank_lines=False):

        self.output_filename = filename
        self.output_file_created = False
        self.results_output_file_created = False
        self.blank_lines = blank_lines

    def store(self, results):
        if not self.output_file_created:
            save_to_file(results, self.output_filename)
            self.output_file_created = True
        else:
            append_to_file(results, self.output_filename, self.blank_lines)


class Simulation(object):

    def __init__(self, configuration_file, output_filename=None):
        self.conf = Configuration(configuration_file)

        self.N = self.conf.n ** 3
        self.V = 0
        self.P = 0
        self.temperature = self.conf.To
        self.H = 0

        # prefix last name of output_filename with configuration
        coordinate_output = self.prepare_output_path(output_filename, 'coords')
        self.coordinate_reporter = Reporter(coordinate_output, blank_lines=True)

        potential_output = self.prepare_output_path(output_filename, 'potential')
        potential_reporter = Reporter(potential_output)

        temperature_output = self.prepare_output_path(output_filename, 'temperature')
        temperature_reporter = Reporter(temperature_output)

        hamiltonian_output = self.prepare_output_path(output_filename, 'H')
        hamiltonian_reporter = Reporter(hamiltonian_output)

        pressure_output = self.prepare_output_path(output_filename, 'P')
        pressure_reporter = Reporter(pressure_output)

        self.reporters_to_params = (
            (potential_reporter, lambda: self.V),
            (temperature_reporter, lambda: self.temperature),
            (hamiltonian_reporter, lambda: self.H),
            (pressure_reporter, lambda: self.P),
        )

        a = self.conf.a
        b1 = np.array((a, 0, 0))
        b2 = np.array((a / 2, a * sqrt(3) / 2, 0))
        b3 = np.array((a / 2, a * sqrt(3) / 6, a * sqrt(2 / 3)))

        self.atoms = atoms_coordinates(self.conf.n, b1, b2, b3)
        self.momentums = start_momentum(self.atoms, self.conf.m, self.conf.To)
        self.forces, _ = self.compute_forces_and_potential()

    def prepare_output_path(self, output_filename, param_name):
        dirs = "/".join(output_filename.split('/')[:-1])
        filename = output_filename.split('/')[-1]
        return dirs + '/' + param_name + '_' + str(self.conf) + filename

    def kinetic_energy(self, momentum):
        return np.linalg.norm(momentum) ** 2 / (2 * self.conf.m)

    def compute_forces_and_potential(self):

        Fis = [np.array([0, 0, 0])] * self.N
        V = 0
        for i in range(self.N):
            for j in range(i + 1, self.N):
                Fij = compute_Fij(self.atoms[i], self.atoms[j],
                                  self.conf.R, self.conf.epsilon)
                Fis[i] = Fis[i] +  Fij
                Fis[j] = Fis[j] - Fij

                Vij = compute_Vij(self.atoms[i], self.atoms[j],
                                  self.conf.R, self.conf.epsilon)
                V += Vij
        P = 0

        L = self.conf.L
        for i, atom in enumerate(self.atoms):
            ri = np.linalg.norm(atom)
            if ri > L:
                fs = self.conf.f * (L - ri) / ri * atom
                P += np.linalg.norm(fs)
                Fis[i] = Fis[i] + fs
                Vi = 1 / 2 * self.conf.f * (L - ri) ** 2
                V += Vi
        self.P = P / 4 / np.pi / L ** 2
        self.V = V
        return Fis, V

    def compute_temperature(self):
        return sum(map(self.kinetic_energy, self.momentums)) * 2 / (3 * k * self.N)

    def compute_H(self):
        return sum(map(self.kinetic_energy, self.momentums)) + self.V

    def step(self):
        momentum_half = [np.array((0, 0, 0))] * self.N
        for i in range(self.N):
            momentum_half[i] = self.momentums[i] + self.conf.tau / 2 * self.forces[i]
            self.atoms[i] = self.atoms[i] + self.conf.tau / self.conf.m * momentum_half[i]

        self.forces, _ = self.compute_forces_and_potential()
        for i in range(self.N):
            self.momentums[i] = momentum_half[i] + self.conf.tau / 2 * self.forces[i]

    def run(self, s_o=None, s_d=None, s_out=None, s_xyz=None):
        if s_o is None:
            s_o = self.conf.s_o
        if s_d is None:
            s_d = self.conf.s_d
        if s_out is None:
            s_out = self.conf.s_out
        if s_xyz is None:
            s_xyz = self.conf.s_xyz

        a = self.conf.a
        b1 = np.array((a, 0, 0))
        b2 = np.array((a / 2, a * sqrt(3) / 2, 0))
        b3 = np.array((a / 2, a * sqrt(3) / 6, a * sqrt(2 / 3)))

        self.atoms = atoms_coordinates(self.conf.n, b1, b2, b3)
        self.momentums = start_momentum(self.atoms, self.conf.m, self.conf.To)
        self.forces, _ = self.compute_forces_and_potential()

        for j in range(s_o):
            self.step()
        print('potential', self.V)

        self.coordinate_reporter.store(self.atoms)
        temperature_sum_in_period = 0
        for j in range(1, s_d):
            self.step()
            temperature = self.compute_temperature()
            print(temperature)
            self.H = self.compute_H()
            temperature_sum_in_period += temperature
            if j % s_xyz == 0:
                self.coordinate_reporter.store(self.atoms)
            if j % s_out == 0:

                self.coordinate_reporter.store(self.atoms)
            if j % s_o == 0:
                print('temperature', temperature_sum_in_period / s_o)
                self.temperature = temperature_sum_in_period / s_o
                print('pot', self.V)
                print('H', self.H)
                print('P', self.P)
                for reporter, param in self.reporters_to_params:
                    reporter.store([[j * self.conf.tau, param()]])

                temperature_sum_in_period = 0

        return self.atoms, self.momentums


def find_best_a(simulation):
    a_params = np.linspace(0.30, 0.51, 50)
    vs = []
    simulation.a = 0.38
    simulation.run()
    min_v = simulation.V
    best_a = simulation.a

    for a in a_params:
        simulation.a = a
        simulation.run()
        new_v = simulation.V
        if new_v < min_v:
            min_v = new_v
            best_a = a

        vs.append(new_v)

    print('Best pair:', best_a, min_v)

    with open('cristal/potential.dat', 'w+') as f:
        for a, v in zip(a_params, vs):
            f.write('%s %s\n' % (a, v))


def main():
    import sys
    configuration_file = sys.argv[1]
    output_file = sys.argv[2]

    simulation = Simulation(configuration_file, output_filename=output_file)
    simulation.run()
    find_best_a(simulation)


if __name__ == '__main__':
    main()