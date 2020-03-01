import functions
import math
import random


class SensoryNeuron:
    def __init__(self, f='LINEAR'):
        self.activation = 0
        self.activation_function = functions.function(f)

    def activate(self, i):
        self.activation = self.activation_function(i)


class Unit:
    def __init__(self, f='LINEAR'):
        self.activation = 0
        self.inputs = []
        self.function = functions.function(f)

    def add_input(self, i=None):
        for x in i:
            self.inputs.append(x)

    def activation(self):
        return self.activation


class Neuron:
    def __init__(self, f='LINEAR'):
        self.activation = 0
        self.update = 0
        self.inputs = 0
        self.activation_function = functions.function(f)

    def input(self, i):
        self.inputs += i

    def activate(self):
        self.activation = self.activation_function(self.inputs)
        self.inputs = 0


class Weight:
    def __init__(self, n1, n2, f='RELU'):
        self.n1 = n1
        self.n2 = n2
        self.function = functions.function(f)
        self.step_size = 0.5
        self.decay_rate = 0.5
        self.strength = 0.5

    def simulate(self):
        self.n2.input(self.strength * self.n1.activation)

    def learn(self):
        self.strength += self.strength * self.step_size * self.n1.activation * self.n2.activation - self.step_size * self.decay_rate * self.strength
        self.strength = self.function(self.strength)
