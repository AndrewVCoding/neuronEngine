import neuron


class Network():
    def __init__(self):
        # Create the initial layer of inputs neurons
        self.inputs = []
        self.neurons = []
        self.weights = []
        self.history = []

        for i in range(0, 256):
            n1 = neuron.SensoryNeuron()
            n2 = neuron.Neuron()
            w = neuron.Weight(n1, n2)
            self.inputs.append(n1)
            self.neurons.append(n2)
            self.weights.append(w)


    def simulate(self, input):
        for i in range(0, len(self.inputs)):
            self.inputs[i].activate(input[i])
        for n in self.neurons:
            n.activate()
        for w in self.weights:
            w.simulate()
        for w in self.weights:
            w.learn()
        self.record()

    def record(self):
        i = []
        n = []
        w = []
        for x in self.inputs:
            i.append(x.activation)
        for x in self.neurons:
            n.append(x.activation)
        for x in self.weights:
            w.append(x.strength)
        self.history.append([i, n, w])
