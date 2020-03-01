import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np
import random
import network

plt.style.use('ggplot')

network = network.Network()

# Create input data
data = []
x = np.linspace(0, np.pi * 2, 256)
for i in range(0, 1000):
    sinx = np.sin(0.1 * i + x)
    data.append(sinx)
data = np.array(data)

# Simulate the network
for d in data:
    network.simulate(d)
simulation = np.array(network.history)

# Create the plot
fig, ax = plt.subplots(figsize=(5, 3))
line_input = ax.plot(x, simulation[0, 0, :], color='black')[0]
line_neurons = ax.plot(x, simulation[0, 1, :], color='blue')[0]
line_weights = ax.plot(x, simulation[0, 2, :], color='red')[0]

def animate(i):
    line_input.set_ydata(simulation[i, 0, :])
    line_neurons.set_ydata(simulation[i, 1, :])
    line_weights.set_ydata(simulation[i, 2, :])


anim = FuncAnimation(
    fig, animate, interval=1, frames=1000)

plt.draw()
plt.show()
