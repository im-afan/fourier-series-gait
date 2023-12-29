import sys
import numpy as np
import matplotlib.pyplot as plt
from util.fourier_series_agent import FourierSeriesAgent

#agent = FourierSeriesAgent(np.array([[[1, 1], [2, 2], [3, 3]]]))
#wave = [[1, 1], [1, 1]]
#agent = FourierSeriesAgent(np.array([[[0, 1], [0, 0.5], [0, 0.33], [0, 0.25], [0, 0.2], [0, 0.167]]]))
agent = FourierSeriesAgent(np.array([[[0, 0], [0, 1], [0, 0], [0, 0.33], [0, 0], [0, 0.2], [0, 0]]]))
#agent = FourierSeriesAgent(np.array([wave]))
dx, dy = [], []
x, y = [], []
for i in range(1000):
    t = i/100

    x.append(t)
    y.append(agent.sample(t, deriv=False, L=1)[0])
    dx.append(t)
    dy.append(agent.sample(t, deriv=True, L=1)[0])

plt.plot(x, y)
#plt.plot(dx, dy)
plt.show()
