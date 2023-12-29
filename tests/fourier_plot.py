import sys
import numpy as np
import matplotlib.pyplot as plt
from util.fourier_series_agent import FourierSeriesAgent

#agent = FourierSeriesAgent(np.array([[[1, 1], [2, 2], [3, 3]]]))
agent = FourierSeriesAgent(np.array([[[0, 1], [0, 0.5], [0, 0.33], [0, 0.25], [0, 0.2], [0, 0.167]]]))
x, y = [], []
for i in range(100):
    t = i/10
    x.append(t)
    y.append(agent.sample(t)[0])

plt.scatter(x, y)
plt.show()
