import numpy as np
import matplotlib.pyplot as plt
from util.fourier_series_agent import FourierSeriesAgent
import pybullet_envs_gymnasium
import gymnasium as gym

#agent = FourierSeriesAgent(np.array([[[1, 1], [2, 2], [3, 3]]]))
env = gym.make("AntBulletEnv-v0", render_mode="human")
action_size = env.action_space.shape[0]

wave = [[0, 0], [0, 1], [0, 0], [0, 0.33], [0, 0], [0, 0.2], [0, 0]]
#wave = [[0, 0], [1, 1]]
joints = [wave for i in range(action_size)]

agent = FourierSeriesAgent(np.array(joints))

obs = env.reset()

t = 0
while(True):
    t += 0.01
    action = agent.sample(t) * 0.1
    print(action)
    env.step(action)
    #env.render()

