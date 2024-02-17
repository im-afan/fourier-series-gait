import numpy as np
import matplotlib.pyplot as plt
from util.fourier_series_agent import FourierSeriesAgent
import pybullet_envs_gymnasium
import gymnasium as gym

#agent = FourierSeriesAgent(np.array([[[1, 1], [2, 2], [3, 3]]]))
env = gym.make("AntBulletEnv-v0", render_mode="human", reset_noise_scale=0)
#env = gym.make("Ant-v4", render_mode="human", reset_noise_scale=0)
action_size = env.action_space.shape[0]

#wave = [[0, 0], [0, 1], [0, 0], [0, 0.33], [0, 0], [0, 0.2], [0, 0]]
#wave = [[0, 0], [0, 1], [0, 0], [0, 0.33], [0, 0], [0, 0.2], [0, 0]]
#wave = [[0, 0], [0.1, 0.1]]
obs, _ = env.reset()

for i in range(1000):
    #wanted_pos = self.agents[ind].sample(t, L=100, deriv=False)
    #time.sleep(0.01)
    # from documentation: action = hip_4, angle_4, hip_1, angle_1, hip_2, angle_2, hip_3, angle_3
    # observation: hip_4=11, ankle_4=12, hip_1=5, ankle_1=6, hip_2=7, ankle_2=8 hip_3=9, ankle_3=10
    joint_state = np.array([obs[11], obs[12], obs[5], obs[6], obs[7], obs[8], obs[9], obs[10]]) 
    #times.append(np.ones((action_size)) * i)
    print(joint_state[1])

   
    #action = np.random.uniform(low=-1, high=1, size=(action_size))
    action = np.zeros((action_size))
    obs, reward, _, _, _ = env.step(action)
    #print(wanted_state)
    #t += self.freq_step
    #print(i)



