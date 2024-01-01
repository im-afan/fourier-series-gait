import multiprocessing
from util.fourier_series_agent import FourierSeriesAgent
import numpy as np
import gymnasium as gym
import pybullet_envs_gymnasium
import time
from threading import Thread
from multiprocessing import Process
import copy

class GATester():
    def __init__(
        self,
        env,
        n_frequencies=5,
        freq_step=0.01,
        kp=0.1,
    ):
        self.env = env
        self.freq_step = freq_step
        self.n_frequencies = n_frequencies
        self.kp = kp
        self.action_dim = env.action_space.shape[0]

        self.agent = FourierSeriesAgent(np.zeros((self.action_dim, n_frequencies, 2)))
        self.agent.load()
        
    def test(self, timesteps=10000):
        t = 0
        total_reward = 0
        obs, _ = self.env.reset()

        for i in range(timesteps):
            # from documentation: action = hip_4, angle_4, hip_1, angle_1, hip_2, angle_2, hip_3, angle_3
            # observation: hip_4=11, ankle_4=12, hip_1=5, ankle_1=6, hip_2=7, ankle_2=8 hip_3=9, ankle_3=10
            joint_state = np.array([obs[11], obs[12], obs[5], obs[6], obs[7], obs[8], obs[9], obs[10]]) 

            wanted_state = self.agent.sample(i, deriv=False, norm=False)
        
            action = self.kp * (wanted_state-joint_state)
            obs, reward, _, _, _ = env.step(action)
            total_reward += reward
            t += 1

        print("total reward: {}".format(total_reward))



if(__name__ == "__main__"):
    env = gym.make("Ant-v4", reset_noise_scale=0, render_mode="human")
    trainer = GATester(env) 
    trainer.test()
