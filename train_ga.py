import multiprocessing
from util.fourier_series_agent import FourierSeriesAgent
import numpy as np
import gymnasium as gym
import pybullet_envs_gymnasium
import time
from threading import Thread
from multiprocessing import Process
import copy
import torch
from torch import nn
import matplotlib.pyplot as plt
#from torch.utils.tensorboard import SummaryWriter

class RewardPredictor(nn.Module):
    def __init__(
        self, 
        action_size : int, 
        n_frequencies : int, 
        hidden_size: int = 64
    ):
        super().__init__()
        self.dense1 = nn.Linear(action_size*n_frequencies*2 + 1, hidden_size)
        self.dense2 = nn.Linear(hidden_size, hidden_size)
        self.dense3 = nn.Linear(hidden_size, hidden_size)
        self.dense4 = nn.Linear(hidden_size, 1)
        self.layers = [
            self.dense1,
            nn.LeakyReLU(),
            self.dense2,
            nn.LeakyReLU(),
            self.dense3,
            nn.LeakyReLU(),
            self.dense4,
            nn.Tanh(),
        ]

    def forward(self, coefs, L):
        x = torch.tensor(coefs.flatten())
        x = torch.cat([x, torch.tensor([L])])
        for layer in self.layers:
            x = layer(x)
        return x


class GATrainer():
    def __init__(
        self,
        env,
        population,
        mutation_range = 0.2,
        coef_min=0.0,
        coef_max=1.0,
        n_frequencies=5,
        freq_step=0.01,
        kp=0.1,
        epochs=10,
        samples_per_epoch=50,
        checkpoint_path="saved_agents/best_agent.npy",
        tensorboard_path="tensorboard/"
    ):
        self.env = env
        self.coef_min = coef_min
        self.coef_max = coef_max
        self.mutation_range = mutation_range
        self.freq_step = freq_step
        self.population = population
        self.n_frequencies = n_frequencies
        self.kp = kp
        self.epochs = epochs
        self.samples_per_epoch = samples_per_epoch
        self.checkpoint_path = checkpoint_path
        self.tensorboard_path = tensorboard_path

        self.agents = []

        self.action_dim = env.action_space.shape[0]
        for i in range(population):
            coefs = np.random.uniform(
                low=coef_min, 
                high=coef_max, 
                size=(self.action_dim, n_frequencies, 2)
            )
            for j in range(self.action_dim):
                gamma = 1
                for k in range(n_frequencies):
                    coefs[j][k][0] *= gamma
                    coefs[j][k][1] *= gamma
                    gamma *= 0.6

            self.agents.append(FourierSeriesAgent(coefs))

    def normalize_reward(self, x, inverse=False): #hard coded once again woohoo!
        if(inverse):
            return np.arctanh(x) * 2000
        return np.tanh(x/2000)

    def calc_num_simulate(self, x):
        return int(-(self.population//2)*np.tanh((x-500)/400) + self.population//2)

    def agent_reward(self, agent, env, procnum, return_dict, ep_length=1000):
        t = 0
        total_reward = 0
        obs, _ = env.reset()

        for i in range(1000):
            # from documentation: action = hip_4, angle_4, hip_1, angle_1, hip_2, angle_2, hip_3, angle_3
            # observation: hip_4=11, ankle_4=12, hip_1=5, ankle_1=6, hip_2=7, ankle_2=8 hip_3=9, ankle_3=10
            joint_state = np.array([obs[11], obs[12], obs[5], obs[6], obs[7], obs[8], obs[9], obs[10]]) 

            wanted_state = agent.sample(i, deriv=False, norm=False)
        
            action = self.kp * (wanted_state-joint_state)
            obs, reward, _, _, _ = env.step(action)
            total_reward += reward
            t += 1

            """ logging for tests
            states.append(joint_state)
            wanted_states.append(wanted_state)
            times.append(i)
            actions.append(action)
            """

        return_dict[procnum] = total_reward


    def sample_coef_mutation(self, reward):
        new_range = self.mutation_range * 100/(reward-1000) #hard coded from mujoco env, todo 
        return np.random.uniform(
            low=1-new_range, 
            high=1+new_range, 
            size=(self.action_dim, self.n_frequencies, 2)
        )

    def sample_L_mutation(self, reward):
        new_range = self.mutation_range * 100/(reward-1000) #hard coded from mujoco env, todo 
        return np.random.uniform(
            low=1-new_range,
            high=1+new_range,
        )

    def train_step(self, generation=0, ep_length=1000):
        agent_rewards = []

        manager = multiprocessing.Manager()
        return_dict = manager.dict()
        jobs = []
        for i in range(self.population):
            p = multiprocessing.Process(target=self.agent_reward, args=[copy.deepcopy(self.agents[i]), self.env, i, return_dict])
            jobs.append(p)
            p.start()
        for p in jobs:
            p.join()

        for i in range(self.population):
            agent_rewards.append((self.agents[i], return_dict[i]))

        agent_rewards.sort(key = lambda x: x[1], reverse=True)
        best_reward = agent_rewards[0][1]
        print("best reward: {}".format(best_reward))
        agent_rewards[0][0].save()
        self.agents.clear()
        for i in range(self.population // 2):
            coefs = agent_rewards[i][0].coefs
            L = agent_rewards[i][0].L
            self.agents.append(agent_rewards[i][0])
            self.agents.append(FourierSeriesAgent(coefs * self.sample_coef_mutation(reward=best_reward), L*self.sample_L_mutation(reward=best_reward)))
        np.random.shuffle(self.agents)

    def train(self, generations=1000):
        for i in range(generations):
            self.train_step(generation=i)

if(__name__ == "__main__"):
    #env = gym.make("AntBulletEnv-v0", render_mode="rgb_array")
    #env = gym.make("AntBulletEnv-v0", render_mode="human")
    #env = gym.make("Ant-v4", reset_noise_scale=0, render_mode="human")
    env = gym.make("Ant-v4", reset_noise_scale=0)
    trainer = GATrainer(env, 50)
    trainer.train()
