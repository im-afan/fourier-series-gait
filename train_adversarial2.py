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
from torch.optim import Adam, RMSprop
import matplotlib.pyplot as plt
#from torch.utils.tensorboard import SummaryWriter

class A2C(nn.Module):
    def __init__(
        self,
        action_size: int,
        n_frequencies: int,
        noise_size: int = 16,
        hidden_size: int = 64,
    ):
        super().__init__()

        self.action_size = action_size
        self.n_frequencies = n_frequencies 
        self.noise_size = noise_size 
        self.hidden_size = hidden_size 

        self.latent = nn.Sequential(
            nn.Linear(noise_size, hidden_size),
            nn.Mish(),
            nn.Linear(hidden_size, hidden_size),
            nn.Mish(),
            nn.Linear(hidden_size, hidden_size),
            nn.Mish()
        )
        self.critic = nn.Sequential(
            nn.Linear(hidden_size, 1),
            #nn.Sigmoid(),
        )
        self.actor = nn.Sequential(
            nn.Linear(hidden_size, 2*action_size*n_frequencies+1),
            nn.Sigmoid(),
        )

    def forward(self, noise):
        latent = self.latent(noise)
        value = self.critic(latent)
        generated = self.actor(latent)
        return generated, value

    def sample_gait(self, mean, dev, n=1):
        sz = 2*self.action_size*self.n_frequencies+1 
        dist = torch.distributions.MultivariateNormal(torch.zeros((sz)), torch.eye(sz))
        return dist.sample_n(n) * dev + mean.repeat(n, 1)
        

class Trainer:
    def __init__(
        self,
        action_size : int,
        n_frequencies : int = 5, 
        gen_noise_size : int = 16,
        gen_hidden_size : int = 128,
        val_hidden_size : int = 256,
        kp : float = 0.01,
        k_critic: float = 1.0,
        k_entropy: float = 1.0,
    ):
        self.action_size = action_size
        self.n_frequencies = n_frequencies
        self.gen_noise_size = gen_noise_size
        self.gen_hidden_size = gen_hidden_size
        self.val_hidden_size = val_hidden_size
        self.kp = kp
        self.k_critic = k_critic
        self.k_entropy = k_entropy

        self.model = A2C(
            action_size,
            n_frequencies,
            noise_size=gen_noise_size,
            hidden_size=gen_hidden_size
        )
        self.model.double()
       
    def normalize_reward(self, x, inverse=False):
        if(inverse):
            return x * 200 + 900
        return (x-900) / 200

    def vec_test(self, agents):
        envs = gym.vector.make("Ant-v4", num_envs=len(agents), reset_noise_scale=0)

        t = 0
        total_reward = np.zeros((len(agents)))
        obs, _ = envs.reset()

        for i in range(1000):
            # from documentation: action = hip_4, angle_4, hip_1, angle_1, hip_2, angle_2, hip_3, angle_3
            # observation: hip_4=11, ankle_4=12, hip_1=5, ankle_1=6, hip_2=7, ankle_2=8 hip_3=9, ankle_3=10
            joint_state = np.array([obs[:, 11], obs[:, 12], obs[:, 5], obs[:, 6], obs[:, 7], obs[:, 8], obs[:, 9], obs[:, 10]]) 
            joint_state = joint_state.T #magic!

            wanted_state = np.array([agents[j].sample(i, deriv=False, norm=False) for j in range(len(agents))])
        
            action = self.kp * (wanted_state-joint_state)
            obs, reward, _, _, _ = envs.step(action)
            total_reward += reward
            t += 1

            """ logging for tests
            states.append(joint_state)
            wanted_states.append(wanted_state)
            times.append(i)
            actions.append(action)
            """
        
        return total_reward

    def train(self, epochs=1000, batch_size=32, val_batch_size=100, value_samples=32):
        optim = torch.optim.Adam(self.model.parameters(), lr=0.004)
        """actor_optim = torch.optim.Adam([
            {"params": self.model.latent.parameters(), "lr": 0.002},
            {"params": self.model.actor.parameters()},
        ], lr=0.004)
        critic_optim = torch.optim.Adam([
            {"params": self.model.latent.parameters(), "lr": 0.002},
            {"params": self.model.critic.parameters()},
        ], lr=0.004)"""
        #gen_loss_fn = nn.BCELoss()
        #val_loss_fn = nn.BCELoss()
        loss_fn = nn.BCELoss()
        #noise = torch.rand((batch_size, self.gen_noise_size), dtype=torch.double)
        for i in range(epochs):
            print("epoch %d" % i)
            optim.zero_grad()

            noise = torch.rand((self.gen_noise_size), dtype=torch.double)
            means, value = self.model(noise)
            value_detached = value.detach()
            std = 1 #todo use a different scheduler
           
            simulate = self.model.sample_gait(means, std, n=batch_size) 
            simulate_agents = []
            for j in range(batch_size):
                agent = simulate[j].detach().numpy()
                coefs = agent[:-1].reshape((self.action_size, self.n_frequencies, 2))
                L = agent[-1]
                simulate_agents.append(FourierSeriesAgent(coefs=coefs, L=L))
            actual_rewards = self.vec_test(simulate_agents)

            binary_rewards = torch.zeros(len(actual_rewards))
            normalized_rewards = self.normalize_reward(actual_rewards)
            inds = list(range(simulate.shape[0]))
            inds.sort(key = lambda x: actual_rewards[x], reverse=True)
            for j in range(len(inds)):
                ind = inds[j]
                binary_rewards[ind] = j/len(inds)

            actor_loss = torch.zeros((1))
            critic_loss = torch.zeros((1))
            for j in range(simulate.shape[0]):
                advantage_actor = normalized_rewards[j] - value_detached
                advantage_critic = normalized_rewards[j] - value 
                #advantage_actor = actual_rewards[j] - value_detached
                #advantage_critic = actual_rewards[j] - value 
                print("advantage: {} {}".format(advantage_actor, advantage_critic))
                actor_loss += -advantage_actor
                critic_loss += advantage_critic**2 
          
            print("actor loss: {}, critic loss: {}, rewards: {}".format(actor_loss, critic_loss, np.mean(actual_rewards)))
            loss = actor_loss + self.k_critic * critic_loss 
            loss.backward()
            optim.step()

if __name__ == "__main__":
    env = gym.make("Ant-v4")
    action_size = env.action_space.shape[0]
    trainer = Trainer(action_size=action_size)
    trainer.train()
