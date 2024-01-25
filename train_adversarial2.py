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

class Generator(nn.Module):
    def __init__(
        self,
        action_size: int,
        n_frequencies: int,
        noise_size: int = 16,
        hidden_size: int = 64
    ):
        self.action_size = action_size
        self.n_frequencies = n_frequencies

        super().__init__();
        self.actor = nn.Sequential(
            nn.Linear(noise_size, hidden_size),
            nn.Mish(),
            nn.Linear(hidden_size, hidden_size),
            nn.Mish(),
            nn.Linear(hidden_size, hidden_size),
            nn.Mish(),
            nn.Linear(hidden_size, 2*action_size*n_frequencies+1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.actor(x)

    def sample_gait(self, mean, dev, n=1):
        sz = 2*self.action_size*self.n_frequencies+1 
        dist = torch.distributions.MultivariateNormal(
          torch.ones((sz), dtype=torch.double) * mean,
          torch.eye(sz, dtype=torch.double) * dev
        )
        sample = dist.sample_n(n)
        log_prob = dist.log_prob(sample)
        entropy = dist.entropy()
        return sample, log_prob, entropy


class Value(nn.Module):
    def __init__(
        self,
        action_size: int,
        n_frequencies: int,
        noise_size: int = 16,
        hidden_size: int = 64
    ):
        super().__init__();
        self.actor = nn.Sequential(
            nn.Linear(2*action_size*n_frequencies+1, hidden_size),
            nn.Mish(),
            nn.Linear(hidden_size, hidden_size),
            nn.Mish(),
            nn.Linear(hidden_size, hidden_size),
            nn.Mish(),
            nn.Linear(hidden_size, 1),
            nn.Sigmoid()
        )
        print(2*action_size*n_frequencies+1)

    def forward(self, x):
        print(x.shape)
        return self.actor(x)

class Trainer:
    def __init__(
        self,
        action_size : int,
        n_frequencies : int = 2, 
        gen_noise_size : int = 16,
        gen_hidden_size : int = 128,
        val_hidden_size : int = 256,
        kp : float = 0.01,
        k_critic: float = 0.1,
        k_entropy: float = 0.0000, # should be negative to maximize entropy ???
        reward_discount: float = 0.99
    ):
        self.action_size = action_size
        self.n_frequencies = n_frequencies
        self.gen_noise_size = gen_noise_size
        self.gen_hidden_size = gen_hidden_size
        self.val_hidden_size = val_hidden_size
        self.kp = kp
        self.k_critic = k_critic
        self.k_entropy = k_entropy
        self.reward_discount = reward_discount

        self.generator = Generator(
            action_size,
            n_frequencies,
            gen_noise_size,
            gen_hidden_size
        )
        self.generator.double()

        self.value = Value(
            action_size,
            n_frequencies,
            gen_noise_size,
            gen_hidden_size
        )
        self.value.double()
       
    def normalize_reward(self, x, inverse=False):
        if(inverse):
            return x * 200 + 900
        return (x-900) / 200

    def test(self, agents):
        env = gym.make("Ant-v4", render_mode="human", reset_noise_scale=0)

        t = 0
        total_reward = 0 
        obs, _ = env.reset()

        for i in range(1000):
            # from documentation: action = hip_4, angle_4, hip_1, angle_1, hip_2, angle_2, hip_3, angle_3
            # observation: hip_4=11, ankle_4=12, hip_1=5, ankle_1=6, hip_2=7, ankle_2=8 hip_3=9, ankle_3=10
            joint_state = np.array([obs[11], obs[12], obs[5], obs[6], obs[7], obs[8], obs[9], obs[10]]) 
            #joint_state = joint_state.T #magic!

            wanted_state = agents[0].sample(i, deriv=False, norm=False) * 5
            print(i, wanted_state)
        
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
        
        return total_reward


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

            wanted_state = np.array([agents[j].sample(i, deriv=False, norm=False) for j in range(len(agents))]) * 5
        
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

    def train(self, epochs=1000, batch_size=1, val_batch_size=100, value_samples=32):
        gen_optim = torch.optim.Adam(self.generator.parameters(), lr=0.01)
        val_optim = torch.optim.Adam(self.value.parameters(), lr=0.01)

        torch.autograd.set_detect_anomaly(True)
        noise = torch.rand((self.gen_noise_size), dtype=torch.double)

        sum_reward = 0
        weights_sum = 0

        for i in range(epochs):
            print("epoch %d" % i)

            #noise = torch.rand((self.gen_noise_size), dtype=torch.double)

            generated = self.generator(noise)
            mean = generated[:]
            std = generated[-1]
            #print(means.shape)
            value = self.value(generated.detach())
            #print(self.normalize_reward(value, inverse=True))
            value_detached = value.detach()
           
            simulate, log_prob, entropy = self.generator.sample_gait(mean, std, n=batch_size) 
            simulate_agents = []
            for j in range(batch_size):
                agent = simulate[j].detach().numpy()
                coefs = agent[:-1].reshape((self.action_size, self.n_frequencies, 2))
                L = agent[-1]
                simulate_agents.append(FourierSeriesAgent(coefs=coefs, L=10))
            actual_rewards = self.vec_test(simulate_agents)
            #actual_rewards = np.array([self.test(simulate_agents)])

            sum_reward = sum_reward * self.reward_discount + np.mean(actual_rewards)
            weights_sum = weights_sum * self.reward_discount + 1

            binary_rewards = torch.zeros(len(actual_rewards)).detach()
            for j in range(len(actual_rewards)):
                if(actual_rewards[j] >= sum_reward/weights_sum):
                    binary_rewards[j] = 1 

            actor_loss = torch.ones((1)) * self.k_entropy * entropy
            critic_loss = torch.zeros((1))
            for j in range(simulate.shape[0]):
                advantage_actor = binary_rewards[j] - value_detached
                advantage_critic = binary_rewards[j] - value
                #advantage_critic = value
                #advantage_actor = actual_rewards[j] - value_detached
                #advantage_critic = actual_rewards[j] - value 
                print("advantage: {} {}".format(advantage_actor, advantage_critic))
                print(log_prob[j])
                actor_loss += advantage_actor * log_prob[j]
                critic_loss += advantage_critic.pow(2)
                #critic_loss += nn.MSELoss()(value, torch.tensor([normalized_rewards[j]]))
            #print("value: {}".format(self.normalize_reward(value, inverse=True))) 
            print("entropy: {}, std: {}".format(entropy, std))
            print("actor loss: {}, critic loss: {}, rewards: {}".format(actor_loss, critic_loss, np.mean(actual_rewards)))
            #loss = actor_loss + self.k_critic * critic_loss 
            gen_optim.zero_grad()
            actor_loss.backward(retain_graph=True)
            gen_optim.step()
            
            val_optim.zero_grad()
            critic_loss.backward()
            val_optim.step()

if __name__ == "__main__":
    env = gym.make("Ant-v4", reset_noise_scale=0, render_mode="human")
    action_size = env.action_space.shape[0]
    trainer = Trainer(action_size=action_size)
    trainer.train()
