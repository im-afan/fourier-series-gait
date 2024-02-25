import multiprocessing
from util.fourier_series_agent import FourierSeriesAgent
from util.fourier_series_agent import from_array
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
from torch.utils.tensorboard import SummaryWriter
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
            nn.Linear(hidden_size, 2*(2*action_size*n_frequencies+1)),
            #nn.Sigmoid(),
        )

        logstd_param = nn.Parameter(2*torch.rand(2*action_size*n_frequencies+1) - 1)
        self.register_parameter("logstd", logstd_param)
        #print(self.logstd)

    def forward(self, x):
        u = self.actor(x)
        return torch.cat((u[:u.shape[0]//2], self.logstd))

    def sample_gait(self, mean, dev, n=1):
        sz = 2*self.action_size*self.n_frequencies+1 
        dist = torch.distributions.MultivariateNormal(
          torch.ones((sz), dtype=torch.double) * mean,
          torch.diag(dev).to(torch.double)
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
        n_frequencies : int = 3, 
        gen_noise_size : int = 2,
        gen_hidden_size : int = 128,
        val_hidden_size : int = 256,
        kp : float = 0.01,
        k_critic: float = 0.1,
        #k_entropy: float = -0.0001, # should be negative to maximize entropy ???
        k_entropy: float = 0.1,
        reward_discount: float = 0.5
    ):
        #print("aciton size: ", action_size)
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

        self.writer = SummaryWriter(log_dir="tensorboard/")
       
    def normalize_reward(self, x, inverse=False):
        if(inverse):
            return x * 200 + 900
        return (x-900) / 200

    def test(self, agents):
        env = gym.make("Hopper-v4", render_mode="human", reset_noise_scale=0)

        t = 0
        total_reward = 0 
        obs, _ = env.reset()

        for i in range(1000):
            # FOR ANT
            # from documentation: action = hip_4, angle_4, hip_1, angle_1, hip_2, angle_2, hip_3, angle_3
            # observation: hip_4=11, ankle_4=12, hip_1=5, ankle_1=6, hip_2=7, ankle_2=8 hip_3=9, ankle_3=10
            #joint_state = np.array([obs[11], obs[12], obs[5], obs[6], obs[7], obs[8], obs[9], obs[10]]) 
            #joint_state = joint_state.T #magic!

            # FOR HOPPER
            # from documentation: action: thigh_joint, leg_joint, foot_joint
            # observation: thigh_joint=2, leg_joint=3, foot_joint=4
            joint_state = np.array([obs[2], obs[3], obs[4]])

            wanted_state = agents[0].sample(i, deriv=False, norm=False)
        
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
        envs = gym.vector.make("Hopper-v4", num_envs=len(agents), reset_noise_scale=0)

        t = 0
        total_reward = np.zeros((len(agents)))
        obs, _ = envs.reset()

        for i in range(1000):
            # FOR ANT
            # from documentation: action = hip_4, angle_4, hip_1, angle_1, hip_2, angle_2, hip_3, angle_3
            # observation: hip_4=11, ankle_4=12, hip_1=5, ankle_1=6, hip_2=7, ankle_2=8 hip_3=9, ankle_3=10
            # joint_state = np.array([obs[:, 11], obs[:, 12], obs[:, 5], obs[:, 6], obs[:, 7], obs[:, 8], obs[:, 9], obs[:, 10]]) 
            # joint_state = joint_state.T #magic!

            # FOR HOPPER
            # from documentation: action: thigh_joint, leg_joint, foot_joint
            # observation: thigh_joint=2, leg_joint=3, foot_joint=4
            joint_state = np.array([obs[:, 2], obs[:, 3], obs[:, 4]])
            joint_state = joint_state.T

            # FOR HALF CHEETAH
            # from documentation: action: bthigh, bshin, bfoot, fthigh, fshin, ffoot
            # observation: bthigh=2, bshin=3, bfoot

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

    def train(self, epochs=1000, batch_size=32, val_batch_size=100, value_samples=32):
        gen_optim = torch.optim.Adam(self.generator.parameters(), lr=0.003)
        val_optim = torch.optim.Adam(self.value.parameters(), lr=0.01)

        torch.autograd.set_detect_anomaly(True)
        noise = torch.rand((self.gen_noise_size), dtype=torch.double)

        sum_reward = 0
        weights_sum = 0

        #noise = torch.normal(mean=torch.zeros((self.gen_noise_size)), std=torch.ones((self.gen_noise_size)))
        for i in range(epochs):
            print("epoch %d" % i)

            #noise = torch.rand((self.gen_noise_size))
            noise = torch.tensor(noise, dtype=torch.double)

            generated = self.generator(noise)
            sz = generated.shape[0]
            mean = generated[:sz//2]
            log_std = generated[sz//2:]
            print("mean: ", mean)
            #print(log_std)
            #print(means.shape)
            #value = self.value(generated.detach())
            #print(self.normalize_reward(value, inverse=True))
            #value_detached = value.detach()

            np.savetxt("saved_agents/" + str(i) + ".txt", mean.detach().numpy())
           
            simulate, log_prob, entropy = self.generator.sample_gait(mean, torch.exp(log_std), n=batch_size) 
            simulate_agents = []
            for j in range(batch_size):
                #agent = simulate[j].detach().numpy()
                #coefs = agent[:-1].reshape((self.action_size, self.n_frequencies, 2))
                #L = agent[-1]
                #simulate_agents.append(FourierSeriesAgent(coefs=coefs, L=10))
                simulate_agents.append(from_array(self.action_size, self.n_frequencies, simulate[j].detach().numpy()))
            actual_rewards = self.vec_test(simulate_agents)
            #actual_rewards = np.array([self.test(simulate_agents)])
 
            binary_rewards = -torch.ones(len(actual_rewards)).detach()
            for j in range(len(actual_rewards)):
                if(weights_sum == 0):
                    binary_rewards[j] = 1
                elif(actual_rewards[j] >= sum_reward/weights_sum):
                    binary_rewards[j] = 1 

            actor_loss = torch.ones((1)) * self.k_entropy * entropy
            #actor_loss = torch.ones((1)) * self.k_entropy * (10*torch.mean(std))
            critic_loss = torch.zeros((1))
            #print("predicted value: {}".format(value))
            for j in range(simulate.shape[0]):
                #advantage_actor = nn.BCELoss()(binary_rewards[j], 1)
                advantage_actor = binary_rewards[j]
                actor_loss += -advantage_actor * log_prob[j]
            #print("value: {}".format(self.normalize_reward(value, inverse=True))) 
            print("entropy: {}, std: {}".format(entropy, log_std))
            print("actor loss: {}, critic loss: {}, rewards: {}".format(actor_loss, critic_loss, np.mean(actual_rewards)))
            print("binary reward: {}".format(np.mean(binary_rewards.numpy())))

            self.writer.add_scalar("Reward", np.mean(actual_rewards))
            self.writer.add_scalar("Binary Reward", np.mean(binary_rewards.numpy()))
            self.writer.add_scalar("Entropy", entropy)

            if(weights_sum != 0):
                print("avg reward (discounted): {}".format(sum_reward/weights_sum))
            #loss = actor_loss + self.k_critic * critic_loss 
            gen_optim.zero_grad()
            actor_loss.backward()
            nn.utils.clip_grad_norm_(self.generator.parameters(), 0.01)
            gen_optim.step()

            sum_reward = sum_reward * self.reward_discount + np.mean(actual_rewards)
            weights_sum = weights_sum * self.reward_discount + 1
            
            #val_optim.zero_grad()
            #critic_loss.backward()
            #val_optim.step()

if __name__ == "__main__":
    env = gym.make("Hopper-v4", reset_noise_scale=0, render_mode="human")
    #env = gym.make("Ant-v4", reset_noise_scale=0, render_mode="human")
    action_size = env.action_space.shape[0]
    trainer = Trainer(action_size=action_size)
    trainer.train()
