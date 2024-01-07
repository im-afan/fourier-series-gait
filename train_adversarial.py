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
from torch.optim import Adam
import matplotlib.pyplot as plt
#from torch.utils.tensorboard import SummaryWriter

class Value(nn.Module):
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
        self.seq = nn.Sequential(
            self.dense1,
            nn.BatchNorm1d(hidden_size),
            #nn.Dropout(),
            nn.Mish(),
            self.dense2,
            nn.BatchNorm1d(hidden_size),
            #nn.Dropout(),
            nn.Mish(),
            self.dense3,
            nn.BatchNorm1d(hidden_size),
            #nn.Dropout(),
            nn.Mish(),
            self.dense4,
            nn.BatchNorm1d(1),
            nn.Sigmoid(),
        )
        self.double()

    def forward(self, *args):
        if(len(args) == 1):
            x = args[0]
            return self.seq(x) 

        coefs, L = args[0], args[1]
        x = torch.tensor(coefs.flatten())
        x = torch.cat([x, torch.tensor([L])])
        return self.seq(x)
        
class Generator(nn.Module):
    def __init__(
        self, 
        action_size : int, 
        n_frequencies : int, 
        hidden_size: int = 64,
        noise_size: int = 16
    ):
        super().__init__()
        self.dense1 = nn.Linear(noise_size, hidden_size)
        self.dense2 = nn.Linear(hidden_size, hidden_size)
        self.dense3 = nn.Linear(hidden_size, hidden_size)
        self.dense4 = nn.Linear(hidden_size, action_size*n_frequencies*2 + 1)
        self.seq = nn.Sequential(
            self.dense1,
            nn.BatchNorm1d(hidden_size),
            nn.Mish(),
            self.dense2,
            nn.BatchNorm1d(hidden_size),
            nn.Mish(),
            self.dense3,
            nn.BatchNorm1d(hidden_size),
            nn.Mish(),
            self.dense4,
            nn.BatchNorm1d(action_size*n_frequencies*2+1),
            nn.Sigmoid(),
        )
        self.double()

    def forward(self, x):
        return self.seq(x) 

class Trainer:
    def __init__(
        self,
        action_size : int,
        n_frequencies : int = 5, 
        gen_noise_size : int = 16,
        gen_hidden_size : int = 128,
        val_hidden_size : int = 128,
        kp : float = 0.01,
    ):
        self.action_size = action_size
        self.n_frequencies = n_frequencies
        self.gen_noise_size = gen_noise_size
        self.gen_hidden_size = gen_hidden_size
        self.val_hidden_size = val_hidden_size
        self.kp = kp

        self.generator = Generator(action_size, n_frequencies, gen_hidden_size, gen_noise_size)
        self.value = Value(action_size, n_frequencies, val_hidden_size)
        
        self.dataset_train = []
        self.dataset_test = []

    def normalize_reward1(self, x, inverse=False): #hard coded once again woohoo!
        if(inverse):
            return 500*np.arctanh(2*x-1) + 1500 
        return (np.tanh((x-1500)/500)+1)/2

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

    def train(self, epochs=1000, batch_size=32, value_samples=16):
        gen_optim = Adam(self.generator.parameters())
        val_optim = Adam(self.value.parameters())
        gen_loss_fn = nn.BCELoss()
        val_loss_fn = nn.HuberLoss()
        for i in range(epochs):
            print("epoch %d" % i)
            gen_optim.zero_grad()

            noise = torch.rand((batch_size, self.gen_noise_size), dtype=torch.double)
            generated = self.generator(noise)
            
            """
            for j in range(value_samples):
                sample = generated[i].detach().numpy()
                L = sample[-1]
                coefs = sample[:-1].reshape((self.n_frequencies, 2))

                agent = FourierSeriesAgent(coefs, L)
                reward = agent.test(gym.make("Ant-v4"))
                self.dataset.append((sample, self.normalize_reward(reward)))
            """
            generated_dataset = generated.detach().numpy()
            simulate = generated_dataset[:value_samples]
            simulate_agents = []
            for j in range(value_samples):
                coefs = simulate[j][:-1].reshape((self.action_size, self.n_frequencies, 2))
                L = simulate[j][-1]
                simulate_agents.append(FourierSeriesAgent(coefs=coefs, L=L))
            actual_rewards = self.vec_test(simulate_agents)
            print(actual_rewards)
            for j in range(value_samples):
                if(j < 12):
                    self.dataset_train.append((simulate[j], self.normalize_reward(actual_rewards[j])))
                else:
                    self.dataset_test.append((simulate[j], self.normalize_reward(actual_rewards[j])))

            for l in range(3):
                val_optim.zero_grad()
                logits, labels = [], []
                logits_test, labels_test = [], []
                for j in range(batch_size):
                    if(j < 12):
                        k = len(self.dataset_train)-j-1
                    else:
                        k = np.random.randint(len(self.dataset_train))
                    logits.append(self.dataset_train[k][0])
                    labels.append(self.dataset_train[k][1])
                    k = np.random.randint(len(self.dataset_test))
                    logits_test.append(self.dataset_test[k][0])
                    labels_test.append(self.dataset_test[k][1])
                logits = torch.tensor(np.array(logits))
                labels = torch.tensor(np.array(labels))
                logits_test = torch.tensor(np.array(logits_test))
                labels_test = torch.tensor(np.array(labels_test))

                output = self.value(logits)
                val_loss = val_loss_fn(output.squeeze(), labels)
                print("value train loss: {}".format(val_loss))
                val_loss.backward()
                val_optim.step()
           
                output_test = self.value(logits_test)
                val_loss_test = val_loss_fn(output_test.squeeze(), labels_test)
                print("value test loss: {}".format(val_loss_test))
            
            predicted_value_test = self.value(torch.tensor(np.array(simulate)))
            print(self.normalize_reward(predicted_value_test.detach().numpy(), inverse=True).squeeze())
            
            #plt.scatter(actual_rewards, self.normalize_reward(predicted_value_test.detach().numpy(), inverse=True))
            #plt.show()

            predicted_value = self.value(generated)
            
            gen_loss = gen_loss_fn(predicted_value.squeeze(), torch.ones((batch_size), dtype=torch.double))
            gen_loss.backward()
            gen_optim.step()


if __name__ == "__main__":
    env = gym.make("Ant-v4")
    action_size = env.action_space.shape[0]
    trainer = Trainer(action_size=action_size)
    trainer.train()
