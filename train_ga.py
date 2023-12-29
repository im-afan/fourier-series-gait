import multiprocessing
from util.fourier_series_agent import FourierSeriesAgent
import numpy as np
import gymnasium as gym
import pybullet_envs_gymnasium
import time
from threading import Thread
from multiprocessing import Process
import copy

class GATrainer():
    def __init__(
        self,
        env,
        population,
        mutation_coef_min=0.8,
        mutation_coef_max=1.2,
        coef_min=0.0,
        coef_max=1.0,
        n_frequencies=3,
        freq_step=0.01
    ):
        self.env = env
        self.coef_min = coef_min
        self.coef_max = coef_max
        self.mutation_coef_min = mutation_coef_min
        self.mutation_coef_max = mutation_coef_max
        self.freq_step = freq_step
        self.population = population
        self.n_frequencies = n_frequencies

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

    def agent_reward(self, agent, env, procnum, return_dict, ep_length=1000):
        t = 0
        total_reward = 0
        obs, _ = env.reset()
        for i in range(ep_length):
            #wanted_pos = self.agents[ind].sample(t, L=100, deriv=False)
            #time.sleep(0.01)
            # from documentation: action = hip_4, angle_4, hip_1, angle_1, hip_2, angle_2, hip_3, angle_3
            # observation: hip_4=11, ankle_4=12, hip_1=5, ankle_1=6, hip_2=7, ankle_2=8 hip_3=9, ankle_3=10
            joint_state = np.array([obs[11], obs[12], obs[5], obs[6], obs[7], obs[8], obs[9], obs[10]]) 
            wanted_state = agent.sample(t, L=100, deriv=False, norm=True)
            
            _, reward, _, _, _ = env.step(wanted_state-joint_state)
            total_reward += reward
            #t += self.freq_step
            #print(i)
            t += 1
        #print(total_reward)
        return_dict[procnum] = total_reward

    def sample_mutation(self):
        return np.random.uniform(
            low=self.mutation_coef_min, 
            high=self.mutation_coef_max, 
            size=(self.action_dim, self.n_frequencies, 2)
        )

    def train_step(self, ep_length=1000):
        """agent_rewards = []
        t = []
        for i in range(len(self.agents)):
            t.append(ProcessWithReturn(target=self.agent_reward, args=[copy.deepcopy(self.agents[i]), copy.deepcopy(self.env)]))
        for i in range(len(t)):
            t[i].start()
        for i in range(len(t)):
            agent_rewards.append((self.agents[i], t[i].join()))"""
        agent_rewards = []
        manager = multiprocessing.Manager()
        return_dict = manager.dict()
        jobs = []
        for i in range(self.population):
            #p = multiprocessing.Process(target=self.agent_reward, args=[copy.deepcopy(self.agents[i]), copy.deepcopy(self.env), i, return_dict])
            p = multiprocessing.Process(target=self.agent_reward, args=[copy.deepcopy(self.agents[i]), self.env, i, return_dict])
            jobs.append(p)
            p.start()
        for p in jobs:
            p.join()
        for i in range(self.population):
            print("reward: {}".format(return_dict[i]))
            agent_rewards.append((self.agents[i], return_dict[i]))
        #print(return_dict)
        #for i in range(len(t)):
        #    agent_rewards.append((self.agents[i], t[i].get_return()))
            #print("reward: {}".format(agent_rewards[i][1]))
        #sort(self.agents) 

        agent_rewards.sort(key = lambda x: x[1], reverse=True)
        print("best reward: {}".format(agent_rewards[0][1]))
        self.agents.clear()
        for i in range(self.population // 2):
            coefs = agent_rewards[i][0].coefs
            self.agents.append(FourierSeriesAgent(coefs * self.sample_mutation()))
            self.agents.append(FourierSeriesAgent(coefs * self.sample_mutation()))

    def train(self, generations=1000):
        for i in range(generations):
            self.train_step()

if(__name__ == "__main__"):
    #env = gym.make("AntBulletEnv-v0", render_mode="rgb_array")
    env = gym.make("AntBulletEnv-v0")
    trainer = GATrainer(env, 50)
    trainer.train()
