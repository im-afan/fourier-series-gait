import gymnasium as gym
import torch
from torch import nn
import fourier_series_agent

class AntLowFrequency(gym.envs.mujoco.Antv4):
    def __init__(self, generator: nn.Module, frame_skip: int, **kwargs):
        super.__init__(**kwargs)
        self.generator = generator
        self.frame_skip = frame_skip

    def step(self, action):
        generated = self.generator(action).detach().numpy()
        agent = fourier_series_agent.from_array(generated)

        obs = None
        reward = 0
        terminated = False
        
        for i in range(self.frame_skip):
            wanted_state = agent.sample(i, deriv=False, norm=False)
            # from documentation: action = hip_4, angle_4, hip_1, angle_1, hip_2, angle_2, hip_3, angle_3
            # observation: hip_4=11, ankle_4=12, hip_1=5, ankle_1=6, hip_2=7, ankle_2=8 hip_3=9, ankle_3=10
            joint_state = np.array([obs[11], obs[12], obs[5], obs[6], obs[7], obs[8], obs[9], obs[10]]) 

            a = self.kp * (wanted_state - joint_state)

            obs, d_reward, terminated, _, info = super().step(a)
            reward += d_reward
            if(terminated):
                break

        return obs, reward, terminated, False, None


