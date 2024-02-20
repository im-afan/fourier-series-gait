from typing import List, Tuple
import numpy as np
import matplotlib.pyplot as plt

class FourierSeriesAgent():
    def __init__(
        self,
        coefs: np.ndarray,
        L: int = 10,
        kp: float = 0.01,
    ):
        assert(len(coefs.shape) == 3)
        assert(coefs.shape[2] == 2)

        self.coefs = np.array(coefs)
        self.joints = len(coefs)
        self.n = len(coefs[0])
        self.L = L
        self.kp = kp

        self.max_vals = np.zeros(shape=coefs.shape[0])
        for i in range(len(coefs)):
            for p in coefs[i]:
                self.max_vals[i] += p[0]**2 + p[1]**2
            self.max_vals[i] = np.sqrt(self.max_vals[i])
        #print(self.max_vals)

    
    def sample(self, t: float, deriv: bool = True, norm: bool = True) -> np.ndarray:
        res = np.zeros((self.joints))
        for j in range(self.n):
            if(deriv):
                res += j * -self.coefs[:, j, 0] * np.sin(j*t/self.L)
                res += j * self.coefs[:, j, 1] * np.cos(j*t/self.L)
            else:
                res += self.coefs[:, j, 0] * np.cos(j*t/self.L)
                res += self.coefs[:, j, 1] * np.sin(j*t/self.L)

        if(norm):
            return res / self.max_vals
        return res

    def test(self, env, ep_length=1000):
        t = 0
        total_reward = 0
        obs, _ = env.reset()

        for i in range(ep_length):
            # from documentation: action = hip_4, angle_4, hip_1, angle_1, hip_2, angle_2, hip_3, angle_3
            # observation: hip_4=11, ankle_4=12, hip_1=5, ankle_1=6, hip_2=7, ankle_2=8 hip_3=9, ankle_3=10
            joint_state = np.array([obs[11], obs[12], obs[5], obs[6], obs[7], obs[8], obs[9], obs[10]]) 

            wanted_state = self.sample(i, deriv=False, norm=False)
        
            action = self.kp * (wanted_state-joint_state)
            obs, reward, _, _, _ = env.step(action)
            total_reward += reward
            t += 1

        return total_reward

    def save(self, coef_checkpoint="saved_agents/best_agent_coef.npy", L_checkpoint="saved_agents/best_agent_L.npy"):
        np.save(coef_checkpoint, self.coefs)
        np.save(L_checkpoint, np.array([self.L]))

    def load(self, coef_checkpoint="saved_agents/best_agent_coef.npy", L_checkpoint="saved_agents/best_agent_L.npy"):
        self.coefs = np.load(coef_checkpoint)
        self.L = np.load(L_checkpoint)

def from_array(joints: int, n: int, a: np.ndarray, kp: float=0.01):
    coefs = np.array(a[:-1]).reshape((joints, n, 2))
    L = a[-1]
    return FourierSeriesAgent(coefs, L=L, kp=kp)

