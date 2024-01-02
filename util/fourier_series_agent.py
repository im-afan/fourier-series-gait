from typing import List, Tuple
import numpy as np
import matplotlib.pyplot as plt

class FourierSeriesAgent():
    def __init__(
        self,
        coefs: np.ndarray,
        L: int = 10
    ):
        assert(len(coefs.shape) == 3)
        assert(coefs.shape[2] == 2)

        self.coefs = np.array(coefs)
        self.joints = len(coefs)
        self.n = len(coefs[0])
        self.L = L

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

    def save(self, coef_checkpoint="saved_agents/best_agent_coef.npy", L_checkpoint="saved_agents/best_agent_L.npy"):
        np.save(coef_checkpoint, self.coefs)
        np.save(L_checkpoint, np.array([self.L]))

    def load(self, coef_checkpoint="saved_agents/best_agent_coef.npy", L_checkpoint="saved_agents/best_agent_L.npy"):
        self.coefs = np.load(coef_checkpoint)
        self.L = np.load(L_checkpoint)

