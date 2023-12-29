from typing import List, Tuple
import numpy as np
import matplotlib.pyplot as plt

class FourierSeriesAgent():
    def __init__(
        self,
        coefs: np.ndarray,
    ):
        assert(len(coefs.shape) == 3)
        assert(coefs.shape[2] == 2)

        self.coefs = np.array(coefs)
        self.joints = len(coefs)
        self.n = len(coefs[0])

        self.max_vals = np.zeros(shape=coefs.shape[0])
        for i in range(len(coefs)):
            for p in coefs[i]:
                self.max_vals[i] += p[0]**2 + p[1]**2
            self.max_vals[i] = np.sqrt(self.max_vals[i])
        #print(self.max_vals)

    def sample(self, t: float, L: float = 1, deriv: bool = True, norm: bool = True) -> np.ndarray:
        res = np.zeros((self.joints))
        for j in range(self.n):
            if(deriv):
                res += j * -self.coefs[:, j, 0] * np.sin(j*t/L)
                res += j * self.coefs[:, j, 1] * np.cos(j*t/L)
            else:
                res += self.coefs[:, j, 0] * np.cos(j*t/L)
                res += self.coefs[:, j, 1] * np.sin(j*t/L)

        if(norm):
            return res / self.max_vals
        return res


