import numpy as np
import matplotlib.pyplot as plt
from util.fourier_series_agent import FourierSeriesAgent
import pybullet_envs_gymnasium
import gymnasium as gym

env = gym.make("Ant-v4", render_mode="human", reset_noise_scale=0)
action_size = env.action_space.shape[0]

#wave = [[0, 0], [0, 1], [0, 0], [0, 0.33], [0, 0], [0, 0.2], [0, 0]]
#wave = [[0, 0], [0, 1], [0, 0], [0, 0.33], [0, 0], [0, 0.2], [0, 0]]
#wave = [[0, 0], [0.1, 0.1]]
wave = [[0, 0], [1, 1]]
joints = [wave for i in range(action_size)]

agent = FourierSeriesAgent(np.array(joints))

t = 0
kp = 0.1
total_reward = 0
obs, _ = env.reset()

times = []
states = []
wanted_states = []
actions = []

for i in range(1000):
    # from documentation: action = hip_4, angle_4, hip_1, angle_1, hip_2, angle_2, hip_3, angle_3
    # observation: hip_4=11, ankle_4=12, hip_1=5, ankle_1=6, hip_2=7, ankle_2=8 hip_3=9, ankle_3=10
    joint_state = np.array([obs[11], obs[12], obs[5], obs[6], obs[7], obs[8], obs[9], obs[10]]) 
    print(joint_state)

    wanted_state = 0.25 * agent.sample(i, deriv=False, norm=False)
    wanted_state += np.array([0, 1, 0, 1, 0, 1, 0, 1]) #manually change center of mass for some of htem lmao
   
    action = kp * (wanted_state-joint_state)
    obs, reward, _, _, _ = env.step(action)
    total_reward += reward
    t += 1

    states.append(joint_state)
    wanted_states.append(wanted_state)
    times.append(i)
    actions.append(action)


times, states = np.array(times), np.array(states)
wanted_states = np.array(wanted_states)
actions = np.array(actions)
print(actions * 1/kp) 
for i in range(2):
    plt.plot(times, states[:, i])
    plt.plot(times, wanted_states[:, i])
    plt.plot(times, actions[:, i] * 1/kp)
    plt.show()
