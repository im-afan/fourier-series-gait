import sys
import gymnasium as gym
import numpy as np
from util.fourier_series_agent import from_array
import matplotlib.pyplot as plt


def vec_test_agent(agent):
    #env = gym.make("Hopper-v4", render_mode="rgb_array", reset_noise_scale=0)
    #env = gym.make("Hopper-v4", reset_noise_scale=0)
    envs = gym.vector.make("Hopper-v4", render_mode="human", num_envs=1, reset_noise_scale=0.01, terminate_when_unhealthy=False)

    t = 0
    total_reward = 0 
    obs, _ = envs.reset()

    states, wanted_states, times, actions, diff = [], [], [], [], []

    for i in range(1000):
        # FOR ANT
        # from documentation: action = hip_4, angle_4, hip_1, angle_1, hip_2, angle_2, hip_3, angle_3
        # observation: hip_4=11, ankle_4=12, hip_1=5, ankle_1=6, hip_2=7, ankle_2=8 hip_3=9, ankle_3=10
        #joint_state = np.array([obs[11], obs[12], obs[5], obs[6], obs[7], obs[8], obs[9], obs[10]]) 
        #joint_state = joint_state.T #magic!

        # FOR HOPPER
        # from documentation: action: thigh_joint, leg_joint, foot_joint
        # observation: thigh_joint=2, leg_joint=3, foot_joint=4
        joint_state = np.array([obs[:, 2], obs[:, 3], obs[:, 4]])
        joint_state = joint_state.T
        print(joint_state)

        wanted_state = np.array([agent.sample(i, deriv=False, norm=False)])
        print(i, wanted_state)
    
        action = agent.kp * (wanted_state-joint_state)
        obs, reward, _, _, _ = envs.step(action)
        total_reward += reward[0]
        t += 1

        #logging for tests
        states.append(joint_state[0])
        wanted_states.append(wanted_state[0])
        times.append(i)
        actions.append(action)
        diff.append((wanted_state-joint_state)[0])

        envs.call("render")
     

    states = np.array(states);
    wanted_states = np.array(wanted_states)
    diff = np.array(diff);
    print("total reward: ", total_reward)
    #plt.plot(times, states)
    plt.plot(times[300:600], states[300:600, 0], label="states")
    plt.plot(times[300:600], diff[300:600, 0], label="diff")
    plt.plot(times[300:600], wanted_states[300:600, 0], label="wanted states")
    plt.legend()
    plt.show()



def test_agent(agent):
    env = gym.make("Hopper-v4", render_mode="rgb_array", reset_noise_scale=0)
    #env = gym.make("Hopper-v4", reset_noise_scale=0)

    t = 0
    total_reward = 0 
    obs, _ = env.reset()

    states, wanted_states, times, actions, diff = [], [], [], [], []

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
        #joint_state = np.flip(joint_state) 
        print(joint_state)

        wanted_state = agent.sample(i, deriv=False, norm=False)
        print(i, wanted_state)
    
        action = agent.kp * (wanted_state-joint_state)
        obs, reward, _, _, _ = env.step(action)
        total_reward += reward
        t += 1

        #logging for tests
        states.append(joint_state)
        wanted_states.append(wanted_state)
        times.append(i)
        actions.append(action)
        diff.append(wanted_state - joint_state)
    
    states = np.array(states);
    wanted_states = np.array(wanted_states)
    diff = np.array(diff);

    print("total reward: ", total_reward)
    #plt.plot(times[:100], states[:100])
    #plt.plot(times, states)
    #plt.plot(times[300:600], states[300:600])
    plt.plot(times[300:600], states[300:600, 0], label="states")
    plt.plot(times[300:600], diff[300:600, 0], label="diff")
    plt.plot(times[300:600], wanted_states[300:600, 0], label="wanted states")
    plt.show()

if __name__ == "__main__":
    env = gym.make("Hopper-v4")
    joints = env.action_space.shape[0]
    agent_path = sys.argv[1]
    print(agent_path)
    agent = np.loadtxt(agent_path)
    print(agent)
    if(sys.argv[2] == "vec"):
        vec_test_agent(from_array(joints=joints, n=3, a=agent, kp=0.01))
    else:
        test_agent(from_array(joints=joints, n=3, a=agent, kp=0.01))
