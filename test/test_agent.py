import sys
import gymnasium as gym
import numpy as np
import util.fourier_series_agent

def test_agent(agent):
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

        wanted_state = agent[0].sample(i, deriv=False, norm=False)
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

if __name__ == "__main__":
    agent_path = sys.argv[1]
    print(agent_path)
    agent = np.loadtxt(agent_path)
    print(agent)
    test_agent(fourier_series_agent.from_array(agent))
