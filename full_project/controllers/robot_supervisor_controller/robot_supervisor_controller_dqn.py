from deepbots.supervisor.controllers.robot_supervisor_env import RobotSupervisorEnv
# from kostis_controller.Nstep_DQN import NstepAgent
from utilities import normalize_to_range
from PPO_agent import PPOAgent, Transition
import torch as th
from Nstep_DQN import NstepAgent

from gym.spaces import Box, Discrete
import numpy as np
import random
import math

class Drone(RobotSupervisorEnv):
    def __init__(self):
        super().__init__()
        # Define agent's observation space using Gym's Box, setting the lowest and highest possible values
        self.observation_space = Box(low=np.array([0.0, -(math.pi/6), -1.0, -np.inf]),
                                     high=np.array([2.0, (math.pi/6), 1.0, +np.inf]),
                                     dtype=np.float64)
        # Define agent's action space using Gym's Discrete
        self.action_space = Discrete(5)

        self.robot = self.getSelf()  # Grab the robot reference from the supervisor to access various robot methods
        self.imu = self.getDevice("imu")
        self.imu.enable(self.timestep)

        #self.pole_endpoint = self.getFromDef("POLE_ENDPOINT")
        self.motors = []
        for motor_name in ["motorRF",  "motorLF", "motorRB", "motorLB"]:
            motor = self.getDevice(motor_name)  # Get the motor handle
            motor.setPosition(float('inf'))  # Set starting position
            motor.setVelocity(0)  # Zero out starting velocity
            self.motors.append(motor)
        self.steps_per_episode = 1000  # Max number of steps per episode
        self.episode_score = 0  # Score accumulated during an episode
        self.episode_score_list = []  # A list to save all the episode scores, used to check if task is solved

    def get_observations(self):
        # Position on z-axis
        #drone_position = normalize_to_range(self.robot.getPosition()[2], 0, 2, -1.0, 1.0)
        drone_z = self.robot.getPosition()[2]
        drone_x = self.robot.getPosition()[0]
        drone_vz = self.robot.getVelocity()[2]
        drone_pitch = self.imu.getRollPitchYaw()[1]

        return [drone_z, drone_pitch, drone_x, drone_vz]

    def get_default_observation(self):
        # This method just returns a zero vector as a default observation
        return [0.0 for _ in range(self.observation_space.shape[0])]

    def get_reward(self, action=None):
        z, pitch, x, vz = self.get_observations()
        reward = 0
        #x_factor = 5-30*(x**2)
        #pitch_factor = 1-30*(pitch**2)+x_factor
        '''if(abs(z-1) < 0.1):
            reward = +100
        elif z < 0.99:
            reward = 15+20*(z-1)
        elif z > 1.01:
            reward = 15-20*(z-1)
        return reward'''
    
        '''if(abs(z-1) < 0.1):
            reward = +100
        elif z < 0.99:
            reward = 15+20*(z-1)
        elif z > 1.01:
            reward = 15-20*(z-1)
        return reward'''
        return 10 * np.exp(-50 * (1 - z)**2 - 50 * (vz)**2) - np.exp(-50 * z**2) - 0.1 * (1 - z)**2 - 0.1 * (vz)**2

    def is_done(self):
        if self.episode_score > 8000:
            return True

        z, pitch, x, vz = self.get_observations()  # Position on x-axis
        if abs(z) > 2:
            return True
        if abs(x) > 2:  
            return True
        return False

    def solved(self):
        if len(self.episode_score_list) > 100:  # Over 100 trials thus far
            if np.mean(self.episode_score_list[-100:]) > 7000:  # Last 100 episodes' scores average value
                return True
        return False

    def get_info(self):
        #print("Motor Speed: {:.3f}r/s {:.3f}r/s {:.3f}r/s {:.3f}r/s".format(self.motors[0].getVelocity(), self.motors[1].getVelocity(), self.motors[2].getVelocity(), self.motors[3].getVelocity()))
        return None

    def render(self, mode='human'):
        pass

    def apply_action(self, action):
        action = int(action[0])
        pitch = self.get_observations()[1]
        c_pitch = math.cos(pitch)
        w_hover = math.sqrt((9.81*2.5)/(4*1.96503e-05*c_pitch))+1
        w_up = w_hover + 10
        w_down = w_hover - 10
        dw = 2
        #print("Time counter: ", time_counter)
        if action == 0:
            #motor_speed = self.motors[0].getVelocity() + 10
            motor_speed = w_up
        elif action == 1:
            #motor_speed = self.motors[0].getVelocity() - 10
            motor_speed = w_down
        elif action == 2:
            motor_speed = w_hover
        
        if action < 3:
            for i in range(len(self.motors)):
                self.motors[i].setPosition(float('inf'))
                self.motors[i].setVelocity(motor_speed)
        elif action == 3:
            for i in range(len(self.motors)):
                self.motors[i].setPosition(float('inf'))
            self.motors[0].setVelocity(w_hover+dw)
            self.motors[1].setVelocity(w_hover+dw)
            self.motors[2].setVelocity(w_hover-dw)
            self.motors[3].setVelocity(w_hover-dw)
        elif action == 4:
            for i in range(len(self.motors)):
                self.motors[i].setPosition(float('inf'))
            self.motors[0].setVelocity(w_hover-dw)
            self.motors[1].setVelocity(w_hover-dw)
            self.motors[2].setVelocity(w_hover+dw)
            self.motors[3].setVelocity(w_hover+dw)


import torch as th
import random

# Assume NstepAgent and Drone environment are correctly imported

env = Drone()  # Initialize the Drone environment
agent = NstepAgent(env, use_conv=False)  # Initialize the NstepAgent with parameters suited for the Drone

episode_rewards = []  # To track rewards for each episode
max_episodes = 10000
max_steps = env.steps_per_episode  # Assuming `steps_per_episode` is defined in your Drone environment

for episode in range(max_episodes):
    episode_reward = 0
    observation = env.reset()
    observation = th.tensor(observation, dtype=th.float32).to(agent.device)

    motor_speed = 558.586 + random.randint(0, 4) * 10
    for i in range(len(env.motors)):
        env.motors[i].setPosition(float('inf'))
        env.motors[i].setVelocity(motor_speed)

    for step in range(max_steps):
        action = agent.get_action(observation)
        new_observation, reward, done, _ = env.step([action])  # Ensure correct action format for Drone
        new_observation = th.tensor(new_observation, dtype=th.float32).to(agent.device)

        # Update agent with the transition. Note: 'done' acts as both 'term' and 'trunc' in this context
        agent.update(observation, action, reward, new_observation, done)

        episode_reward += reward
        if done:
            episode_rewards.append(episode_reward)
            print(f"Episode {episode}: {episode_reward}")
            agent.finish_nstep()  # Process remaining transitions in n-step buffer
            break  # Exit the loop if the episode is done

        observation = new_observation  # Update state for the next step

    # Optionally, check if the task is solved to break the loop, if applicable to your Drone environment

print(f"Task completed over {len(episode_rewards)} episodes")

# Plot the episode rewards
# plt.plot(episode_rewards)
