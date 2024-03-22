from deepbots.supervisor.controllers.robot_supervisor_env import RobotSupervisorEnv
from utilities import normalize_to_range
from PPO_agent import PPOAgent, Transition

from gymnasium.spaces import Box, Discrete
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
            motor.setVelocity(560)  # Zero out starting velocity
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
        x_factor = 5-30*(x**2)
        pitch_factor = 1-30*(pitch**2)+x_factor
        if(abs(z-1) < 0.01):
            reward = 100+pitch_factor+x_factor
        elif z < 0.99:
            reward = 15+20*(z-1)
        elif z > 1.01:
            reward = 15-20*(z-1)
        return reward
    
        #return 10 * np.exp(-50 * (1 - z)**2 - 50 * (vz)**2) - np.exp(-50 * z**2) - 0.1 * (1 - z)**2 - 0.1 * (vz)**2

    def is_done(self):
        if self.episode_score > 6000:
            return True

        z, pitch, x, vz = self.get_observations()  # Position on x-axis
        if abs(z) > 2:
            return True
        if abs(x) > 2:  
            return True
        return False

    def solved(self):
        if len(self.episode_score_list) > 100:  # Over 100 trials thus far
            if np.mean(self.episode_score_list[-100:]) > 6000:  # Last 100 episodes' scores average value
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

env = Drone()

# motor_speed = 558.586
#
# for i in range(len(env.motors)):
#     env.motors[i].setPosition(float('inf'))
#     env.motors[i].setVelocity(motor_speed)

agent = PPOAgent(number_of_inputs=env.observation_space.shape[0], number_of_actor_outputs=env.action_space.n)

solved = False
episode_count = 0
episode_limit = 10000


# Run outer loop until the episodes limit is reached or the task is solved
while not solved and episode_count < episode_limit:
    observation = env.reset()  # Reset robot and get starting observation

    motor_speed = 558.586 + random.randint(0, 4) * 10

    for i in range(len(env.motors)):
        env.motors[i].setPosition(float('inf'))
        env.motors[i].setVelocity(motor_speed)

    env.episode_score = 0

    time_counter = 0

    for step in range(env.steps_per_episode):

        time_counter += 1

        # In training mode the agent samples from the probability distribution, naturally implementing exploration
        selected_action, action_prob = agent.work(observation, type_="selectAction")
        # Step the supervisor to get the current selected_action's reward, the new observation and whether we reached
        # the done condition
        new_observation, reward, done, info = env.step([selected_action])

        # Save the current state transition in agent's memory
        trans = Transition(observation, selected_action, action_prob, reward, new_observation)
        agent.store_transition(trans)

        if done:
            # Save the episode's score
            env.episode_score_list.append(env.episode_score)
            agent.train_step(batch_size=step + 1)
            solved = env.solved()  # Check whether the task is solved
            break

        env.episode_score += reward  # Accumulate episode reward
        observation = new_observation  # observation for next step is current step's new_observation

    print("Episode #", episode_count, "score:", env.episode_score)
    episode_count += 1  # Increment episode counter

    #print(time_counter)


if not solved:
    print("Task is not solved, deploying agent for testing...")
elif solved:
    print("Task is solved, deploying agent for testing...")

observation = env.reset()
env.episode_score = 0.0
while True:
    selected_action, action_prob = agent.work(observation, type_="selectActionMax")
    observation, _, done, _ = env.step([selected_action])
    if done:
        observation = env.reset()

