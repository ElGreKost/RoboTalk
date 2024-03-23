from deepbots.supervisor.controllers.robot_supervisor_env import RobotSupervisorEnv
from utilities import normalize_to_range
from PPO_agent import PPOAgent, Transition

from gym.spaces import Box, Discrete
import numpy as np
import random
import math


class Drone(RobotSupervisorEnv):
    def __init__(self):
        super().__init__()
        # Define agent's observation space using Gym's Box, setting the lowest and highest possible values
        self.observation_space = Box(low=np.array([0.0, -(math.pi / 6), -1.0, -np.inf, -np.inf, -1.0]),
                                     # Note the additional -1.0 for the action
                                     high=np.array([2.0, (math.pi / 6), 1.0, +np.inf, +np.inf, 4.0]),
                                     # And 4.0 here representing the maximum action index
                                     dtype=np.float64)
        self.action_space = Discrete(5)

        self.robot = self.getSelf()  # Grab the robot reference from the supervisor to access various robot methods
        self.imu = self.getDevice("imu")
        self.imu.enable(self.timestep)

        # self.pole_endpoint = self.getFromDef("POLE_ENDPOINT")
        self.motors = []
        for motor_name in ["motorRF", "motorLF", "motorRB", "motorLB"]:
            motor = self.getDevice(motor_name)  # Get the motor handle
            motor.setPosition(float('inf'))  # Set starting position
            motor.setVelocity(0)  # Zero out starting velocity
            self.motors.append(motor)
        self.steps_per_episode = 1000  # Max number of steps per episode
        self.episode_score = 0  # Score accumulated during an episode
        self.episode_score_list = []  # A list to save all the episode scores, used to check if task is solved
        self.step_counter = 0
        self.living_penalty = 0
        self.avg_pos = (0, 0)
        self.last_action = -1  # -1 for non existing

    def get_observations(self):
        # called after every time after step to return the results
        # Position on z-axis
        # drone_position = normalize_to_range(self.robot.getPosition()[2], 0, 2, -1.0, 1.0)
        drone_z = self.robot.getPosition()[2]
        drone_x = self.robot.getPosition()[0]
        drone_vz = self.robot.getVelocity()[2]
        drone_vx = self.robot.getVelocity()[0]
        drone_pitch = self.imu.getRollPitchYaw()[1]

        return [drone_z, drone_pitch, drone_x, drone_vz, drone_vx, self.last_action]

    def get_default_observation(self):
        # Called every time after env.reset()
        print(f'total:  * s: {self.step_counter}')
        self.living_penalty = 0
        self.step_counter = 0
        self.episode_score = 0

        # This method just returns a zero vector as a default observation
        return [0.0 for _ in range(self.observation_space.shape[0])]

    def gaussian_eucl_distance_reward(self, desired_pos, current_pos, multiplier=10, scale=1.0, width_factor=0.5,
                                      height_factor=2.0):
        """Calculates Gaussian distance reward with adjustable width and height factors."""
        goal_distance = np.linalg.norm([current_pos[0] / width_factor, current_pos[1] / height_factor] - desired_pos)
        gauss_dist = lambda x: np.exp(-(x ** 2) / scale)

        if goal_distance < 0.05:
            reward = 10 * multiplier
        elif goal_distance > 0.9:
            reward = - multiplier * goal_distance
        else:
            reward = multiplier * gauss_dist(goal_distance)

        # reward = 3 * multiplier if goal_distance < 0.1 else multiplier * gauss_dist(goal_distance)
        return reward

    def linear_eucl_distance_reward(self, desired_pos, current_pos, max_rew=100, scale=20):
        goal_distance = np.linalg.norm(current_pos - desired_pos)
        if goal_distance < 0.1:
            reward = max_rew
        else:
            reward = max_rew / 5 - scale * goal_distance

        return reward

    def get_reward(self, action=None):
        self.living_penalty = - self.step_counter / 10  # to decrease exploration as time passes

        z, pitch, x, vz, vx, last_action = self.get_observations()
        circle_top = np.array([0, 1.1])  # referring to x: horizontal, z: vertical
        circle_bottom = np.array([0, 1])
        circle_center = (circle_top + circle_bottom) / 2

        desired_pos = circle_bottom
        current_pos = np.array([x, z])

        # goal_distance_reward = self.gaussian_distance_reward(
        #     current_pos, desired_pos, width_factor=0.4, height_factor=1.2, scale=0.4, multiplier=15)

        goal_distance_reward = self.linear_eucl_distance_reward(desired_pos, current_pos)

        straight_line_penalty = self.gaussian_eucl_distance_reward(current_pos, circle_center, scale=0.1, multiplier=4)

        living_penalty = -2

        # self.living_penalty += 1

        reward = goal_distance_reward + living_penalty  # + straight_line_penalty
        return reward

    def is_done(self):
        if self.episode_score > 40000:
            return True

        z, pitch, x, vz, vx, last_action = self.get_observations()  # Position on x-axis
        if abs(z) > 4:
            return True
        if abs(x) > 3:
            return True
        return False

    def solved(self):
        if len(self.episode_score_list) > 100:  # Over 100 trials thus far
            if np.mean(self.episode_score_list[-100:]) > 35000:  # Last 100 episodes' scores average value
                return True
        return False

    def get_info(self):
        # print("Motor Speed: {:.3f}r/s {:.3f}r/s {:.3f}r/s {:.3f}r/s".format(self.motors[0].getVelocity(), self.motors[1].getVelocity(), self.motors[2].getVelocity(), self.motors[3].getVelocity()))
        return None

    def render(self, mode='human'):
        pass

    def apply_action(self, action):
        self.step_counter += 1

        action = int(action[0])

        self.last_action = action

        pitch = self.get_observations()[1]
        c_pitch = math.cos(pitch)
        w_hover = math.sqrt((9.81 * 2.5) / (4 * 1.96503e-05 * c_pitch)) + 1
        height_step = 7  # original 10
        w_up = w_hover + height_step
        w_down = w_hover - height_step
        angle_step = 0.5  # original 2
        if action == 0:
            motor_speed = w_up
        elif action == 1:
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
            self.motors[0].setVelocity(w_hover + angle_step)
            self.motors[1].setVelocity(w_hover + angle_step)
            self.motors[2].setVelocity(w_hover - angle_step)
            self.motors[3].setVelocity(w_hover - angle_step)
        elif action == 4:
            for i in range(len(self.motors)):
                self.motors[i].setPosition(float('inf'))
            self.motors[0].setVelocity(w_hover - angle_step)
            self.motors[1].setVelocity(w_hover - angle_step)
            self.motors[2].setVelocity(w_hover + angle_step)
            self.motors[3].setVelocity(w_hover + angle_step)


env = Drone()

agent = PPOAgent(number_of_inputs=env.observation_space.shape[0], number_of_actor_outputs=env.action_space.n)
# agent.load_weights('Episode300PlusV1.pth')
solved = False
episode_count = 0
episode_limit = 10000
# Run outer loop until the episodes limit is reached or the task is solved
while not solved and episode_count < episode_limit:
    observation = env.reset()  # Reset robot and get starting observation

    motor_speed = 558.586 + random.randint(0, 4) * 10
    # motor_speed = 1000

    # env.episode_score = 0.0
    # observation = env.reset()
    # pre-trained run
    # agent_0 = PPOAgent(number_of_inputs=env.observation_space.shape[0], number_of_actor_outputs=env.action_space.n)
    # agent_0.load_weights('Trained_Lift_Off.pth')
    # while True:
    #     selected_action, action_prob = agent_0.work(observation, type_="selectActionMax")
    #     observation, _, done, _ = env.step([selected_action])
    #     if done:
    #         observation = env.reset()
    #     if env.robot.getVelocity()[2] < 0.1 and abs(env.robot.getPosition()[2] - 1) < 0.05:
    #         print("Deploying Training")
    #         break

    for i in range(len(env.motors)):
        env.motors[i].setPosition(float('inf'))
        env.motors[i].setVelocity(motor_speed)

    env.episode_score = 0

    time_counter = 0

    for step in range(2 * env.steps_per_episode):
        time_counter += 1
        # In training mode, the agent samples from the probability distribution, naturally implementing exploration
        selected_action, action_prob = agent.work(observation, type_="selectAction")
        # print(action_prob)
        # Step the supervisor to get the current selected_action's reward, the new observation and whether we reached
        # the done condition
        new_observation, reward, done, info = env.step([selected_action])

        # Save the current state transition in agent's memory
        trans = Transition(observation, selected_action, action_prob, reward, new_observation)
        agent.store_transition(trans)

        if done:
            # Save the episode's score
            env.episode_score_list.append(env.episode_score)
            # why does the batch-size change dynamically? The better the performance, the more important the past?
            agent.train_step(batch_size=step + 1)
            solved = env.solved()  # Check whether the task is solved
            break

        env.episode_score += reward  # Accumulate episode reward
        observation = new_observation  # observation for next step is current step's new_observation
    print("Episode #", episode_count, "score:", env.episode_score)
    episode_count += 1  # Increment episode counter

    if episode_count > 300:
        agent.save_weights('Episode300PlusV1.pth')

    # print(time_counter)

if not solved:
    print("Task is not solved, deploying agent for testing...")
elif solved:
    print("Task is solved, deploying agent for testing...")
    # agent.save_weights('Trained_Lift_Off.pth')

observation = env.reset()
env.episode_score = 0.0
while True:
    selected_action, action_prob = agent.work(observation, type_="selectActionMax")
    observation, _, done, _ = env.step([selected_action])
    if done:
        observation = env.reset()
