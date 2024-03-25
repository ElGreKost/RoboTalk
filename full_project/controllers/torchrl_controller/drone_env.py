import math

import numpy as np
from deepbots.supervisor import RobotSupervisorEnv
from gym.spaces import Box, Discrete
from scipy.stats import multivariate_normal


class Drone(RobotSupervisorEnv):
    def __init__(self):
        super().__init__()
        # Define agent's observation space using Gym's Box, setting the lowest and highest possible values
        self.observation_space = Box(low=np.array([0.0, 0.0, -1.0]),
                                     # Note the additional -1.0 for the action
                                     high=np.array([1.0, 1.0, 4.0]),
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
        self.last_action = -1  # -1 for non-existing
        self.goal_pos = (0, 1)  # top of the circle
        self.mean_reward_per_step = 0

    def polar_distance_to_goal(self):
        """
        Calculates the change in radius (d_r) and change in angle (d_theta) between two positions in polar coordinates.
        Returns:
            A tuple containing d_r (change in radius) and d_theta (change in angle) as floats.
        """
        curr_x, curr_z = self.get_drone_pos()
        goal_x, goal_z = self.goal_pos

        # Calculate current radius and angle
        curr_radius = math.sqrt(curr_x ** 2 + curr_z ** 2)  # Consider z for 3D space
        curr_angle = math.atan2(curr_x, curr_z)  # pitch

        # Calculate goal radius and angle
        goal_radius = math.sqrt(goal_x ** 2 + goal_z ** 2)  # Consider z for 3D space
        goal_angle = math.atan2(goal_x, goal_z)  # Adjust for quadrant handling

        # Calculate change in radius (d_r)
        d_r = goal_radius - curr_radius

        # Calculate change in angle (d_theta)
        # Handle potential angle wrapping around the circle (e.g., -pi to pi)
        d_theta = goal_angle - curr_angle
        d_theta = (d_theta + math.pi) % (2 * math.pi) - math.pi  # Normalize to -pi to pi range

        return d_r, d_theta

    import numpy as np

    def calc_vel_to_goal(self) -> float:
        curr_pos_vec = np.array(self.get_drone_pos())
        goal_pos_vec = np.array(self.goal_pos)
        drone_vel_vec = np.array(self.get_drone_vel())

        # Vector from current position to goal
        direction_vector = goal_pos_vec - curr_pos_vec

        # Projection of drone's velocity on the direction towards the goal
        proj_on_goal = drone_vel_vec @ direction_vector / np.linalg.norm(direction_vector)

        proj_on_goal_norm = np.linalg.norm(proj_on_goal)

        return proj_on_goal_norm

    def get_drone_pos(self) -> tuple:
        drone_x = self.robot.getPosition()[0]
        drone_z = self.robot.getPosition()[2]
        current_pos = (drone_x, drone_z)
        return current_pos

    def get_drone_vel(self) -> tuple:
        drone_vx = self.robot.getVelocity()[0]
        drone_vz = self.robot.getVelocity()[2]
        current_vel = (drone_vx, drone_vz)
        return current_vel

    def get_observations(self) -> tuple:
        # called after every time after a step to return the result
        # Position on z-axis
        # drone_position = normalize_to_range(self.robot.getPosition()[2], 0, 2, -1.0, 1.0)
        d_r, d_theta = self.polar_distance_to_goal()
        # drone_pitch = self.imu.getRollPitchYaw()[1]

        return d_r, d_theta, self.last_action

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

    def multivariate_normal_weighted(self, mean=(0, 0), cov=None, weight_x=1.0, weight_y=1.0):
        return multivariate_normal(mean=mean, cov=cov if cov is not None else ([weight_x ** 2, 0], [0, weight_y ** 2])).pdf

    def get_reward(self, action=None) -> float:
        # keep_alive_reward = 1.0
        # reward = keep_alive_reward + self.calculate_velocity_projection_norm_towards_goal()  # + straight_line_penalty
        d_r, d_theta, last_action = self.get_observations()

        # Doesn't get way bigger than 2
        v = np.linalg.norm(self.get_drone_vel())
        # w = self.get_drone_

        # reward = (v - c * abs(w)) * np.cos(d_theta)  # - v_max
        # if d_r < 0.07:
        #     goal_dist_reward = +10
        # elif 0.07 < d_r < 1:
        #     # doesn't get way bigger than 2
        #     goal_dist_reward = 10 * norm.pdf(d_r, scale=0.4)  # - v_max
        # else:  # impossible ...
        #     goal_dist_reward = -10

        # todo add something for stability in actions and more continuous ones

        goal_dist_reward = self.linear_eucl_distance_reward(np.array((0, d_r)), np.array((0, 0))) * np.cos(d_theta)

        return goal_dist_reward + 2

    def is_done(self):
        if self.episode_score > 40000:
            return True

        z = self.robot.getPosition()[2]
        x = self.robot.getPosition()[0]

        if abs(z) > 3:
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
