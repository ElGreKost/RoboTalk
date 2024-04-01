# Name: Drone
# Version: 4.1
# Authors: - Dimitrios papageorgiou
#
# Drone environment library set up for robot-imitation


# -= Importing needed Libraries =- #

# System libraries
from deepbots.supervisor.controllers.robot_supervisor_env import RobotSupervisorEnv
from gym.spaces import Box, Discrete
from controller import Supervisor

# Python Libraries
import numpy as np
import math

class Drone(RobotSupervisorEnv):
    def __init__(self):
        super().__init__()
        # Define drone's observation space with upper and lower limits
        self.observation_space = Box(low=np.array([-4, -(math.pi / 6), -3.0, -np.inf]),
                                     high=np.array([4, (math.pi / 6), 3.0, +np.inf]),
                                     dtype=np.float64)
        # Define drone's action space
        self.action_space = Discrete(7)

        # Initialize robot and sensors
        self.robot = self.getSelf()
        self.imu = self.getDevice("imu")
        self.imu.enable(self.timestep)

        # Initialize motors
        self.motors = []
        for motor_name in ["motorRF", "motorLF", "motorRB", "motorLB"]:
            motor = self.getDevice(motor_name)
            motor.setPosition(float('inf'))
            motor.setVelocity(558.586)
            self.motors.append(motor)

        # Other system variables
        self.steps_per_episode = 1000
        self.episode_score = 0
        self.episode_score_list = []
        self.last_action = 2
        self.goalx = 1
        self.goalz = 1

    def get_observations(self):

        # All the needed inputs to determine the state of the drone
        # Also the inputs for the neural network

        drone_z = self.robot.getPosition()[2]        # Z position
        drone_x = self.robot.getPosition()[0]        # X position
        drone_vz = self.robot.getVelocity()[2]       # Z velocity
        drone_pitch = self.imu.getRollPitchYaw()[1]  # Pitch

        # [Error_z, drone_pitch, Error_x, drone_vz]
        return [drone_z - self.goalz, drone_pitch, drone_x - self.goalx, drone_vz]

    def get_default_observation(self):
        # This method just returns a zero vector as a default observation
        return [0.0 for _ in range(self.observation_space.shape[0])]

    def get_reward(self, action=None):

        # Reward function according to state

        z, pitch, x, vz = self.get_observations()

        if abs(z) < 0.05:
            re_z = 10
        else:
            re_z = 10 + 20 * (0.05 - abs(z))

        if abs(x) < 0.05:
            re_x = 15
        else:
            re_x = 10 + 20 * (0.05 - abs(x))

        return (re_z + re_x)/2

    def is_done(self):

        # Returns true if any of the termination criteria has been met

        if self.episode_score > 8000:
            return True

        z, pitch, x, vz = self.get_observations()
        if abs(z) > 4:
            return True
        if abs(x) > 4:
            return True
        return False

    def solved(self, episodes):

        # Returns true after the imitation phase once the mean reward value of 500 consecutive episodes is over 7000

        if episodes > 2000:
            if len(self.episode_score_list) > 500:  # Over 100 trials thus far
                if np.mean(self.episode_score_list[-500:]) > 7000:  # Last 100 episodes' scores average value
                    return True
        return False

    def get_info(self):
        # print("Motor Speed: {:.3f}r/s {:.3f}r/s {:.3f}r/s {:.3f}r/s".format(self.motors[0].getVelocity(), self.motors[1].getVelocity(), self.motors[2].getVelocity(), self.motors[3].getVelocity()))
        return None

    def render(self, mode='human'):
        pass

    def step(self, action, goalz, goalx):

        # Next time step of drone and environment

        self.apply_action(action)
        self.goalx = goalx
        self.goalz = goalz
        if super(Supervisor, self).step(self.timestep) == -1:
            exit()

        return self.get_observations(), self.get_reward(action), self.is_done(), self.get_info(),

    def apply_action(self, action):

        # Action selector method
        # 0-4 actions for vertical motion
        # 5-6 actions for horizontal motion

        action = int(action[0])

        self.last_action = action

        pitch = self.get_observations()[1]
        c_pitch = math.cos(pitch)
        w_hover = math.sqrt((9.81 * 2.5) / (4 * 1.96503e-05 * c_pitch))
        dw = 2

        if action < 5:
            motor_speed = w_hover + action * 40/4 - 20
            for i in range(len(self.motors)):
                self.motors[i].setPosition(float('inf'))
                self.motors[i].setVelocity(motor_speed)
        else:
            if action == 5:
                for i in range(len(self.motors)):
                    self.motors[i].setPosition(float('inf'))
                self.motors[0].setVelocity(w_hover + dw)
                self.motors[1].setVelocity(w_hover + dw)
                self.motors[2].setVelocity(w_hover - dw)
                self.motors[3].setVelocity(w_hover - dw)
            else:
                for i in range(len(self.motors)):
                    self.motors[i].setPosition(float('inf'))
                self.motors[0].setVelocity(w_hover - dw)
                self.motors[1].setVelocity(w_hover - dw)
                self.motors[2].setVelocity(w_hover + dw)
                self.motors[3].setVelocity(w_hover + dw)