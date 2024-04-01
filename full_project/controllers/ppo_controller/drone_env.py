from deepbots.supervisor.controllers.robot_supervisor_env import RobotSupervisorEnv
from gym.spaces import Box, Discrete
import numpy as np

class Drone(RobotSupervisorEnv):
    def __init__(self):
        super().__init__()
        # Define agent's observation space using Gym's Box, setting the lowest and highest possible values
        self.observation_space = Box(low=np.array([0.]),
                                     high=np.array([2.]),
                                     dtype=np.float64)
        # Define agent's action space using Gym's Discrete
        self.action_space = Discrete(3)

        self.robot = self.getSelf()  # Grab the robot reference from the supervisor to access various robot methods
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
        drone_position = self.robot.getPosition()[2]

        return [drone_position]

    def get_default_observation(self):
        # This method just returns a zero vector as a default observation
        return [0.0 for _ in range(self.observation_space.shape[0])]

    def get_reward(self, action=None):
        z = self.get_observations()[0]

        # Run this on the pretrained
        if(abs(z-1) < 0.03):
            return 100
        else:
            return 0

        # How to unlearn
        # if(abs(z-1) < 0.01):
        #     return -30
        # else:
        #     return -15+20*abs(z-1)

        # How to pretrain
        # if(abs(z-1) < 0.01):
        #     return +30
        # else:
        #     return +15-20*abs(z-1)

    def is_done(self):
        if self.episode_score > 12000:
            return True

        drone_position = self.get_observations()[0]  # Position on x-axis
        if abs(drone_position) > 2:
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
        if action == 0:
            motor_speed = 558.586 + 10
        elif action == 1:
            motor_speed = 558.586 - 10
        else:
            motor_speed = 558.586

        for i in range(len(self.motors)):
            self.motors[i].setPosition(float('inf'))
            self.motors[i].setVelocity(motor_speed)