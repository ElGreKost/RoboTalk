from deepbots.supervisor import RobotSupervisorEnv
from gymnasium.spaces import Box
from drone_utils import *


class Drone(RobotSupervisorEnv):
    # noinspection PyTypeChecker
    def __init__(self):
        super().__init__()
        # polar_goal_dist_obs_space = Box(
        #     low=np.array([0, -np.pi, -np.inf, -np.inf]), high=np.array([4, np.pi, np.inf, np.inf]), dtype=np.float64)
        eucl_obs_space = Box(low=np.array([-3, 0, -1]), high=np.array([3, 3, 1]), dtype=np.float64)
        self.observation_space = eucl_obs_space
        self.action_space = Box(low=np.array([-1]), high=np.array([1]), dtype=np.float64)
        self.reward_space = Box(low=-np.inf, high=np.inf, shape=())  # Empty shape for single reward value

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
        self.episode_score = 0  # Score accumulated during an episode
        self.episode_score_list = []  # A list to save all the episode scores, used to check if the task is solved
        self.step_counter = 0
        self.last_action = -1  # -1 for non-existing
        self.last_pos = (0, 0)
        self.goal_pos = (0, 1)  # top of the circle

    def get_drone_pos(self) -> tuple:
        drone_x = self.robot.getPosition()[0]
        drone_z = self.robot.getPosition()[2]
        drone_pos = (drone_x, drone_z)
        return drone_pos

    def get_drone_vel(self) -> tuple:
        drone_vx = self.robot.getVelocity()[0]
        drone_vz = self.robot.getVelocity()[2]
        drone_vel = (drone_vx, drone_vz)
        return drone_vel

    def get_observations(self) -> tuple:              # Called after a step
        drone_x, drone_z = self.get_drone_pos()
        # NORMALIZE OBSERVATIONS
        drone_x /= 3
        drone_z /= 3
        last_action_front = self.last_action
        last_drone_x, last_drone_z = self.last_pos
        last_drone_x /= 3
        last_drone_z /= 3

        return drone_x, drone_z, last_action_front

    def get_reward(self, action=None) -> float:     # Called after env.step()
        z = self.robot.getPosition()[2]

        # UNLEARN
        reward = 100 if (np.abs(z - 1) < 0.03) else reward = 0

        self.episode_score += reward
        return reward

    def get_default_observation(self):               # Called after env.reset()
        print('total:  * s:', self.step_counter, ' with reward:', self.episode_score, ' with up_steps {self.up_steps}')
        self.step_counter = 0
        self.episode_score = 0
        self.last_pos = (0, 0)

        # This method just returns a zero vector as a default observation
        return [0.0 for _ in range(self.observation_space.shape[0])]

    def is_done(self):                                # Called after env.step()
        if self.episode_score > 40000:
            return True
        if self.step_counter > 6000:
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
        return None

    def render(self, mode='human'):
        pass

    def apply_action(self, action: float):
        self.step_counter += 1
        self.last_action = action
        self.last_pos = self.get_drone_pos()
        selected_rpm_change = 7 * action

        pitch = self.imu.getRollPitchYaw()[1]
        c_pitch = math.cos(pitch)

        w_hover = math.sqrt((9.81 * 2.5) / (4 * 1.96503e-05 * c_pitch))
        for i in range(len(self.motors)):
            self.motors[i].setPosition(float('inf'))
        self.motors[0].setVelocity(w_hover + selected_rpm_change)
        self.motors[1].setVelocity(w_hover + selected_rpm_change)
        self.motors[2].setVelocity(w_hover + selected_rpm_change)
        self.motors[3].setVelocity(w_hover + selected_rpm_change)
