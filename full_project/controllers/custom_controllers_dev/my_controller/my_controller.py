from typing import Optional

from torchrl.envs import GymWrapper

from torchrl.data import BoundedTensorSpec, CompositeSpec
from torchrl.envs import EnvBase, TransformedEnv, Compose, ObservationNorm, DoubleToFloat, StepCounter, check_env_specs
from tensordict import MemoryMappedTensor, TensorDict
import torch

# from utilities import normalize_to_range
# from PPO_agent import PPOAgent, Transition
from deepbots.supervisor.controllers.robot_supervisor_env import RobotSupervisorEnv

from gymnasium.spaces import Box, Discrete
import numpy as np
import random
import math


class Drone(RobotSupervisorEnv):
    def __init__(self):
        super().__init__()
        # Define agent's observation space using Gym's Box, setting the lowest and highest possible values
        self.observation_space = Box(low=np.array([0.0, -(math.pi / 6), -1.0, -np.inf]),
                                     high=np.array([2.0, (math.pi / 6), 1.0, +np.inf]),
                                     dtype=np.float64)
        # Define agent's action space using Gym's Discrete
        self.action_space = Discrete(5)

        self.robot = self.getSelf()  # Grab the robot reference from the supervisor to access various robot methods
        self.imu = self.getDevice("imu")
        self.imu.enable(self.timestep)

        # self.pole_endpoint = self.getFromDef("POLE_ENDPOINT")
        self.motors = []
        for motor_name in ["motorRF", "motorLF", "motorRB", "motorLB"]:
            motor = self.getDevice(motor_name)  # Get the motor handle
            motor.setPosition(float('inf'))  # Set starting position
            motor.setVelocity(560)  # Zero out starting velocity
            self.motors.append(motor)
        self.steps_per_episode = 1000  # Max number of steps per episode
        self.episode_score = 0  # Score accumulated during an episode
        self.episode_score_list = []  # A list to save all the episode scores, used to check if task is solved

    def get_observations(self):
        # Position on z-axis
        # drone_position = normalize_to_range(self.robot.getPosition()[2], 0, 2, -1.0, 1.0)
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
        x_factor = 5 - 30 * (x ** 2)
        pitch_factor = 1 - 30 * (pitch ** 2) + x_factor
        if (abs(z - 1) < 0.01):
            reward = 100 + pitch_factor + x_factor
        elif z < 0.99:
            reward = 15 + 20 * (z - 1)
        elif z > 1.01:
            reward = 15 - 20 * (z - 1)
        return reward

        # return 10 * np.exp(-50 * (1 - z)**2 - 50 * (vz)**2) - np.exp(-50 * z**2) - 0.1 * (1 - z)**2 - 0.1 * (vz)**2

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
        # print("Motor Speed: {:.3f}r/s {:.3f}r/s {:.3f}r/s {:.3f}r/s".format(self.motors[0].getVelocity(), self.motors[1].getVelocity(), self.motors[2].getVelocity(), self.motors[3].getVelocity()))
        return None

    def render(self, mode='human'):
        pass

    def apply_action(self, action):
        action = int(action[0])
        pitch = self.get_observations()[1]
        c_pitch = math.cos(pitch)
        w_hover = math.sqrt((9.81 * 2.5) / (4 * 1.96503e-05 * c_pitch)) + 1
        w_up = w_hover + 10
        w_down = w_hover - 10
        dw = 2
        if action == 0:
            # motor_speed = self.motors[0].getVelocity() + 10
            motor_speed = w_up
        elif action == 1:
            # motor_speed = self.motors[0].getVelocity() - 10
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
            self.motors[0].setVelocity(w_hover + dw)
            self.motors[1].setVelocity(w_hover + dw)
            self.motors[2].setVelocity(w_hover - dw)
            self.motors[3].setVelocity(w_hover - dw)
        elif action == 4:
            for i in range(len(self.motors)):
                self.motors[i].setPosition(float('inf'))
            self.motors[0].setVelocity(w_hover - dw)
            self.motors[1].setVelocity(w_hover - dw)
            self.motors[2].setVelocity(w_hover + dw)
            self.motors[3].setVelocity(w_hover + dw)


class DroneEnvWrapper(EnvBase):
    def __init__(self, drone_env, device=None):
        super().__init__(device=device)
        self.drone_env = drone_env  # Your Drone environment instance
        self.observation_space = drone_env.observation_space
        self.action_space = drone_env.action_space
        self._make_specs()  # Convert gym spaces to torchrl composites

    def _convert_space_to_spec(self, space):
        if isinstance(space, Box):
            low = torch.tensor(space.low, dtype=torch.float32)
            high = torch.tensor(space.high, dtype=torch.float32)
            # For a Box space, we return a bounded tensor spec directly
            return BoundedTensorSpec(shape=low.shape, low=low, high=high, dtype=torch.float32)
        elif isinstance(space, Discrete):
            # For Discrete spaces, we create a bounded tensor spec with bounds from 0 to n-1
            n = space.n
            low = torch.tensor(0, dtype=torch.int64)
            high = torch.tensor(n - 1, dtype=torch.int64)
            return BoundedTensorSpec(shape=(1,), low=low, high=high, dtype=torch.int64)
        else:
            raise NotImplementedError(f"Space type {type(space)} not supported.")

    def _make_specs(self):
        # Since TorchRL expects a CompositeSpec even for single spaces,
        # we need to make sure all specs are wrapped within a CompositeSpec.
        # Here we're wrapping the converted specs into a CompositeSpec explicitly.

        observation_spec = self._convert_space_to_spec(self.observation_space)
        action_spec = self._convert_space_to_spec(self.action_space)

        # Wrap the specs in a CompositeSpec
        self.observation_spec = CompositeSpec(observation=observation_spec)
        self.action_spec = CompositeSpec(action=action_spec)

    def _reset(self, tensordict=None, **kwargs):
        initial_state = self.drone_env.reset()
        # Initialize or update tensordict with all required keys
        if tensordict is None:
            tensordict = TensorDict({
                "done": torch.tensor([False], dtype=torch.bool, device=self.device),
                "observation": torch.tensor(initial_state, dtype=torch.float32, device=self.device),
                "reward": torch.tensor([0.0], dtype=torch.float32, device=self.device),

                "action": torch.tensor([0], dtype=torch.int64, device=self.device),  # Assuming default action
                "truncated": torch.tensor([False], dtype=torch.bool, device=self.device),  # Add truncated flag
                "terminated": torch.tensor([False], dtype=torch.bool, device=self.device)  # Add terminated flag
            }, batch_size=[])
        else:
            tensordict.set_("observation", torch.tensor(initial_state, dtype=torch.float32, device=self.device))
            tensordict.set_("reward", torch.tensor([0.0], dtype=torch.float32, device=self.device))
            tensordict.set_("done", torch.tensor([False], dtype=torch.bool, device=self.device))
            tensordict.set_("truncated", torch.tensor([False], dtype=torch.bool, device=self.device))
            tensordict.set_("terminated", torch.tensor([False], dtype=torch.bool, device=self.device))
            tensordict.set_("action", torch.tensor([0], dtype=torch.int64, device=self.device))
        return tensordict

    def _step(self, tensordict):
        # Extract the action from the tensordict
        action = tensordict.get("action").cpu().numpy()
        next_state, reward, done, _ = self.drone_env.step(action)

        # Directly initialize the next TensorDict with required keys and values
        next_td = TensorDict({
            "observation": torch.tensor(next_state, dtype=torch.float32, device=self.device),
            "reward": torch.tensor([reward], dtype=torch.float32, device=self.device),
            "done": torch.tensor([done], dtype=torch.bool, device=self.device),
            "terminated": torch.tensor([done], dtype=torch.bool, device=self.device),
            "truncated": torch.tensor([False], dtype=torch.bool, device=self.device),
        }, batch_size=torch.Size([]), device=self.device)

        # Encapsulate the next state within a 'next' sub-dictionary in the original tensordict
        tensordict.set_("next", next_td)
        return tensordict

    # def _step(self, tensordict):
    #     # Extract the action from the tensordict
    #     action = tensordict.get("action").cpu().numpy()
    #     next_state, reward, done, _ = self.drone_env.step(action)
    #
    #     # Create the nested "next" TensorDict
    #     next_tensordict = TensorDict({
    #         "done": torch.tensor([done], dtype=torch.bool, device=self.device),
    #         "observation": torch.tensor([next_state], dtype=torch.float32, device=self.device),
    #         "reward": torch.tensor([reward], dtype=torch.float32, device=self.device),
    #         # Assuming 'terminated' is equivalent to 'done' for your environment
    #         # Update this part as per your environment's requirements
    #         "terminated": torch.tensor([done], dtype=torch.bool, device=self.device),
    #         # Assuming 'truncated' can be inferred or is not applicable. Adjust as necessary.
    #         "truncated": torch.tensor([False], dtype=torch.bool, device=self.device)
    #     }, batch_size=[])
    #
    #     # Update the outer tensordict to include the "next" tensordict
    #     # and other required fields
    #     tensordict.set_("next", next_tensordict)
    #     tensordict.set_("done", torch.tensor([done], dtype=torch.bool, device=self.device))
    #     tensordict.set_("observation", torch.tensor(next_state, dtype=torch.float32, device=self.device))
    #     tensordict.set_("reward", torch.tensor([reward], dtype=torch.float32, device=self.device))
    #     # Update 'terminated' and other keys as needed
    #
    #     return tensordict

    def _set_seed(self, seed=None):
        # Assuming your Drone environment has a method to set seed, call it here.
        # This is optional and depends on whether your environment is deterministic and supports seeding.
        pass

    def render(self, mode='human'):
        # If your environment supports rendering, you can call its render method here.
        pass


drone_env = Drone()  # Instantiate your Drone environment
wrapped_env = DroneEnvWrapper(drone_env, device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))

td = wrapped_env.reset()

tensordict_out = wrapped_env.step(td)

print(len(td))

# action = wrapped_env.action_space.sample()
# print(f"action chosen was{action}")
# td.set_("action", torch.tensor([action], dtype=torch.int64))
# td = wrapped_env.step(td)
# # td = wrapped_env.step(td)
# # print(td.items()[0])
# for i, item in enumerate(td.items()):
#     print(i)
#     try:
#         print(item)
#     except:
#         print(f'error in item idx {i}')
# print('after step in for loop')
# print(f'next state after rand step is {td}')
#
# # print('after reset')
# # print(td)
# # print("Env observation_spec: \n", wrapped_env.observation_spec)
# # print("Env action_spec: \n", wrapped_env.action_spec)
# # print("Env reward_spec: \n", wrapped_env.reward_spec)
#
#
# for _ in range(10):  # Step through the environment for 10 steps
#     action = wrapped_env.action_space.sample()  # Randomly sample an action
#     td.set_("action", torch.tensor([action], dtype=torch.int64))
#     td = wrapped_env.step(td)
#     print('after step in for loop')
#     # print(f' and the next state is {td}')
#     if td.get("done").item():
#         print("Episode ended.")
#         break
#
# print('created drone env')
# drone_env = Drone()
# wrapped_drone_env = DroneEnvWrapper(drone_env)
#
# env = TransformedEnv(
#     wrapped_drone_env,
#     Compose(
#         ObservationNorm(in_keys=["observation"]),
#         DoubleToFloat(),
#         StepCounter(),
#     ),
# )
#
# print('before init_stats for observation normalization (do this ourselves)')
# env.transform[0].init_stats(num_iter=1000, reduce_dim=0, cat_dim=0)
#
# print('before reset')
# reset = env.reset()
# print(reset)
#
# print('reset finished')
#
# print('now starting to show results:\n\n')
#
# print("normalization constant shape:", env.transform[0].loc.shape)
#
# check_env_specs(env)
#
# rollout = env.rollout(3)
# print("rollout of three steps:", rollout)
# print("Shape of the rollout TensorDict:", rollout.batch_size)
