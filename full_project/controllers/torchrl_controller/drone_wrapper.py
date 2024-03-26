from typing import Optional

import gym
import torch as th
from tensordict import TensorDictBase, TensorDict
from torchrl.envs import EnvBase, step_mdp
from torchrl.data import UnboundedContinuousTensorSpec, CompositeSpec, BoundedTensorSpec




"""
# Info for CartPole-v1 test
    | Num | Action                 |
    |-----|------------------------|
    | 0   | Push cart to the left  |
    | 1   | Push cart to the right |

    | Num | Observation           | Min                 | Max               |
    |-----|-----------------------|---------------------|-------------------|
    | 0   | Cart Position         | -4.8                | 4.8               |
    | 1   | Cart Velocity         | -Inf                | Inf               |
    | 2   | Pole Angle            | ~ -0.418 rad (-24°) | ~ 0.418 rad (24°) |
    | 3   | Pole Angular Velocity | -Inf                | Inf               |
"""

class DroneWrapper(EnvBase):
    def __init__(self, env):
        super().__init__(device="cpu")
        self.env = env  # The underlying environment
        # Define the observation and action specs according to the wrapped environment
        self.observation_spec = CompositeSpec({"observation": BoundedTensorSpec(
            low=th.tensor([0.0, 0.0, -1.0], dtype=th.float64),
            high=th.tensor([1.0, 1.0, +4.0], dtype=th.float64))})
        self.action_spec = CompositeSpec({"action": BoundedTensorSpec(
            low=th.tensor([-1], dtype=th.int),
            high=th.tensor([4], dtype=th.int)
        )})

    def _step(self, tensordict: TensorDictBase) -> TensorDictBase:
        # Extract the action from the tensordict
        action = tensordict.get("action").cpu().numpy()
        # Step the underlying environment with the extracted action
        obs, reward, done, info = self.env.step(action)
        # Create a TensorDict to return, including the new observation, reward, and done flag
        td = TensorDict({
            "next": TensorDict(source={
                "done": th.tensor([done], dtype=th.bool),
                "observation": th.tensor(obs, dtype=th.float32),
                "reward": th.tensor([reward], dtype=th.float32),
                "terminated": th.tensor([done], dtype=th.bool),
                "truncated": th.tensor([done], dtype=th.bool)
            }, batch_size=[]),
        }, batch_size=[])
        return td

    def _reset(self, tensordict: Optional[TensorDictBase] = None, **kwargs) -> TensorDictBase:
        # Reset the underlying environment and get the initial observation
        obs = self.env.reset()
        print(obs)
        # Create a TensorDict for the initial state
        td = TensorDict(source={
            "observation": th.tensor(obs, dtype=th.float32),
            "action": th.tensor([0], dtype=th.float32),  # Placeholder action
            "done": th.tensor([False], dtype=th.bool),
            "terminated": th.tensor([False], dtype=th.bool),
            "truncated": th.tensor([False], dtype=th.bool)
        }, batch_size=[])
        return td

    def _set_seed(self, seed: Optional[int]):
        pass
        # self.env.seed(seed)  # Assuming the underlying env has a seed method


gym_env = gym.make('CartPole-v1')
env = DroneWrapper(gym_env)

reset_td = env.reset()

td = TensorDict(source={
    "action": th.tensor(0, dtype=th.float32),
    "done": th.tensor(False, dtype=th.bool),
    "next": TensorDict(source={
        "done": th.tensor(False, dtype=th.bool),
        "observation": th.tensor(0, dtype=th.float32),
        "reward": th.tensor(10, dtype=th.float32),
        "terminated": th.tensor(False, dtype=th.bool),
        "truncated": th.tensor(False, dtype=th.bool)
    }, batch_size=[]),
    "reward": th.tensor(5, dtype=th.float32),
    "observation": th.tensor(0, dtype=th.float32),
    "terminated": th.tensor(False, dtype=th.bool),
    "truncated": th.tensor(False, dtype=th.bool)
}, batch_size=[])

step_td = env.step(td)
env.rollout(3)

reset_with_action = env.rand_action(reset_td)
reset_with_action["action"]

data = step_mdp(step_td)
data
