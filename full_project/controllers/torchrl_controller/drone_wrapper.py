from typing import Optional

import gym
import numpy as np
import torch as th
from tensordict import TensorDictBase, TensorDict
from torchrl.envs import EnvBase, step_mdp
from torchrl.data import UnboundedContinuousTensorSpec, CompositeSpec, BoundedTensorSpec


class DroneWrapper(EnvBase):
    # check_env_specs(Drone())  # todo for some reason it explodes

    def __init__(self, env):
        super().__init__(device="cpu")
        self.env = env  # The underlying environment
        # Define the observation and action specs according to the wrapped environment
        # obs = d_r, d_theta, last_action
        self.observation_spec = CompositeSpec({"observation": BoundedTensorSpec(
            low=th.tensor([0.0, -np.pi, -1.0], dtype=th.float64),
            high=th.tensor([3.0, np.pi, +4.0], dtype=th.float64))})
        self.action_spec = CompositeSpec({"action": BoundedTensorSpec(
            low=th.tensor([-1], dtype=th.int),
            high=th.tensor([4], dtype=th.int)
        )})

    def _step(self, tensordict: TensorDictBase) -> TensorDictBase:
        action = tensordict.get("action").detach().cpu().numpy()
        obs, reward, done, info = self.env.step(action)
        out = TensorDict({
            "done": th.tensor([done], dtype=th.bool),
            "observation": th.tensor(obs, dtype=th.float32),
            "reward": th.tensor([reward], dtype=th.float32),
        }, batch_size=[])
        return out

    def _reset(self, tensordict: Optional[TensorDictBase] = None, **kwargs) -> TensorDictBase:
        # Reset the underlying environment and get the initial observation
        obs = self.env.reset()[0]
        print(obs)
        # Create a TensorDict for the initial state
        out = TensorDict({
            "observation": th.tensor(obs, dtype=th.float32),
            "action": th.tensor([0], dtype=th.float32),  # Placeholder action
            "done": th.tensor([False], dtype=th.bool),
        }, batch_size=[])
        return out

    def _set_seed(self, seed: Optional[int]):  # for test reproduction
        pass
        # self.env.seed(seed)  # Assuming the underlying env has a seed method


if __name__ == "__name__":
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
