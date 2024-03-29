from typing import Optional

# import gym
# import gymnasium as gym
import numpy as np
import torch as th
from tensordict import TensorDictBase, TensorDict
from torchrl.envs.libs.gym import _gym_to_torchrl_spec_transform
from torchrl.envs import EnvBase, step_mdp

from torchrl.data import UnboundedContinuousTensorSpec, CompositeSpec, BoundedTensorSpec, OneHotDiscreteTensorSpec


class DroneWrapper(EnvBase):
    # check_env_specs(Drone())  # todo for some reason it explodes

    def __init__(self, env):
        super().__init__(device="cpu")
        self.env = env  # The underlying environment
        self.observation_spec = CompositeSpec({
            "observation": _gym_to_torchrl_spec_transform(env.observation_space, dtype=th.float32)},)
        self.action_spec = _gym_to_torchrl_spec_transform(env.action_space)
        self.state_spec = self.observation_spec.clone()
        self.reward_spec = _gym_to_torchrl_spec_transform(env.reward_space)

    def _step(self, tensordict: TensorDictBase) -> TensorDictBase:
        action = [np.argmax(tensordict.get("action").detach().cpu().numpy())]    # has to be in a list for drone_env
        obs, reward, done, info = self.env.step(action)
        out = TensorDict({
            "done": th.tensor([done], dtype=th.bool),
            "observation": th.tensor(obs, dtype=th.float32),
            "reward": th.tensor([reward], dtype=th.float32),
        }, batch_size=[])
        return out

    def _reset(self, tensordict: Optional[TensorDictBase] = None, **kwargs) -> TensorDictBase:
        # Reset the underlying environment and get the initial observation
        obs = self.env.reset()
        # print(obs)
        # Create a TensorDict for the initial state
        out = TensorDict({
            "observation": th.tensor(obs, dtype=th.float32),
            "action": th.tensor([0], dtype=th.float32),  # Placeholder action
            "done": th.tensor([False], dtype=th.bool),
        }, batch_size=[])
        return out

    def _set_seed(self, seed: Optional[int]):
        rng = th.manual_seed(seed)
        self.rng = rng
