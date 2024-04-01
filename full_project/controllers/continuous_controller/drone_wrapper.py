from typing import Optional
# noinspection PyPackageRequirements
import torch as th
# noinspection PyProtectedMember
from torchrl.envs.libs.gym import _gym_to_torchrl_spec_transform
from torchrl.envs import EnvBase
from torchrl.data import CompositeSpec
from tensordict import TensorDictBase, TensorDict


class DroneWrapper(EnvBase):
    def __init__(self, env):
        super().__init__(device="cpu")
        self.env = env  # The underlying environment
        self.observation_spec = CompositeSpec({
            "observation": _gym_to_torchrl_spec_transform(env.observation_space, dtype=th.float32)},)
        self.action_spec = _gym_to_torchrl_spec_transform(env.action_space)
        self.action_spec_size = len(env.action_space.low)
        self.state_spec = self.observation_spec.clone()
        self.reward_spec = _gym_to_torchrl_spec_transform(env.reward_space)

    def _step(self, tensordict: TensorDictBase) -> TensorDictBase:
        action = tensordict.get("action").detach().cpu().numpy()    # has to be in a list for drone_env
        obs, reward, done, info = self.env.step(action)
        out = TensorDict({
            "done": th.tensor([done], dtype=th.bool),
            "observation": th.tensor(obs, dtype=th.float32),
            "reward": th.tensor([reward], dtype=th.float32),
        }, batch_size=[])
        return out

    def _reset(self, tensordict: Optional[TensorDictBase] = None, **kwargs) -> TensorDictBase:
        obs = self.env.reset()
        out = TensorDict({
            "observation": th.tensor(obs, dtype=th.float32),
            "action": th.tensor([0]*self.action_spec_size, dtype=th.float32),  # Placeholder action
            "done": th.tensor([False], dtype=th.bool),
        }, batch_size=[])
        return out

    def _set_seed(self, seed: Optional[int]):
        rng = th.manual_seed(seed)
        self.rng = rng
