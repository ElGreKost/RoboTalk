from typing import Optional

import gym
import torch as th
from tensordict import TensorDictBase, TensorDict
from torchrl.envs import EnvBase, step_mdp, check_env_specs
from torchrl.data import UnboundedContinuousTensorSpec, CompositeSpec, BoundedTensorSpec, BinaryDiscreteTensorSpec

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


class CartPoleWrapper(EnvBase):
    def __init__(self, env):
        super().__init__(device="cpu")
        self.env = env  # The underlying environment
        # Define the observation and action specs according to the wrapped environment
        self.observation_spec = CompositeSpec({"observation": BoundedTensorSpec(
            low=th.tensor([-4.8, -th.inf, -0.418, -th.inf], dtype=th.float64),
            high=th.tensor([4.8, th.inf, 0.418, th.inf], dtype=th.float64))})
        self.action_spec = CompositeSpec({"action": BinaryDiscreteTensorSpec(1)})
        self.reward_spec = CompositeSpec({"reward": BoundedTensorSpec(th.tensor([0]), th.inf)})

    def _step(self, tensordict: TensorDictBase) -> TensorDictBase:
        # Extract the action from the tensordict
        action = tensordict.get("action").detach().cpu().numpy()
        # Step the underlying environment with the extracted action
        obs, reward, term, trunc, info = self.env.step(int(action))
        done = term
        # Create a TensorDict to return, including the new observation, reward, and done flag
        out = TensorDict({
            "done": th.tensor([done], dtype=th.bool),
            "observation": th.tensor(obs, dtype=th.float32),
            "reward": th.tensor([reward], dtype=th.float32),
        }, batch_size=[])
        return out

    def _reset(self, tensordict: Optional[TensorDictBase] = None, **kwargs) -> TensorDictBase:
        # Reset the underlying environment and get the initial observation
        obs = self.env.reset()[0]
        # Create a TensorDict for the initial state
        out = TensorDict({
            "observation": th.tensor(obs, dtype=th.float32),
            "action": th.tensor([0], dtype=th.float32),  # Placeholder action
            "done": th.tensor([False], dtype=th.bool),
        }, batch_size=[])
        return out

    def _set_seed(self, seed: Optional[int]):  # for reproduction of same results
        pass
        # self.env.seed(seed)  # Assuming the underlying env has a seed method


gym_env = gym.make('CartPole-v1')
env = CartPoleWrapper(gym_env)

reset_td = env.reset()

step_td = env.step(reset_td)
rollout_td = env.rollout(3)
print(rollout_td)
print('finished rollout')
reset_with_action = env.rand_action(reset_td)
print(reset_with_action["action"])

data = step_mdp(step_td)
print(data)

check_env_specs(env)
