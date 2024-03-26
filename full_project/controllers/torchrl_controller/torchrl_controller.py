from torchrl.envs import check_env_specs

from drone_wrapper import DroneWrapper
from drone_env import Drone

import torch as th
from torch.distributions import Normal

from torchrl.modules import MLP, Actor, ProbabilisticActor

from tensordict.nn import TensorDictModule
from tensordict.nn.distributions import NormalParamExtractor

drone_env = Drone()
env = DroneWrapper(drone_env)

reset_td = env.reset()
print(reset_td.items())

module = th.nn.LazyLinear(env.action_spec.shape[-1])
policy = Actor(module)
# policy = TensorDictModule(
#     module,
#     in_keys=["observation"],
#     out_keys=["actions"],
# )

rollout = env.rollout(max_steps=1, policy=policy)
print(rollout)
# print(f'reset_td is: {reset_td}')

# Now doing Module testing


# print(env.action_spec.shape[-1])
# print(env.observation_spec)
# print(env.reward_spec.shape[-1])
# print(env.done_spec.shape[-1])

# check_env_specs(env)
#


# policy = Actor(module)
# rollout = env.rollout(max_steps=10, policy=policy)
# print(rollout)
