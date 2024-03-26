from torchrl.envs import step_mdp, check_env_specs

from drone_wrapper import DroneWrapper
from drone_env import Drone

drone_env = Drone()
env = DroneWrapper(drone_env)

reset_td = env.reset()
print(f'reset_td is: {reset_td}')

step_td = env.step(reset_td)
print(f'step_td is: {step_td}')

rollout_td = env.rollout(100)
print(f'rollout_td is: {rollout_td}')

data = step_mdp(step_td)
print(data)
