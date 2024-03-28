import time

from torchrl.envs import check_env_specs
from torchrl.record import CSVLogger

from drone_wrapper import DroneWrapper
from drone_env import Drone

import torch as th
from torch.distributions import Normal
from torch.optim import Adam

from torchrl.envs import TransformedEnv, StepCounter, Compose, ObservationNorm
from torchrl.modules import MLP, EGreedyModule, QValueActor
from torchrl.collectors import SyncDataCollector
from torchrl.data import LazyTensorStorage, ReplayBuffer
from torchrl.objectives import DQNLoss, SoftUpdate
from torchrl._utils import logger as torchrl_logger

from tensordict.nn import TensorDictModule
from tensordict.nn.distributions import NormalParamExtractor
from tensordict.nn import TensorDictModule as Mod, TensorDictSequential as Seq

drone_env = Drone()
env = TransformedEnv(DroneWrapper(drone_env), Compose(ObservationNorm(in_keys=["observation"]), StepCounter()))
env.set_seed(0)
env.transform[0].init_stats(num_iter=1000, reduce_dim=0, cat_dim=0)
# print("normalization constant shape:", env.transform[0].loc)

value_mlp = MLP(out_features=env.action_spec.shape[-1], num_cells=[64, 64])
value_net = Mod(value_mlp, in_keys=["observation"], out_keys=["action_value"])
policy = QValueActor(value_net, spec=env.action_spec)   # there is problem with the support but I think that this is because of QValueModule
exploration_module = EGreedyModule(env.action_spec, annealing_num_steps=100_000, eps_init=0.5)
policy_explore = Seq(policy, exploration_module)

# Data collector
init_rand_steps = 700
frames_per_batch = 128
optim_steps = 10
collector = SyncDataCollector(
    env,
    policy,
    frames_per_batch=frames_per_batch,
    total_frames=-1,
    init_random_frames=init_rand_steps
)
rb = ReplayBuffer(storage=LazyTensorStorage(5_000))

# Loss Module and Optimization

loss = DQNLoss(value_network=policy, action_space=env.action_spec, delay_value=True)
optim = Adam(loss.parameters(), lr=0.02)
updater = SoftUpdate(loss, eps=0.99)

# Logger
path = "./training_loop"
logger = CSVLogger(exp_name="dqn", log_dir=path)

# Main Loop
total_count = 0
total_episodes = 0
t0 = time.time()
print('starting loop')
for i, data in enumerate(collector):
    print('outer loop')
    # print(data)
    # Write data in rb
    rb.extend(data)
    max_length = rb[:]['next', 'step_count'].max()
    if len(rb) > init_rand_steps:
        print('enough rb collected frames')
        for _ in range(optim_steps):
            sample = rb.sample(128)
            loss_vals = loss(sample)
            loss_vals["loss"].backward()
            optim.step()
            optim.zero_grad()
            # Update the exploration factor
            exploration_module.step(data.numel())
            # Update target params
            updater.step()
            if i % 10:
                torchrl_logger.info(f"Max num steps: {max_length}, rb length {len(rb)}")
            total_count += data.numel()
            total_episodes += data['next', 'done'].sum()

    if max_length > 4000:
        break   # truncate it


t1 = time.time()

torchrl_logger.info(
    f"solved after {total_count} steps, {total_episodes} episodes and in {t1-t0}s."
)

# Save recording
# env.rollout(max_steps=1000, policy=policy)

