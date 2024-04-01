import time
from collections import defaultdict

from torchrl.objectives.value import GAE
from torchrl.record import CSVLogger
from tqdm import tqdm

from drone_wrapper import DroneWrapper
from drone_env import Drone

import torch as th
from torch import nn

from torchrl.envs import TransformedEnv, StepCounter, Compose, ObservationNorm
from torchrl.modules import MLP, EGreedyModule, QValueActor, ProbabilisticActor, TanhNormal, ValueOperator
from torchrl.collectors import SyncDataCollector
from torchrl.data import LazyTensorStorage, ReplayBuffer, MultiStep, SamplerWithoutReplacement
from torchrl.objectives import ClipPPOLoss
from torchrl._utils import logger as torchrl_logger
from torchrl.envs.utils import ExplorationType, set_exploration_type

from tensordict.nn.distributions import NormalParamExtractor
from tensordict.nn import TensorDictModule as Mod

drone_env = Drone()
env = TransformedEnv(DroneWrapper(drone_env), Compose(StepCounter()))
env.set_seed(0)

# PPO params
num_epochs = 10
clip_epsilon = 0.2  # PPO clip value for loss
gamma = 0.99
lmbda = 0.95
entropy_eps = 1e-4

device = th.device("cpu")
num_cells = 32
depth = 3
lr = 3e-4
max_grad_norm = 1.0


def make_actor_critic():
    # The 2 * action_sped is because for each action we produce 2 scalars (loc, scale)
    actor_net = nn.Sequential(
        MLP(out_features=2 * env.action_spec.shape[-1], depth=depth, num_cells=num_cells, activation_class=nn.ELU),
        NormalParamExtractor(),
    )

    policy_module = Mod(actor_net, in_keys=["observation"], out_keys=["loc", "scale"])

    policy_module = ProbabilisticActor(
        module=policy_module,
        spec=env.action_spec,
        in_keys=["loc", "scale"],
        distribution_class=TanhNormal,
        distribution_kwargs={"min": env.action_spec.space.low, "max": env.action_spec.space.high, },
        return_log_prob=True
    )

    # Value net
    value_module = ValueOperator(
        module=MLP(out_features=1, depth=depth, num_cells=num_cells, activation_class=nn.ELU),
        in_keys=["observation"],
    )
    return policy_module, value_module


policy_module, value_module = make_actor_critic()
print("Running policy:", policy_module(env.reset()))
print("Running value:", value_module(env.reset()))

# Data collector
init_rand_steps = 1024
frames_per_batch = 1024
updates_per_subdata = 10
total_frames = 10_000
sub_batch_size = 256
rb_size = 20_000

collector = SyncDataCollector(
    env,
    policy_module,
    frames_per_batch=frames_per_batch,  # main batch size from agent experience
    total_frames=-1,  # plays the agent and collects data to train forever
    init_random_frames=init_rand_steps,
    postproc=MultiStep(gamma=0.99, n_steps=5),
)

rb = ReplayBuffer(storage=LazyTensorStorage(rb_size), batch_size=sub_batch_size,
                  sampler=SamplerWithoutReplacement())

# Advantage and Loss
advantage_module = GAE(gamma=gamma, lmbda=lmbda, value_network=value_module, average_gae=True)

loss_module = ClipPPOLoss(
    actor_network=policy_module,
    critic_network=value_module,
    clip_epsilon=clip_epsilon,
    entropy_bonus=bool(entropy_eps),
    entropy_coef=entropy_eps,
    critic_coef=1.0,
    loss_critic_type="smooth_l1",
)

optim = th.optim.Adam(loss_module.parameters(), lr)
scheduler = th.optim.lr_scheduler.CosineAnnealingLR(optim, total_frames // frames_per_batch, 0.0)
# Training Loop
logs = defaultdict(list)
pbar = tqdm(total=total_frames)
eval_str = ""

# Logger
path = "./training_loop"
logger = CSVLogger(exp_name="ppo", log_dir=path)

# Main Loop
total_count = 0
total_episodes = 0
max_episode_steps = sub_batch_size
t0 = time.time()
collector.load_state_dict(th.load('weights/pretrained_simple.pt'))

for i, tensordict_data in enumerate(collector):  # len(tensordict_data) = frames_per_batch
    print('max episode steps', max_episode_steps)
    data_view = tensordict_data.reshape(-1)
    rb.extend(data_view.cpu())
    for _ in range(num_epochs):  # how many times to retrain with current rb_data
        # We'll need an "advantage" signal to make PPO work.
        # We re-compute it at each epoch as its value depends on the value
        # network which is updated in the inner loop.
        advantage_module(tensordict_data)
        for _ in range(updates_per_subdata):  # Updates Per Data
            subdata = rb.sample()  # len(subdata) = sub_batch_size, of the rb
            loss_vals = loss_module(subdata.to(device))
            loss_value = (
                    loss_vals["loss_objective"]
                    + loss_vals["loss_critic"]
                    + loss_vals["loss_entropy"]
            )
            loss_value.backward()
            th.nn.utils.clip_grad_norm_(loss_module.parameters(), max_grad_norm)
            optim.step()
            optim.zero_grad()

    logs["reward"].append(tensordict_data["next", "reward"].mean().item())
    total_episodes += tensordict_data['next', 'done'].sum()
    pbar.update(tensordict_data.numel())
    cum_reward_str = (
        f"average reward={logs['reward'][-1]: 4.4f} (init={logs['reward'][0]: 4.4f})"
    )
    logs["step_count"].append(tensordict_data["step_count"].max().item())
    stepcount_str = f"step count (max): {logs['step_count'][-1]}"
    logs["lr"].append(optim.param_groups[0]["lr"])
    lr_str = f"lr policy: {logs['lr'][-1]: 4.4f}"
    if i % 10 == 0:
        # We evaluate the policy once every 10 batches of data.
        # (take the expected value of the action distribution) for a given
        # number of steps (1000, which is our ``env`` horizon).
        with set_exploration_type(ExplorationType.MEAN), th.no_grad():
            eval_rollout = env.rollout(1000, policy_module)
            logs["eval reward"].append(eval_rollout["next", "reward"].mean().item())
            logs["eval reward (sum)"].append(
                eval_rollout["next", "reward"].sum().item()
            )
            logs["eval step_count"].append(eval_rollout["step_count"].max().item())
            eval_str = (
                f"eval cumulative reward: {logs['eval reward (sum)'][-1]: 4.4f} "
                f"(init: {logs['eval reward (sum)'][0]: 4.4f}), "
                f"eval step-count: {logs['eval step_count'][-1]}"
            )
            del eval_rollout
    if i % 40 == 0:
        print("in i: ", i)
        th.save(collector.state_dict(), f"./weights/collector_state_dict{i}.pt")

    pbar.set_description(", ".join([eval_str, cum_reward_str, stepcount_str, lr_str]))
    max_episode_steps = rb[:]['next', 'step_count'].max()

    scheduler.step()

print('finished')

t1 = time.time()

torchrl_logger.info(
    f"solved after {total_count} steps, {total_episodes} episodes and in {t1 - t0}s."
)

# Save recording
env.rollout(max_steps=1000, policy=policy_module)
