from tensordict import TensorDictBase, TensorDict
from torchrl.envs import EnvBase, step_mdp
from torchrl.data import UnboundedContinuousTensorSpec, CompositeSpec
from typing import Optional
import torch as th


class CustomWrapper(EnvBase):
    def __init__(self):
        super().__init__(
            device="cpu"
        )

    def _step(self, tensordict: TensorDictBase, ) -> TensorDictBase:
        """
        TensorDict(
        fields={
            action: Tensor(shape=torch.Size([1]), device=cpu, dtype=torch.float32, is_shared=False),
            done: Tensor(shape=torch.Size([1]), device=cpu, dtype=torch.bool, is_shared=False),
            next: TensorDict(
                fields={
                    done: Tensor(shape=torch.Size([1]), device=cpu, dtype=torch.bool, is_shared=False),
                    observation: Tensor(shape=torch.Size([3]), device=cpu, dtype=torch.float32, is_shared=False),
                    reward: Tensor(shape=torch.Size([1]), device=cpu, dtype=torch.float32, is_shared=False),
                    terminated: Tensor(shape=torch.Size([1]), device=cpu, dtype=torch.bool, is_shared=False),
                    truncated: Tensor(shape=torch.Size([1]), device=cpu, dtype=torch.bool, is_shared=False)},
                batch_size=torch.Size([]),
                device=cpu,
                is_shared=False),
            observation: Tensor(shape=torch.Size([3]), device=cpu, dtype=torch.float32, is_shared=False),
            terminated: Tensor(shape=torch.Size([1]), device=cpu, dtype=torch.bool, is_shared=False),
            truncated: Tensor(shape=torch.Size([1]), device=cpu, dtype=torch.bool, is_shared=False)},
        batch_size=torch.Size([]),
        device=cpu,
        is_shared=False)
        """
        return TensorDict(source={
            "reward": th.tensor(10, dtype=th.float32),
            "next": TensorDict(source={
                "done": th.tensor(False, dtype=th.bool),
                "observation": th.tensor(0, dtype=th.float32),
                "reward": th.tensor(10, dtype=th.float32),
                "terminated": th.tensor(False, dtype=th.bool),
                "truncated": th.tensor(False, dtype=th.bool)
            }, batch_size=[]),
        }, batch_size=[])

    def _reset(self, tensordict: TensorDictBase, **kwargs) -> TensorDictBase:
        """
        TensorDict(
            fields={
                action: Tensor(shape=torch.Size([1]), device=cpu, dtype=torch.float32, is_shared=False),
                done: Tensor(shape=torch.Size([1]), device=cpu, dtype=torch.bool, is_shared=False),
                observation: Tensor(shape=torch.Size([3]), device=cpu, dtype=torch.float32, is_shared=False),
                terminated: Tensor(shape=torch.Size([1]), device=cpu, dtype=torch.bool, is_shared=False),
                truncated: Tensor(shape=torch.Size([1]), device=cpu, dtype=torch.bool, is_shared=False)},
            batch_size=torch.Size([]),
            device=cpu,
            is_shared=False)
        """
        td = TensorDict(source={
            "action": th.tensor(0, dtype=th.float32),
            "done": th.tensor(False, dtype=th.bool),
            "observation": th.tensor(0, dtype=th.float32),
            "terminated": th.tensor(False, dtype=th.bool),
            "truncated": th.tensor(False, dtype=th.bool)
        }, batch_size=[])
        return td

    def _set_seed(self, seed: Optional[int]):
        pass

    # def set_state(self):
    #     raise NotImplementedError
    # def forward(self, tensordict: TensorDictBase) -> TensorDictBase:
    #     pass


env = CustomWrapper()
env.observation_spec = CompositeSpec({"action": UnboundedContinuousTensorSpec(1)})
env.action_spec = CompositeSpec({"observation": UnboundedContinuousTensorSpec(1)})
reset_td = env.reset()  # reset worked

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
