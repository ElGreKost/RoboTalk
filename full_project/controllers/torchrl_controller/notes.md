# todo
* study torhcrl.step_mdp() **learned that it returns the next from the step**
* study softUpdate \theta_t = \theta_{t-1} * \epsilon + \theta_t * (1-\epsilon)
* study EGreedyModule

# after the last commit, the new error from check_env_spec(env) is
```
Traceback (most recent call last):
  File "/home/kostiskak/Documents/GitHub/RoboTalk/full_project/controllers/torchrl_controller/torchrl_controller.py", line 40, in <module>
    check_env_specs(env)
  File "/home/kostiskak/ntua/Robotalk/rl/torchrl/envs/utils.py", line 617, in check_env_specs
    fake_tensordict = env.fake_tensordict()
                      ^^^^^^^^^^^^^^^^^^^^^
  File "/home/kostiskak/ntua/Robotalk/rl/torchrl/envs/common.py", line 2832, in fake_tensordict
    fake_input = fake_state.update(fake_action)
                 ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/kostiskak/ntua/Robotalk/tensordict/tensordict/base.py", line 2690, in update
    self._set_tuple(
  File "/home/kostiskak/ntua/Robotalk/tensordict/tensordict/_td.py", line 1593, in _set_tuple
    return self._set_str(key[0], value, inplace=inplace, validated=validated)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/kostiskak/ntua/Robotalk/tensordict/tensordict/_td.py", line 1563, in _set_str
    value = self._validate_value(value, check_shape=True)
            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/kostiskak/ntua/Robotalk/tensordict/tensordict/base.py", line 4462, in _validate_value
    raise RuntimeError(
RuntimeError: batch dimension mismatch, got self.batch_size=torch.Size([3]) and value.shape=torch.Size([1]).
```