import gymnasium as gym
from BaseAgent import BaseAgent


class NstepAgent(BaseAgent):
    def __init__(self, _env: gym.envs, use_conv=True, _lr=3e-4, _gamma=0.99, _buffer_size=10000, _nsteps=3):
        super().__init__(_env, use_conv, _lr, _gamma, _buffer_size)
        self.nsteps, self.nstep_buffer = _nsteps, []
        self.learn_start = 100  # todo maybe make them attrs of the BaseAgent
        self.update_freq = 1

    def append_to_replay(self, s, a, r, s_, d):
        self.nstep_buffer.append((s, a, r, s_))
        if len(self.nstep_buffer) < self.nsteps:
            return

        R = sum([self.nstep_buffer[i][2] * (self.gamma ** i) for i in range(self.nsteps)])
        _state, _action, _, _ = self.nstep_buffer.pop(0)

        self.replay_buffer.push(_state, _action, R, s_, d)

    def update(self, s, a, r, s_, d):
        self.frame += 1
        self.append_to_replay(s, a, r, s_, d)
        # todo add clip -1, 1, add learn_start frame

        if self.frame < self.learn_start or self.frame % self.update_freq != 0:
            return None
        _batch = self.replay_buffer.sample(self.batch_size)
        _loss = self.compute_loss(_batch)

        self.optimizer.zero_grad()
        _loss.backward()
        self.optimizer.step()

        return _loss

    def finish_nstep(self):
        while len(self.nstep_buffer) > 0:
            R = sum([self.nstep_buffer[i][2] * (self.gamma ** i) for i in range(len(self.nstep_buffer))])
            _state, _action, _, _ = self.nstep_buffer.pop(0)

            self.replay_buffer.push((_state, _action, R, None))