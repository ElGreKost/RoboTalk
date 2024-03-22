from collections import deque
import random


class ReplayBuffer:
    """Available functions are:
    - push (s,a,r,s,term,trunc)
    - sample(batch_size)
    - len()"""

    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    # todo maybe change this to take observations instead of sarstt
    def push(self, state, action, reward, next_state, done):
        experience = (state, action, reward, next_state, done)
        self.buffer.append(experience)

    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)

    def __len__(self):
        return len(self.buffer)