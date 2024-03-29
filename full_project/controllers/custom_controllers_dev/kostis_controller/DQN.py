from torch import nn

class DQN(nn.Module):
    def __init__(self, observation_shape, action_size, use_conv=True):
        super().__init__()

        self.action_size = action_size

        # Because frames (4) frames are stacked, we use them as if they were channels
        self.rolling_frames = observation_shape[0] if use_conv else observation_shape

        self.conv = nn.Sequential(
            nn.Conv2d(self.rolling_frames, 32, kernel_size=8, stride=4), nn.ReLU(),
            nn.LazyConv2d(64, kernel_size=4, stride=2), nn.ReLU(),
            nn.LazyConv2d(64, kernel_size=3, stride=1), nn.ReLU(), nn.Flatten()
        ) if use_conv else nn.Sequential()

        self.fc = nn.Sequential(
            nn.LazyLinear(128), nn.ReLU(),
            nn.LazyLinear(256), nn.ReLU(),
            nn.LazyLinear(self.action_size)
        )
        self.net = nn.Sequential(self.conv, self.fc)

    def forward(self, state):
        return self.net(state)


# simple test
"""# Initialize environment (example: CartPole)
env = gym.make('CartPole-v1')
state = env.reset()

# Initialize your DQN model
model = DQN(observation_shape=env.observation_space.shape, action_size=env.action_space.n, use_conv=False).to(device)

# Test the model with a few steps in the environment
for _ in range(10):
    action = model(torch.from_numpy(state).float().unsqueeze(0).to(device)).max(1)[1].view(1, 1).item()
    state, reward, done, _ = env.step(action)
    if done:
        break"""
