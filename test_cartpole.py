import math
import random
from collections import deque, namedtuple
from itertools import count

import matplotlib
import matplotlib.pyplot as plt

import gymnasium as gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from cplib import *

# Setup Gym
env = gym.make("CartPole-v1", render_mode='human')
state, info = env.reset()

# Load DQN
target_net = DQN(len(state), env.action_space.n).to(device)
target_net.load_state_dict(torch.load('cp.torch'))

# Run
for n in range(1000):
    state = torch.tensor(state, dtype=torch.float32,
                         device=device).unsqueeze(0)
    action = target_net(state).max(1)[1].view(1, 1)
    state, reward, terminated, truncated, info = env.step(action.item())
    if terminated or truncated:
        state, info = env.reset()
    if terminated:
        print('terminated')
    if truncated:
        print('truncated')
env.close()
