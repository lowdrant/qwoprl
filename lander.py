#!/usr/bin/env python3
import matplotlib.pyplot as plt
from numpy import argmax, asarray, cos, max, sin, vstack, zeros
from numpy.random import rand, seed


def r2d(th):
    """1x... angles -> 2x2x... rot matrices"""
    return asarray([[cos(th), -sin(th)], [sin(th), cos(th)]])


def plotobs(obs, fig=None):
    """8xN states"""
    if fig is None:
        fig = plt.figure(1)
    fig.clf()
    x, y, vx, vy, th, w, c1, c2 = obs

    ax = fig.add_subplot(111)
    ax.set_aspect('equal')
    ax.grid()

    ax.plot(x, y, 'k.-')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_xlim((-1.5, 1.5))

    return fig


class QTable:
    """
        n -- state space size
        m -- action space size
        action_sample -- action sampling callable
        alpha -- learning rate param
        gamma -- learning rate param
        eps -- probability of randomly choosing an action vs. a learned one
    """

    def __init__(self, n, m, action_sample, discretize_state,
                 alpha=0.5, gamma=0.5, eps=0.25):
        self.table = zeros([n, m], dtype=float)
        self.action_sample = action_sample
        self.discretize_state = discretize_state
        self.alpha, self.gamma, self.eps = alpha, gamma, eps

    def pick_action(self, state):
        """pick action from qtable"""
        if rand() < self.eps:
            return self.action_sample()
        return argmax(self.table[self.discretize_state(state)])

    def update_reward(self, state, action, next_state, reward):
        """update optimal action in qtable"""
        state = self.discretize_state(state)
        next_state = self.discretize_state(next_state)
        old_value = self.table[state, action]
        next_max = max(self.table[next_state])

        # Update the new value
        new_value = ((1 - self.alpha) * old_value
                     + self.alpha * (reward + self.gamma * next_max))
        self.table[state, action] = new_value


if __name__ == '__main__':
    import gymnasium as gym
    env = gym.make("LunarLander-v2")  # , render_mode="human")

    olog, rlog, alog, ilog = [], [], [], []

    obs, info = env.reset(seed=42)
    # policy = QTable(8, 4, env.action_space.sample, discretize_state)
    for _ in range(500):
        # Action
        # action = policy.pick_action(obs)
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            observation, info = env.reset()

        # Update
        # policy.update_reward(state, action, next_state, reward)
        # state = next_state

        olog.append(obs)
        alog.append(action)
    env.close()

    olog = vstack(olog).T
    plotobs(olog)
    fig = plt.figure(2)
    fig.clf()
    ax = fig.add_subplot(111)
    ax.step(range(len(alog)), alog)
    plt.show()
