#!/usr/bin/env python3
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from numpy import argmax, asarray, cos, max, sin, vstack, zeros
from numpy.random import rand, seed


def plotrl(obs, reward, action, fig=None, **figkw):
    """Plot RL outputs
        INPUTS:
            obs -- NxM -- M observation vectors of dimension N
            reward -- Mx1 -- reward at each timestep
            action -- Mx1 -- action at each timestep
            fig -- figure object for simpler plotting
            figkw -- If fig is None, kwargs for fig call. Otherwise unused
        OUTPUTS:
            fig
    """
    if fig is None:
        fig = plt.figure(**figkw)
    fig.set_tight_layout(True)
    gs = GridSpec(3, 2)
    axobs = fig.add_subplot(gs[:-1, :])
    axact = fig.add_subplot(gs[-1, 0])
    axrwrd = fig.add_subplot(gs[-1, 1], sharex=axact)

    plt.setp(axobs, ylabel='y', xlabel='x', xlim=(-1.5, 1.5), aspect='equal')
    plt.setp(axact, ylabel='action', xlabel='time index')
    plt.setp(axrwrd, ylabel='reward', xlabel='time index')

    axobs.grid()
    axobs.plot(*obs[:2], 'k.-')
    axact.step(range(len(action)), action)
    axrwrd.grid()
    axrwrd.plot(reward, '.-')
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


def discretize_state(state):
    """Convert state to index in state table.
        "Grids" out state space into uneven chunks to simplify policy learning.
        TODO: I wonder if there's a way to make this work continuously
    """
    ret = 0
    x, y, vx, vy, th, w = state[:-2]

    # X Coord
    # - left, right, close to center
    if x < -0.1:  # left
        ret += 1
    elif x < 0:  # center left
        ret += 2
    elif x > 0.1:  # right
        ret += 3
    elif x > 0:  # center right
        ret += 4

    # Y Coord
    # - log-ish-scale approaching ground
    # - ret can shift by up to 3, so consecutive diff must be >3 now
    retdelta = 5
    if y < 0:
        ret += 1 * retdelta
    elif y < 0.01:
        ret += 2 * retdelta
    elif y < 0.1:
        ret += 3 * retdelta
    elif y < 1:
        ret += 4 * retdelta

    # Vx
    # - log scale, pos and neg vel
    retdelta *= 4
    if vx < -10:
        ret += 1 * retdelta
    elif vx < -1:
        ret += 2 * retdelta
    elif vx < -0.1:
        ret += 3 * retdelta
    elif vx > 10:
        ret += 4 * retdelta
    elif vx > 1:
        ret += 5 * retdelta
    elif vx > 0.1:
        ret += 6 * retdelta

    # Vy
    # - log scale, pos and neg vel
    retdelta *= 6
    if vy < -10:
        ret += 1 * retdelta
    elif vy < -1:
        ret += 2 * retdelta
    elif vy < -0.1:
        ret += 3 * retdelta
    elif vy > 10:
        ret += 4 * retdelta
    elif vy > 1:
        ret += 5 * retdelta
    elif vy > 0.1:
        ret += 6 * retdelta

    # TODO: Angle
    # TODO: Angular Velocity
    # TODO: auto-chop up logarithmically-ish

    return ret


if __name__ == '__main__':
    from sys import argv

    import gymnasium as gym
    env = gym.make("LunarLander-v2", render_mode="human")

    olog, rlog, alog, ilog = [], [], [], []

    state, info = env.reset(seed=42)
    N = 1000
    if len(argv) > 1:
        N = int(argv[1])

    policy = QTable(6000, 4, env.action_space.sample, discretize_state)
    for _ in range(N):
        # Action
        action = policy.pick_action(state)
        # action = env.action_space.sample()
        next_state, reward, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            observation, info = env.reset()

        # Update
        policy.update_reward(state, action, next_state, reward)
        state = next_state

        olog.append(state)
        alog.append(action)
        rlog.append(reward)
    env.close()

    olog = vstack(olog).T
    plotrl(olog, rlog, alog)
    plt.show()
