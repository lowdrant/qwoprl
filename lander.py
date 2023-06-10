#!/usr/bin/env python3
import matplotlib.pyplot as plt
from numpy import argmax, asarray, cos, sin, vstack, zeros
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


if __name__ == '__main__':
    import gymnasium as gym
    env = gym.make("LunarLander-v2")  # , render_mode="human")

    olog, rlog, alog, ilog = [], [], [], []
    obs, info = env.reset(seed=42)

    for _ in range(500):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            observation, info = env.reset()
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
