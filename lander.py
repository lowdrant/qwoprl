#!/usr/bin/env python3
"""
Lunar Lander gym playground
"""
from sys import argv

import matplotlib.pyplot as plt
from numpy import linspace, pi, vstack

import gymnasium as gym
from myrl import DiscretizerFactory, QTable, plotrl

if __name__ == '__main__':

    # Gym Setup
    env = gym.make("LunarLander-v2", render_mode="human")

    # Policy Setup
    ds = DiscretizerFactory([
        {-0.1: '<', 0: '<', 0.1: '>'},  # x
        {0: '<', 0.1: '<', 1: '<'},  # y
        {-1: '<', -0.1: '<', 1: '>', 0.1: '>'},  # vx
        {-1: '<', -0.1: '<', 1: '>', 0.1: '>'},  # vy
        {v: '<' for v in linspace(0.1, pi, 4)},  # theta
        {-1: '<', -0.1: '<', 1: '>', 0.1: '>'}  # omega
    ])
    policy = QTable(ds.n, 4, env.action_space.sample, ds, eps=0.1, alpha=0.8)

    # Experiment
    olog, rlog, alog, ilog = [], [], [], []
    N = 100 if len(argv) < 2 else int(argv[1])
    state, info = env.reset(seed=42)
    for _ in range(N):
        # Action
        action = policy.pick_action(state[:-2])
        next_state, reward, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            observation, info = env.reset()

        # Update
        policy.update_reward(state[:-2], action, next_state[:-2], reward)
        state = next_state

        olog.append(state)
        alog.append(action)
        rlog.append(reward)
    env.close()

    olog = vstack(olog).T
    plotrl(olog, rlog, alog)
    plt.show()
