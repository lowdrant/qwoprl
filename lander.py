#!/usr/bin/env python3
"""
Lunar Lander gym playground
"""
from argparse import ArgumentParser

import matplotlib.pyplot as plt
from numpy import linspace, pi, vstack
from numpy.random import seed

import gymnasium as gym
from myrl import DiscretizerFactory, QTable, plotrl

parser = ArgumentParser('Lunar lander RL playground')
parser.add_argument(
    'N', nargs='?', help='Number of experiments to run', default=10, type=int)
parser.add_argument('--eps', default=0.25, type=float,
                    help='Random action probability (def:0.25)')
parser.add_argument('--alpha', default=0.5, type=float,
                    help='Learning parameter (def:0.5)')
parser.add_argument('--gamma', default=0.5, type=float,
                    help='Reward parameter (def:0.5)')
parser.add_argument('--render-mode', default=None, help='gym render mode')
parser.add_argument('--fn', help='filename for load/save',
                    default='lander.txt')
parser.add_argument('--load', action='store_true', help='load from fn')
parser.add_argument('--save', action='store_true', help='load to fn')


if __name__ == '__main__':
    args = parser.parse_args()

    # Setup
    env = gym.make("LunarLander-v2", render_mode=args.render_mode)
    ds = DiscretizerFactory([
        {-0.1: '<', 0: '<', 0.1: '>'},  # x
        {0: '<', 0.1: '<', 1: '<'},  # y
        {-1: '<', -0.1: '<', 1: '>', 0.1: '>'},  # vx
        {-1: '<', -0.1: '<', 1: '>', 0.1: '>'},  # vy
        {v: '<' for v in linspace(0.1, pi, 4)},  # theta
        {-1: '<', -0.1: '<', 1: '>', 0.1: '>'}  # omega
    ])
    if args.load:
        policy = QTable(ds.n, 4, env.action_space.sample, ds)
        policy.load(args.fn)
    else:
        policy = QTable(ds.n, 4, env.action_space.sample, ds,
                        eps=args.eps, alpha=args.alpha, gamma=args.gamma)

    # Experiment
    olog, rlog, alog = [], [], []
    state, info = env.reset(seed=42)
    seed(0)
    for _ in range(args.N):
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

    # End
    if args.save:
        policy.save(args.fn)
    olog = vstack(olog).T
    plotrl(olog, rlog, alog)
    plt.show()
