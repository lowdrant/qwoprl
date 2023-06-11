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
parser.add_argument('--load-fn', help='filename for load. Default: lander.txt',
                    default='lander.txt')
parser.add_argument('--save-fn', help='filename for save. Default: lander.txt',
                    default='lander.txt')
parser.add_argument('--load', action='store_true', help='load from fn flag')
parser.add_argument('--save', action='store_true', help='save to fn flag')
parser.add_argument('--overwrite', action='store_true',
                    help='overwrite save file flag')
parser.add_argument('--load-params', action='store_true',
                    help='load params from fn flag')
parser.add_argument('--plot', action='store_true', help='plot data flag')
parser.add_argument('--epoch-size', default=10000, type=int,
                    help='number of experiments per epoch')
parser.add_argument('--save-epochs', action='store_true',
                    help='save epoch progress flag')

if __name__ == '__main__':
    args = parser.parse_args()

    # Setup
    env = gym.make("LunarLander-v2", render_mode=args.render_mode)
    ds = DiscretizerFactory([
        {-0.1: '<', 0: '<', 0.1: '>'},  # x
        {0: '<', 0.1: '<', 1: '<'},  # y
        {-1: '<', -0.1: '<', 1: '>'},  # vx
        {-1: '<', -0.1: '<', 1: '>'},  # vy
        {v: '<' for v in linspace(0.1, pi, 4)},  # theta
        {-1: '<', -0.1: '<', 1: '>'}  # omega
    ])
    policy = QTable(ds.n, 4, env.action_space.sample, ds)
    num_epoch = 0  # track epoch number
    if args.load:
        policy.load(args.load_fn, args.load_params)
        split = args.load_fn.split('_epoch')
        epoch_basefn = split[0]
        if len(split) > 1:
            num_epoch = int(split[1].rsplit('.')[0])
    else:
        policy = QTable(ds.n, 4, env.action_space.sample, ds,
                        eps=args.eps, alpha=args.alpha, gamma=args.gamma)
        epoch_basefn = args.save_fn.rsplit('.')[0]

    # Experiment
    olog, rlog, alog = [], [], []
    state, info = env.reset(seed=42)
    seed(0)
    for n in range(args.N):
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

        if args.save_epochs:
            if (n + 1) % args.epoch_size == 0:
                num_epoch += 1
                policy.save(epoch_basefn + f'_epoch{num_epoch}.txt')
                print(f'Saved epoch {num_epoch}')

    env.close()

    # End
    if args.save:
        fn = args.save_fn
        if not args.overwrite:  # test file existence
            found_fn = False
            n = 1
            while not found_fn:
                try:
                    open(fn, 'r').close()
                except FileNotFoundError:
                    found_fn = True
                else:
                    print(f"'{fn}' already exists! incrementing number")
                    fn = args.save_fn.rsplit(
                        '.')[0] + f'_{n}.' + fn.rsplit('.')[-1]
                    n += 1
        policy.save(fn)

    if args.plot:
        olog = vstack(olog).T
        plotrl(olog, rlog, alog)
        plt.show()
