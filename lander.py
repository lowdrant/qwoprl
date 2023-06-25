#!/usr/bin/env python3
"""
Lunar Lander gym playground
"""
from argparse import ArgumentParser

import matplotlib.pyplot as plt
from numpy import linspace, pi
from numpy.random import seed

import gymnasium as gym
from myrl import DQN, DQNOptimizer, ReplayMemory
from torch import argmax, optim, tensor


def create_parser():
    """Create CLI. Functionalized for code folding."""
    parser = ArgumentParser('Lunar lander RL playground')
    parser.add_argument(
        'N', nargs='?', help='number of experiments to run (epochs if --save-epochs)', default=10, type=int)
    parser.add_argument('--eps', default=0.25, type=float,
                        help='random action probability (def:0.25)')
    parser.add_argument('--alpha', default=0.5, type=float,
                        help='learning rate (def:0.5)')
    parser.add_argument('--gamma', default=0.5, type=float,
                        help='discount parameter (def:0.5)')
    parser.add_argument('--render-mode', default=None, help='gym render mode')
    parser.add_argument('--load-fn', help='filename for load. Default: lander.txt',
                        default='lander.txt')
    parser.add_argument('--save-fn', help='filename for save. Default: lander.txt',
                        default='lander.txt')
    parser.add_argument('--load', action='store_true',
                        help='load from fn flag')
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
    parser.add_argument('--demo', action='store_true',
                        help='run net instead of training')
    return parser


def train(policy, args):
    """train wrapper - policy net, cli args"""
    optimizer = DQNOptimizer(env,
                             optim.AdamW(policy.parameters(),
                                         lr=1e-4, amsgrad=True),
                             ReplayMemory(10000), policy)

    if args.save_epochs:
        import os
        edir = 'epoch'
        if not os.path.exists(edir):
            os.mkdir(edir)
        plt.ion()
        avg_len = max(2, args.epoch_size // 5)
        for i in range(1, args.N + 1):
            optimizer(args.epoch_size, avg_len=avg_len)
            fn = f'{edir}/lander_{i}.torch'
            policy.save(fn)
            print(f'Epoch {i} complete -- saved to {fn}')
        print('Complete!')

    if args.save:
        policy.save(args.save_fn)
        print(f'Saved NN to {args.save_fn}')

    plt.ioff()
    plt.show()


if __name__ == '__main__':
    parser = create_parser()
    args = parser.parse_args()

    # Setup
    env = gym.make("LunarLander-v2", render_mode=args.render_mode)
    state, info = env.reset()
    policy = DQN(len(state), env.action_space.n, env.action_space.sample)

    # Load
    if args.load:
        policy.load(args.load_fn)

    # Demo v. Train
    if not args.demo:
        train(policy, args)
    else:
        state, _ = env.reset()
        N = args.N * args.epoch_size if args.save_epochs else args.N
        for i in range(N):
            state = tensor(state).unsqueeze(0)
            action = policy(state).max(1)[1].view(1, 1)
            next_state, reward, term, trunc, _ = env.step(action.item())
            next_state = None if term else next_state
            state = next_state
            if term or trunc:
                state, _ = env.reset()
