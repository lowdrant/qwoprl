#!/usr/bin/env python3
"""
Lunar Lander gym playground
"""
from argparse import ArgumentParser

import matplotlib.pyplot as plt
from numpy import linspace, pi
from numpy.random import seed

import gymnasium as gym
from myrl import DQN, DQNOptimizer, ReplayMemory, plotrl
from torch import optim


def create_parser():
    """Create CLI. Functionalized for code folding."""
    parser = ArgumentParser('Lunar lander RL playground')
    parser.add_argument(
        'N', nargs='?', help='number of experiments to run', default=10, type=int)
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
    return parser


if __name__ == '__main__':
    parser = create_parser()
    args = parser.parse_args()

    # Setup
    env = gym.make("LunarLander-v2", render_mode=args.render_mode)
    state, info = env.reset()
    policy = DQN(len(state), env.action_space.n, env.action_space.sample)

    optimizer = DQNOptimizer(env,
                             optim.AdamW(policy.parameters(),
                                         lr=1e-4, amsgrad=True),
                             ReplayMemory(10000), policy)
    optimizer(10)

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

    plt.ioff()
    plt.show()
