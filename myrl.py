
#!/usr/bin/env python3
"""
My reinforcement learning helpers

Provides:
    plotrl -- plotting helper for rl behavior over time
    QTable -- QTable implementation
    DiscretizerFactory -- takes hashmaps describing discretization and
                          returns a callable for discretizing the state
"""
from collections import deque, namedtuple
from copy import deepcopy
from itertools import count
from itertools import product as iterprod
from random import sample

import matplotlib.pyplot as plt
from matplotlib import get_backend as get_mpl_backend
from matplotlib.gridspec import GridSpec
from numpy import array_equal, exp, loadtxt, savetxt
from numpy.random import rand

import torch
import torch.nn.functional as F
from torch import argmax, cat, float32, nn, no_grad, tensor, zeros

__all__ = ['plotrl', 'QTable', 'DiscretizerFactory',
           'DQN', 'DQNOptimizer', 'ReplayMemory']
is_ipython = 'inline' in get_mpl_backend()
try:
    from IPython import display
except ImportError:
    pass


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
    fig.clf()
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
        alpha -- learning rate
        gamma -- discount parameter
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

    def save(self, fn):
        """Save QTable to raw text
            INPUTS:
                fn -- filename to save
        """
        header = f'eps={self.eps}\nalpha={self.alpha}\ngamma={self.gamma}'
        savetxt(fn, self.tabletable, header=header)

    def load(self, fn, load_params=False):
        """Load QTable from raw text
            INPUTS:
                fn -- filename to load
                load_params -- load eps,alpha,gamma from file. Default: False
        """
        self.table = loadtxt(fn)
        if load_params:
            with open(fn, 'r') as f:
                self.eps = float(f.readline().split('=')[-1])
                self.alpha = float(f.readline().split('=')[-1])
                self.gamma = float(f.readline().split('=')[-1])


class DiscretizerFactory:
    """Construct state discretization function from hashmaps which describe
        thresholds. Intended for use with Q Tables.

        INPUTS:
            qdicts -- iterable of dictionaries

        EXAMPLES:
            Let's grid out the X Axis into Left (<-1), Right (>1), and Center
            (else) and grid out the Y Axis into Top (>1), Bottom (<-1), and
            Center (else):
                >>> ds = DiscretizerFactory([{-1: '<', 1: '>'},
                                             {-1: '<', 1: '>'}])
                >>> ds([1.5, 0])
                    -> 2
                >>> ds.n  # number of discrete states
                    -> 9

            The dicts are evaluated in-order, so we can refine the grid
            (effectively using elifs) when ordered correctly:
                >>> ds2 = DiscretizerFactory([
                        {-1: '<', -0.1: '<', 1: '>', 0.1: '>'},
                        {-1: '<', -0.1: '<', 1: '>', 0.1: '>'}])
                >>> ds2([1.5, 0])
                    -> 3
                >>> ds2.n  # number of discrete states
                    -> 25
    """

    def __init__(self, qdicts):
        for i, d in enumerate(qdicts):
            assert len(d) > 0, f'Empty state hashmap at qdicts[{i}]'
        self.qdicts = deepcopy(qdicts)

    def __call__(self, state):
        ret, delta_r = 0, 1
        for i, x in enumerate(state):
            breakflag = False  # False <-> `else` <-> 0
            for j, (k, v) in enumerate(self.qdicts[i].items()):
                if self._eval(x, k, v):
                    breakflag = True
                    j += 1  # never let j = 0 on a break
                    break
            if breakflag:
                ret += j * delta_r
            delta_r *= len(self.qdicts[i]) + 1  # +1 to avoid overlap
        return ret

    @staticmethod
    def _eval(x, k, v):
        """Return output of given comparison operation
            INPUTS:
                x -- value
                k -- comparison value
                v -- comparison operator

            EXAMPLES:
                >>> _eval(1, 5, '<')
                    -> True
                >>> _eval(1, 1, '>')
                    -> False
        """
        if v == '>':
            return x > k
        if v == '>=':
            return x >= k
        if v == '<':
            return x < k
        if v == '<=':
            return x <= k
        if v == '==':
            return x == k
        raise RuntimeError(f'Invalid operator: {v}')

    @property
    def n(self):
        """Count how many discrete states are described by self.qdicts"""
        n = 1
        for d in self.qdicts:
            n *= len(d) + 1
        return n


def test_DiscretizerFactory():
    """Test DiscretizerFactory by gridding out the XY plane"""
    ds = DiscretizerFactory([{-1: '<', 1: '>'}, {-1: '<', 1: '>'}])
    a = [-1.5, 0, 1.5]
    b = [ds(x) for x in iterprod(a, a)]
    assert array_equal(list(set(b)), sorted(b)), 'bad discretization'
    assert ds.n == 9, 'bad size'

    ds2 = DiscretizerFactory([{-1: '<', -0.1: '<', 1: '>', 0.1: '>'},
                              {-1: '<', -0.1: '<', 1: '>', 0.1: '>'}])
    a = [-1.5, -1, 0, 1, 1.5]
    b = [ds2(x) for x in iterprod(a, a)]
    assert array_equal(list(set(b)), sorted(b)), 'bad fine discretization'
    assert ds2.n == 25, 'bad fine size'


class DQN(nn.Module):
    """Deep Q-Learning implementation
        https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html

        INPUTS:
            n -- int -- state space size
            m -- int -- action space size
            random_action -- callable -- pick a random action
            eps -- float -- probability of choosing random action over optimal
    """

    def __init__(self, n, m, random_action):
        assert callable(random_action)
        self.random_action = random_action
        super().__init__()

        connection_size = 128
        self.input_layer = nn.Linear(n, connection_size)
        self.connecting_layer = nn.Linear(connection_size, connection_size)
        self.output_layer = nn.Linear(connection_size, m)

    def forward(self, x):
        x = F.relu(self.input_layer(x))
        x = F.relu(self.connecting_layer(x))
        return self.output_layer(x)

    def save(self, fn):
        """save state_dict to fn"""
        torch.save(self.state_dict, fn)

    def load(self, fn):
        """load state_dict from fn"""
        self.load_state_dict(torch.load(fn))

    # def __call__(self, x, eps=0):
    #     if rand() < eps:
    #         return self.random_action()
    #     return argmax(self.forward(x))


Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))


class ReplayMemory:
    """https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html#replay-memory"""

    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        """Save a transition"""
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        """pick something"""
        return sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


class DQNOptimizer:
    """
    https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html

        INPUTS:

        KWARGS:
            epsfun -- callable -- Training iteration -> probability of choosing random actions
            eps_start -- float -- initial probability of choosing random action
            eps_end -- float -- final probability of choosing random action
            eps_decay -- float -- exponenital decay rate of epsilon
    """

    def __init__(self, env, optimizer, memory, target_net, **kwargs):
        self.optimizer = optimizer
        self.env = env
        self.target_net = target_net
        self.policy_net = deepcopy(target_net)
        self.optimizer = deepcopy(optimizer)
        self.memory = deepcopy(memory)

        self._epsfun = kwargs.get('epsfun', self._epsfun_default)
        self.eps_start = kwargs.get('eps_start', 0.9)
        self.eps_end = kwargs.get('eps_end', 0.05)
        self.eps_decay = kwargs.get('eps_decay', 1000)

        self.BATCH_SIZE = kwargs.get('batch_size', 128)
        self.GAMMA = kwargs.get('gamma', 128)
        self.TAU = kwargs.get('tau', 128)
        self.BATCH_SIZE = kwargs.get('batch_size', 128)

        device_str = 'cpu'
        if kwargs.get('try_cuda', False) and torch.cuda.is_available():
            device_str = 'cuda'
        self.device = torch.device(device_str)

        self.episode_durations = []
        self.lines = None

    def _epsfun_default(self, trial_num):
        """Training iteration -> probability of choosing random action
            This is the default implementation, which exponentially decays the
            probability of selecting a random action as the trial number
            increases,

            INPUTS:
                trial_num -- int -- trial number
            OUTPUTS:
                eps -- float -- probability of
        """
        return self.eps_end + (self.eps_start - self.eps_end) * exp(-trial_num / self.eps_decay)

    def _tensor_unsqueeze(self, x, dtype=float32):
        """Perform `tensor(x, dtype=dtype, device=self.device).unsqueeze(0)`"""
        return tensor(x, dtype=dtype, device=self.device).unsqueeze(0)

    def optimization_step(self):
        if len(self.memory) < self.BATCH_SIZE:
            return
        transitions = self.memory.sample(self.BATCH_SIZE)
        batch = Transition(*zip(*transitions))  # transpose

        # Compute a mask of non-final states and concatenate the batch elements
        # (a final state would've been the one after which simulation ended)
        non_final_mask = tensor(tuple(map(lambda s: s is not None,
                                          batch.next_state)), device=self.device, dtype=torch.bool)
        non_final_next_states = cat([s for s in batch.next_state
                                     if s is not None])
        state_batch = cat(batch.state)
        action_batch = cat(batch.action)
        reward_batch = cat(batch.reward)

        # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
        # columns of actions taken. These are the actions which would've been taken
        # for each batch state according to policy_net
        state_action_values = self.policy_net(
            state_batch).gather(1, action_batch)

        # Compute V(s_{t+1}) for all next states.
        # Expected values of actions for non_final_next_states are computed based
        # on the "older" target_net; selecting their best reward with max(1)[0].
        # This is merged based on the mask, such that we'll have either the expected
        # state value or 0 in case the state was final.
        next_state_values = zeros(self.BATCH_SIZE, device=self.device)
        with no_grad():
            next_state_values[non_final_mask] = self.target_net(
                non_final_next_states).max(1)[0]
        # Compute the expected Q values
        expected_state_action_values = (
            next_state_values * self.GAMMA) + reward_batch

        # Optimization
        criterion = nn.SmoothL1Loss()
        loss = criterion(state_action_values,
                         expected_state_action_values.unsqueeze(1))
        self.optimizer.zero_grad()
        loss.backward()
        # - in-place gradient clipping
        torch.nn.utils.clip_grad_value_(self.policy_net.parameters(), 100)
        self.optimizer.step()

    def update_weights(self):
        """Soft update of the target weights"""
        target_net_state_dict = self.target_net.state_dict()
        policy_net_state_dict = self.policy_net.state_dict()
        for key in policy_net_state_dict:
            target_net_state_dict[key] = policy_net_state_dict[key] * \
                self.TAU + target_net_state_dict[key] * (1 - self.TAU)
        self.target_net.load_state_dict(target_net_state_dict)

    def optimize(self, num_episodes, fignum, avg_len):
        for i_episode in range(num_episodes):
            state, _ = self.env.reset()
            state = self._tensor_unsqueeze(state)

            # Run Single Episode
            for t in count():
                # Run Time Step
                action = self._select_action(state, i_episode)
                next_state, reward, term, trunc, _ = self.env.step(
                    action.item())
                reward = tensor([reward], device=self.device)
                next_state = None if term else self._tensor_unsqueeze(
                    next_state)
                self.memory.push(state, action, next_state, reward)
                state = next_state

                # Optimize
                self.optimization_step()
                self.update_weights()

                if term or trunc:
                    break

            # Epsiode End
            self.episode_durations.append(t + 1)
            self._plot_durations(False, fignum, avg_len)

        self.episode_durations.append(t + 1)

    def _select_action(self, state, n):
        """select action"""
        if rand() > self._epsfun(n):
            with torch.no_grad():
                return self.policy_net(state).max(1)[1].view(1, 1)
        return torch.tensor([[self.env.action_space.sample()]], device=self.device, dtype=torch.long)

    def __call__(self, num_episodes=100, fignum=1, avg_len=100, show_result=False):
        # avg_len -> length of running average
        self.optimize(num_episodes, fignum, avg_len)
        self._plot_durations(show_result, fignum, avg_len)

    def _plot_durations(self, show_result, num, avg_len):
        # Computation
        durations_t = torch.tensor(self.episode_durations, dtype=torch.float)
        if len(durations_t) >= avg_len:
            means = durations_t.unfold(0, avg_len, 1).mean(1).view(-1)
            means = cat((zeros(avg_len - 1), means))

        # Plotting
        fig = plt.figure(num)
        if show_result:
            plt.title('Result')
        print(self.lines)
        if self.lines is None:
            plt.title('Training...')
            plt.xlabel('Episode')
            plt.ylabel('Duration')
            line, = plt.plot(durations_t.numpy())
            self.lines = [line]
        elif len(self.lines) == 1:
            self.lines[0].set_data(range(len(durations_t)), durations_t)
            if len(durations_t) > avg_len:
                line, = plt.plot(means.numpy())
                self.lines.append(line)
        else:
            self.lines[0].set_data(range(len(durations_t)), durations_t)
            self.lines[1].set_data(range(len(means)), means)

        # Update
        ax = fig.gca()
        fig.gca().set_xlim((0, len(durations_t)))
        ax.relim()
        ax.autoscale(True)
        fig.canvas.draw()
        fig.canvas.flush_events()
        plt.pause(0.001)  # pause a bit so that plots are updated
        # if is_ipython:
        #     if not show_result:
        #         display.display(plt.gcf())
        #         display.clear_output(wait=True)
        #     else:
        #         display.display(plt.gcf())

        if show_result:
            plt.ioff()


if __name__ == '__main__':
    print('Running unit tests...')
    test_DiscretizerFactory()
