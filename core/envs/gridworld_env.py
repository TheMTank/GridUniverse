import sys

import numpy as np
from six import StringIO
import gym
from gym import spaces


class GridWorldEnv(gym.Env):
    metadata = {'render.modes': ['human', 'ansi', 'graphic']}

    def __init__(self, grid_shape=(4, 4), initial_state=0, **kwargs):
        # set state space params
        self.x_max = grid_shape[0]
        self.y_max = grid_shape[1]
        self.world = self._generate_world()
        # set action space params
        self.action_space = spaces.Discrete(4)
        # main boundary check for edges of map done here
        self.action_state_to_next_state = [lambda s: s if self.world[s][1] == (self.y_max - 1) else s + 1,
                                           lambda s: s if self.world[s][0] == (self.x_max - 1) else s + self.y_max,
                                           lambda s: s if self.world[s][1] == 0 else s - 1,
                                           lambda s: s if self.world[s][0] == 0 else s - self.y_max]
        self.action_descriptors = ['up', 'right', 'down', 'left']
        # set observed params: [current state, world state]
        self.observation_space = spaces.Discrete(self.world.size)
        # set initial state for the agent
        self.previous_state = self.current_state = self.initial_state = initial_state
        # set terminal state(s) and wall(s)
        self.terminal_states = kwargs['terminal_states'] if 'terminal_states' in kwargs else [self.world.size - 1]
        # kwargs['walls'] = [1, 4, 14] # uncomment for quick test
        # need index positioning for efficient check in _is_valid()
        # but also need list to easily access each wall sequentially (e.g in render())
        self.wall_indices = []
        self.walls = np.zeros(self.world.shape)
        if 'walls' in kwargs:
            for state_index in kwargs['walls']:
                self.walls[state_index] = 1
                self.wall_indices.append(state_index)
        # set reward matrix
        self.reward_matrix = np.full(self.world.shape, -1)
        for terminal_state in self.terminal_states:
            self.reward_matrix[terminal_state] = 0
        # self.reward_range = [-inf, inf] # default values already
        # set additional parameters for the environment
        self.done = False
        self.info = {}

    def _generate_world(self):
        """
        Creates and returns the gridworld map as a numpy array.

        The states are defined by their index and contain a tuple of uint16 values that represent the
        coordinates (x,y) of a state in the grid.
        """
        world = np.fromiter(((x, y) for x in np.nditer(np.arange(self.x_max))
                             for y in np.nditer(np.arange(self.y_max))), dtype='int64, int64')
        return world

    def look_step_ahead(self, state, action):
        """
        Computes the results of a hypothetical action taking place at the given state.

        Returns the state to what that action would lead, the reward at that new state and a boolean value that
        determines if the next state is terminal
        """
        if self.is_terminal(state):
            next_state = state
        else:
            next_state = self.action_state_to_next_state[action](state)
            next_state = next_state if self._is_valid(next_state) else state

        return next_state, self.reward_matrix[next_state], self.is_terminal(next_state)

    def _is_valid(self, state):
        """
        Checks if a given state is a wall or any other element that shall not be trespassed.
        """
        if self.walls[state] == 1:
            return False
        return True

    def is_terminal(self, state):
        """
        Check if the input state is terminal.
        """
        if state in self.terminal_states:
            return True
        return False

    def _step(self, action):
        """
        Moves the agent one step according to the given action.
        """
        self.previous_state = self.current_state
        self.current_state, reward, self.done = self.look_step_ahead(self.current_state, action)
        return self.current_state, reward, self.done, self.info

    def _reset(self):
        self.done = False
        self.current_state = self.previous_state = self.initial_state
        return self.current_state

    def _render(self, mode='human', close=False):
        new_world = np.fromiter(('o' for _ in np.nditer(np.arange(self.x_max))
                                 for _ in np.nditer(np.arange(self.y_max))), dtype='S1')
        new_world[self.current_state] = 'x'
        for t_state in self.terminal_states:
            new_world[t_state] = 'T'

        for w_state in self.wall_indices:
            new_world[w_state] = '#'

        if mode == 'human' or mode == 'ansi':
            outfile = StringIO() if mode == 'ansi' else sys.stdout
            for row in np.reshape(new_world, (self.x_max, self.y_max))[:, ::-1].T:
                for state in row:
                    outfile.write((state.decode('UTF-8') + ' '))
                outfile.write('\n')
            outfile.write('\n')
            return outfile

        elif mode == 'graphic':
            raise NotImplementedError
        else:
            super(GridWorldEnv, self).render(mode=mode)

    def _close(self):
        pass

    def _seed(self, seed=None):
        raise NotImplementedError
