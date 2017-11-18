import gym
from gym import error, spaces, utils
from gym.utils import seeding

import numpy as np
import sys
from six import StringIO


class GridWorldEnv(gym.Env):
    metadata = {'render.modes': ['human', 'ansi', 'graphic']}

    def __init__(self, grid_dim=(4, 4)):
        self.x_max = grid_dim[0]
        self.y_max = grid_dim[1]
        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Box(spaces.Discrete(self.x_max), spaces.Discrete(self.y_max))
        # self.reward_range = None  # by default [-inf, inf]

        self.done = False
        self.prev_state = [0, 0]
        self.curr_state = [0, 0]
        self.terminal_states = [(self.x_max-1, self.y_max-1), (self.x_max-1, 0)]
        self.info = {}
        self.world = self._generate_world()

    def _is_valid(self, state):
        """
        Checks if a given state is inside the grid. In the future it could also check for invalid spaces
        inside the grid.
        """
        if 0 <= state[0] < self.x_max and 0 <= state[1] < self.y_max:
            return True
        return False

    def _is_terminal(self, state):
        """
        Check if the input state is terminal.
        """
        for t_state in self.terminal_states:
            if tuple(state) == t_state:
                return True
        return False

    def _generate_world(self):
        """
        Creates the gridworld map and places the agent and goal in their corresponding locations.
        """
        new_world = [['o' for _ in range(self.x_max)] for _ in range(self.y_max)]
        new_world[self.curr_state[0]][self.curr_state[1]] = 'x'
        for t_state in self.terminal_states:
            new_world[t_state[0]][t_state[1]] = 'T'
        return new_world

    def _update_world(self):
        """
        Updates the status of the world after a single step
        """
        self.world[self.prev_state[0]][self.prev_state[1]] = 'o'
        self.world[self.curr_state[0]][self.curr_state[1]] = 'x'

    def _step(self, action):
        action_str = ACTION_MEANING[action]
        self.prev_state = self.curr_state.copy()
        if action_str == "UP":
            if self._is_valid((self.curr_state[0] - 1, self.curr_state[1])):
                self.curr_state[0] -= 1
        elif action_str == "DOWN":
            if self._is_valid((self.curr_state[0] + 1, self.curr_state[1])):
                self.curr_state[0] += 1
        elif action_str == "LEFT":
            if self._is_valid((self.curr_state[0], self.curr_state[1] - 1)):
                self.curr_state[1] -= 1
        elif action_str == "RIGHT":
            if self._is_valid((self.curr_state[0], self.curr_state[1] + 1)):
                self.curr_state[1] += 1
        self._update_world()

        observation = self.curr_state
        reward = -1

        if self._is_terminal(self.curr_state):
            reward = 10
            self.done = True

        return observation, reward, self.done, self.info

    def _reset(self):
        self.done = False
        self.curr_state = [0, 0]
        self.terminal_state = (self.x_max-1, self.y_max-1)
        return self.curr_state

    def _render(self, mode='human', close=False):
        if mode == 'human' or mode == 'ansi':
            outfile = StringIO() if mode == 'ansi' else sys.stdout
            for row in self.world:
                for cell in row:
                    outfile.write((cell + ' '))
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


ACTION_MEANING = ["UP", "RIGHT", "DOWN", "LEFT"]

if __name__ == '__main__':
    from random import randint
    env = GridWorldEnv()
    for i_episode in range(1):
        observation = env.reset()
        for t in range(100):
            env.render()
            # print(observation)
            action = env.action_space.sample()
            action = randint(0, 3)
            print(ACTION_MEANING[action])
            observation, reward, done, info = env.step(action)

            if done:
                print("Episode finished after {} timesteps".format(t + 1))
                break
