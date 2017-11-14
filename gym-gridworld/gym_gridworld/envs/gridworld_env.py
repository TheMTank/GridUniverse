import gym
from gym import error, spaces, utils
from gym.utils import seeding

import numpy as np
import sys
from six import StringIO


class GridWorldEnv(gym.Env):
    metadata = {'render.modes': ['human', 'ansi', 'graphic']}

    def __init__(self, x_y_dim=4):
        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Tuple((spaces.Discrete(x_y_dim), spaces.Discrete(x_y_dim)))
        # self.reward_range = None  # by default [-inf, inf]

        self.x_y_dim = x_y_dim
        self.done = False
        self.curr_state = [0, 0]
        self.terminal_state = (self.x_y_dim-1, self.x_y_dim-1)
        self.info = {}

    def _step(self, action):
        action_str = ACTION_MEANING[action]
        if action_str == "UP":
            self.curr_state[0] -= 1
        elif action_str == "DOWN":
            self.curr_state[0] += 1
        elif action_str == "LEFT":
            self.curr_state[1] -= 1
        elif action_str == "RIGHT":
            self.curr_state[1] += 1

        if self.curr_state[0] < 0:
            self.curr_state[0] = 0
        elif self.curr_state[0] > self.x_y_dim:
            self.curr_state[0] = self.x_y_dim
        if self.curr_state[1] < 0:
            self.curr_state[1] = 0
        elif self.curr_state[1] > self.x_y_dim:
            self.curr_state[1] = self.x_y_dim

        observation = self.curr_state
        reward = 0
        if self.curr_state[0] == self.terminal_state[0] and self.curr_state[1] == self.terminal_state[1]:
            reward = 10
            self.done = True

        return observation, reward, self.done, self.info

    def _reset(self):
        self.done = False
        self.curr_state = [0, 0]
        self.terminal_state = (self.x_y_dim-1, self.x_y_dim-1)
        return self.curr_state

    def _render(self, mode='human', close=False):
        if mode == 'human' or mode == 'ansi':
            world = [['o' for _ in range(self.x_y_dim)] for _ in range(self.x_y_dim)]
            world[self.curr_state[0]][self.curr_state[1]] = 'x'
            world[self.terminal_state[0]][self.terminal_state[1]] = 'T'

            outfile = StringIO() if mode == 'ansi' else sys.stdout
            for row in world:
                for cell in row:
                    outfile.write((cell + ' '))
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
    env = GridWorldEnv()
    _, _, _, _ = env.step(1)
    out = env.render()
    print(out)
    pass
    # from random import randint
    # env = GridWorld()
    # for i_episode in range(1):
    #     observation = env.reset()
    #     for t in range(100):
    #         env.render()
    #         # print(observation)
    #         action = randint(0, 3)
    #         observation, reward, done, info = env.step(action)
    #
    #         if done:
    #             print("Episode finished after {} timesteps".format(t + 1))
    #             env.render()
    #             break
