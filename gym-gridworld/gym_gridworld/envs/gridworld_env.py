import gym
from gym import error, spaces, utils
from gym.utils import seeding
from random import randint


class GridWorldEnv(gym.Env):
    metadata = {'render.modes': ['human', 'graphic']}

    def __init__(self, x_y_dim=3):
        self.action_space = spaces.Discrete(4)
        """
        UP = 0
        RIGHT = 1
        DOWN = 2
        LEFT = 3
        """
        self.observation_space = None
        self.reward_range = None # by default [-inf, inf]

        self.x_y_dim = x_y_dim
        self.done = False
        self.curr_state = [0, 0]
        self.terminal_state = (self.x_y_dim, self.x_y_dim)
        self.info = 'No info'

    def _step(self, action):

        if action == self.UP:
            self.curr_state[0] -= 1
        elif action == self.DOWN:
            self.curr_state[0] += 1
        elif action == self.LEFT:
            self.curr_state[1] -= 1
        elif action == self.RIGHT:
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
        self.terminal_state = (self.x_y_dim, self.x_y_dim)
        return self.curr_state

    def _render(self, mode='human', close=False):
        if mode == 'human':
            grid = [['o', 'o', 'o', 'o'], ['o', 'o', 'o', 'o'], ['o', 'o', 'o', 'o'], ['o', 'o', 'o', 'o']]

            grid[self.curr_state[0]][self.curr_state[1]] = 'x'
            grid[self.terminal_state[0]][self.terminal_state[1]] = 'T'

            for row in grid:
                for el in row:
                    print(el, end=' ')
                print()
            print()
        elif mode == 'graphic':
            #TODO: Rendering of the GridWorld in a window
            raise NotImplementedError
        else:
            super(GridWorld, self).render(mode=mode)

    def _close(self):
        raise NotImplementedError

    def _seed(self, seed=None):
        raise NotImplementedError


if __name__ == '__main__':
    pass
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
