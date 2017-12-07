import gym
from gym import error, spaces, utils
from gym.utils import seeding

import numpy as np
import sys
from six import StringIO


class GridWorldEnv(gym.Env):
    metadata = {'render.modes': ['human', 'ansi', 'graphic']}

    def __init__(self, grid_shape=(4, 4), initial_state=0, **kwargs):
        # set state space params
        self.x_max = grid_shape[0]
        self.y_max = grid_shape[1]
        self.world = self._generate_world()
        # set action space params
        self.action_space = spaces.Discrete(4)
        self.actions_list = np.array([(0, 1), (1, 0), (0, -1), (-1, 0)], dtype='int16, int16')
        self.action_descriptors = ['up', 'right', 'down', 'left']
        # set observed params: [current state, world state]
        self.observation_space = spaces.Box(spaces.Discrete(self.world.size),
                                            spaces.Box(spaces.Discrete(self.x_max), spaces.Discrete(self.y_max)))
        # set initial state for the agent
        self.previous_state = self.current_state = initial_state
        # set terminal state(s)
        self.terminal_states = kwargs['terminal_states'] if 'terminal_states' in kwargs else [self.world.size - 1]
        # set reward matrix
        self.reward_matrix = np.full(self.world.shape, -1)
        for terminal_state in self.terminal_states:
            self.reward_matrix[terminal_state] = 0
        # self.reward_range = [-inf, inf] # default values already
        # set additional parameters for the environment
        self.done = False
        self.info = {}
        # create first render with the initial status of the world

        # np.nditer(grid, flags=['multi_index'])

    def look_step_ahead(self, state, action):
        if self._is_terminal(state):
            next_state = state
        else:
            state_x, state_y = self.world[self.current_state]
            movement_x, movement_y = self.actions_list[action]
            next_location = np.array((state_x + movement_x, state_y + movement_y), dtype='int16, int16')
            next_state = np.where(self.world == next_location)[0][0]

        if not self._is_valid(next_state):
            next_state = state

            return next_state, self.reward_matrix[state]

    def _is_valid(self, state):
        """
        Checks if a given state is inside the grid. In the future it could also check for invalid spaces
        inside the grid.
        """
        if 0 <= state < self.world.size:
            return True
        return False

    def _is_terminal(self, state):
        """
        Check if the input state is terminal.
        """
        for t_state in self.terminal_states:
            if state == t_state:
                return True
        return False

    def _generate_world(self):
        """
        Creates and returns the gridworld map as a numpy array.

        The states are defined by their index and contain a tuple of uint16 values that represent the
        coordinates (x,y) of a state in the grid.
        """
        world = np.fromiter(((x, y) for x in np.nditer(np.arange(self.x_max))
                             for y in np.nditer(np.arange(self.y_max))), dtype='int16, int16')
        return world

    def _update_world(self):
        """
        Updates the status of the world after a single step
        """
        self.world[self.prev_state[0]][self.prev_state[1]] = 'o'
        self.world[self.curr_state[0]][self.curr_state[1]] = 'x'

    def _step(self, action):
        action_str = self.ACTION_MEANING[action]
        self.prev_state = self.curr_state.copy()
        # TODO: Remap states to a grid for calculating next state
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
        # TODO: Observation should be also the world state
        observation = self.curr_state

        if self._is_terminal(self.curr_state):
            self.done = True

        reward = self.reward_matrix[self.curr_state]
        return observation, reward, self.done, self.info

    def _reset(self):
        self.done = False
        self.curr_state = 0
        self.terminal_state = (self.x_max-1, self.y_max-1)
        return self.curr_state

    def _render(self, mode='human', close=False):
        pass

        # new_world = [['o' for _ in range(self.x_max)] for _ in range(self.y_max)]
        # new_world[self.curr_state[0]][self.curr_state[1]] = 'x'
        # for t_state in self.terminal_states:
        #     new_world[t_state[0]][t_state[1]] = 'T'

        # if mode == 'human' or mode == 'ansi':
        #     outfile = StringIO() if mode == 'ansi' else sys.stdout
        #     for row in self.world:
        #         for cell in row:
        #             outfile.write((cell + ' '))
        #         outfile.write('\n')
        #     outfile.write('\n')
        #     return outfile
        #
        # elif mode == 'graphic':
        #     raise NotImplementedError
        # else:
        #     super(GridWorldEnv, self).render(mode=mode)

    def _close(self):
        pass

    def _seed(self, seed=None):
        raise NotImplementedError


if __name__ == '__main__':
    gw_env = GridWorldEnv()
    policy0 = np.ones([gw_env.world.size, len(gw_env.actions_list)]) / len(gw_env.actions_list)
    v0 = np.zeros(gw_env.world.size)


    def policy_eval(policy, env, discount_factor=1.0, threshold=0.00001, **kwargs):
        if 'value_function' in kwargs:
            v = kwargs['value_function']
        else:
            v = np.zeros(env.world.size)

        while True:
            delta = 0
            for state in range(env.world.size):
                v_update = 0
                for action, action_prob in enumerate(policy[state]):
                    for next_state, reward in env.look_step_ahead(state, action):
                        v_update += action_prob * (reward + discount_factor * v[next_state])
                delta = max(delta, np.abs(v - v[state]))
                v[state] = v_update
            if delta < threshold:
                break
        return v

    v1 = policy_eval(policy0, gw_env, value_function=v0)

    def policy_improvement(policy, env, discount_factor=1.0):
        while True:
            v = policy_eval(policy, env, discount_factor)
            policy_stable = True

            for state in range(env.world.size):
                chosen_action = np.argmax(policy[state])
                action_values = np.zeros(env.action_space_size)
                for action in range(env.action_space_size):
                    for next_state, reward in env.look_step_ahead(state, action):
                        action_values[action] += reward + discount_factor * v[next_state]
                best_action = np.argmax(action_values)

                if chosen_action != best_action:
                    policy_stable = False
                policy[state] = np.eye(env.action_space_size)[best_action]

            if policy_stable:
                return policy, v

    policy1 = policy_improvement(policy0, gw_env)
    v2 = policy_eval(policy1, gw_env, value_function=v1)
    # and keep alternating between policy evaluation and improvement until convergence


    # for i_episode in range(1):
    #     observation = env.reset()
    #     for t in range(100):
    #         env.render()
    #         # print(observation)
    #         action = env.action_space.sample()
    #         print(ACTION_MEANING[action])
    #         observation, reward, done, info = env.step(action)
    #
    #         if done:
    #             print("Episode finished after {} timesteps".format(t + 1))
    #             break
