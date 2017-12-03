import gym
from gym import error, spaces, utils
from gym.utils import seeding

import numpy as np
import sys
from six import StringIO


class GridWorldEnv(gym.Env):
    metadata = {'render.modes': ['human', 'ansi', 'graphic']}

    def __init__(self, grid_shape=(4, 4), initial_state=(0, 0), **kwargs):
        # set state space params
        self.state_space_size = int(np.prod(grid_shape))
        self.state_space = np.arange(self.state_space_size)  # .reshape(grid_shape)
        self.x_max = grid_shape[0]
        self.y_max = grid_shape[1]
        # set action space params
        self.action_space = spaces.Discrete(4)
        self.action_space_size = int(np.prod(self.action_space.shape))
        self.ACTION_MEANING = ['UP', 'RIGHT', 'DOWN', 'LEFT']
        # set observed params: [current state, world state]
        self.observation_space = spaces.Box(spaces.Discrete(self.state_space_size),
                                            spaces.Box(spaces.Discrete(self.y_max), spaces.Discrete(self.y_max)))
        # set initial state for the agent
        self.prev_state = list(initial_state)
        self.curr_state = self.prev_state[:]
        # set terminal state(s)
        if 'terminal_states' in kwargs:
            self.terminal_states = kwargs['terminal_states']
        else:
            self.terminal_states = [self.state_space[-1]]
        # set reward matrix
        self.reward_matrix = np.full(self.state_space_size, -1)
        for terminal_state in self.terminal_states:
            self.reward_matrix[terminal_state] = 0
        # self.reward_range = [-inf, inf] # default values already
        # set additional parameters for the environment
        self.done = False
        self.info = {}
        # create first render with the initial status of the world
        self.world = self._generate_world_render()

    def look_step_ahead(self, state, action):
        if self._is_terminal(state):
            next_state = state
        else:
            # TODO: change by actually looking for the next step. Extract reward from reward matrix
            next_state = 0

        return next_state, self.reward_matrix[state]

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

    def _generate_world_render(self):
        """
        Creates the gridworld map and places the agent and goal in their corresponding locations.
        """
        # TODO: New definition of states broke this function.
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


if __name__ == '__main__':
    gw_env = GridWorldEnv()
    policy0 = np.ones([gw_env.state_space_size, gw_env.action_space_size]) / gw_env.action_space_size
    v0 = np.zeros(gw_env.state_space_size)


    def policy_eval(policy, env, discount_factor=1.0, threshold=0.00001, **kwargs):
        if 'value_function' in kwargs:
            v = kwargs['value_function']
        else:
            v = np.zeros(env.state_space_size)

        while True:
            delta = 0
            for state in range(env.state_space_size):
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

            for state in range(env.state_space_size):
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
