import sys
import random
import time

import numpy as np
from six import StringIO
import gym
from gym import spaces
from gym.utils import seeding

from core.envs import maze_generation


class GridUniverseEnv(gym.Env):
    metadata = {'render.modes': ['human', 'ansi', 'graphic']}

    def __init__(self, grid_shape=(4, 4), *, initial_state=0, goal_states=None, lava_states=None, walls=None,
                 sensor_mode='current_state_index', custom_world_fp=None, random_maze=False):
        """
        The constructor for creating a GridUniverse environment. The default GridUniverse is a square grid of 4x4 where the
        agent starts in the top left corner and the terminal goal state is in the bottom right corner.

        :param grid_shape: Tuple of size 2 to specify (width, height) of grid
        :param initial_state: int for single initial state or list of possible states chosen uniform randomly
        "Terminal states". The episode ends if the agent reaches this type of state (done = True).
        :param goal_states: Terminal states with positive reward
        :param lava_states: Terminal states with negative reward
        :param walls: list of walls. These are blocked states where the agent can't enter/walk on
        :param sensor_mode: choose which type of observation returned by step function. Options are
                            ['current_state_index', 'whole_grid']
        :param custom_world_fp: optional parameter to create the grid from a text file.
        :param random_maze: optional parameter to randomly generate a maze from the algorithm within maze_generation.py
                            This will override the params initial_state, goal_states, lava_states,
                            walls and custom_world_fp params
        """
        # check state space params
        if goal_states is not None and not isinstance(goal_states, list):
            raise TypeError("goal_states parameter must be a list of integer indices")
        if lava_states is not None and not isinstance(lava_states, list):
            raise TypeError("lava_states parameter must be a list of integer indices")
        if walls is not None and not isinstance(walls, list):
            raise TypeError("walls parameter must be a list of integer indices")
        if not (isinstance(grid_shape, list) or isinstance(grid_shape, tuple)) or len(grid_shape) != 2 \
                or not isinstance(grid_shape[0], int) or not isinstance(grid_shape[1], int):
            raise TypeError("grid_shape parameter must be tuple/list of two integers")

        self.possible_sensor_modes = ['current_state_index', 'whole_grid']
        if sensor_mode not in self.possible_sensor_modes:
            raise TypeError("sensor_mode parameter must be one of {}".format(self.possible_sensor_modes))
        self.num_cols = grid_shape[0] # num columns
        self.num_rows = grid_shape[1] # num rows
        self.world = self._generate_world()
        # set action space params
        self.action_space = spaces.Discrete(4)
        # main boundary check for edges of map done here.
        # To get to another row, we subtract or add the width of the grid (self.num_cols) since the state is an integer
        self.action_state_to_next_state = [lambda s: s - self.num_cols if self.world[s][1] > 0 else s,                # up
                                           lambda s: s + 1 if self.world[s][0] < (self.num_cols - 1) else s,          # right
                                           lambda s: s + self.num_cols if self.world[s][1] < (self.num_rows - 1) else s, # down
                                           lambda s: s - 1 if self.world[s][0] > 0 else s]                         # left

        self.action_descriptors = ['UP', 'RIGHT', 'DOWN', 'LEFT']
        self.action_descriptor_to_int = {desc: idx for idx, desc in enumerate(self.action_descriptors)}
        # set observed params: [current state, world state]
        self.observation_space = spaces.Discrete(self.world.size)
        # set initial state for the agent. If initial_state is a list, choose randomly
        if isinstance(initial_state, int):
            initial_state = [initial_state] # convert to list
        self.starting_states = initial_state
        self.previous_state = self.current_state = self.initial_state = random.choice(self.starting_states)
        # set terminal goal states state(s) and default terminal state if None given
        if goal_states is None or len(goal_states) == 0:
            self.goal_states = [self.world.size - 1]
        else:
            self.goal_states = goal_states
        # set lava terminal states
        if lava_states is None:
            self.lava_states = []
        else:
            self.lava_states = lava_states
        # set walls
        self.wall_indices = []
        self.wall_grid = np.zeros(self.world.shape)
        self._generate_walls(walls)
        # set reward matrix
        self.reward_matrix = np.full(self.world.shape, -1)
        for terminal_state in self.goal_states:
            try:
                self.reward_matrix[terminal_state] = 10
            except IndexError:
                raise IndexError("Terminal goal state {} is out of grid bounds or is wrong type. Should be an integer.".format(terminal_state))
        for terminal_state in self.lava_states:
            try:
                self.reward_matrix[terminal_state] = -10
            except IndexError:
                raise IndexError("Lava terminal state {} is out of grid bounds or is wrong type. Should be an integer.".format(terminal_state))
        # sensor_mode check
        self.sensor_mode = sensor_mode

        # self.reward_range = [-inf, inf] # default values already
        self.num_previous_states_to_store = 500
        self.last_n_states = []
        # set additional parameters for the environment
        self.done = False
        self.info = {}
        self.screen_width = 1200
        self.screen_height = 800

        self.viewer = None
        self._seed()
        self.np_random, seed = seeding.np_random(55)

        if custom_world_fp:
            self._create_custom_world_from_file(custom_world_fp)
        if random_maze:
            self._create_random_maze(self.num_cols, self.num_rows)

    def _generate_world(self):
        """
        Creates and returns the griduniverse map as a numpy array.

        The states are defined by their index and contain a tuple of uint16 values that represent the
        coordinates (x,y) of a state in the grid.
        """
        world = np.fromiter(((x, y) for y in np.nditer(np.arange(self.num_rows))
                             for x in np.nditer(np.arange(self.num_cols))), dtype='int64, int64')
        return world

    def _generate_walls(self, walls):
        """
        Given a list of wall indices, fills in self.wall_indices list
        and places "1"s appropriately within self.walls numpy array

        self.walls: need index positioning for efficient check in _is_wall() but
        self.wall_indices: we also need list of indices to easily access each wall sequentially (e.g in render())
        """
        if walls is not None:
            for wall_state in walls:
                if wall_state < 0 or wall_state > (self.world.size - 1):
                    raise ValueError("Wall state {} is out of grid bounds".format(wall_state))

                self.wall_grid[wall_state] = 1
                self.wall_indices.append(wall_state)

    def look_step_ahead(self, state, action, care_about_terminal=True):
        """
        Computes the results of a hypothetical action taking place at the given state.

        Returns the state to what that action would lead, the reward at that new state and a boolean value that
        determines if the next state is terminal
        """

        if care_about_terminal:
            if self.is_terminal(state):
                next_state = state
            else:
                next_state = self.action_state_to_next_state[action](state)
                next_state = next_state if not self._is_wall(next_state) else state
        else:
            # repeating code for now, but for good reason
            next_state = self.action_state_to_next_state[action](state)
            next_state = next_state if not self._is_wall(next_state) else state

        return next_state, self.reward_matrix[next_state], self.is_terminal(next_state)

    def _is_wall(self, state):
        """
        Checks if a given state is a wall or any other element that shall not be trespassed.
        """
        return True if self.wall_grid[state] == 1 else False

    def is_terminal(self, state):
        """
        Check if the input state is terminal.
        Which can either be a lava (negative reward) or goal state (positive reward)
        """
        return True if self.is_lava(state) or self.is_terminal_goal(state) else False

    def is_lava(self, state):
        return True if state in self.lava_states else False

    def is_terminal_goal(self, state):
        return True if state in self.goal_states else False

    def state_idx_to_x_y(self, state_index):
        """
        Returns x and y coordinates given state index
        """

        return self.world[state_index][0], self.world[state_index][1]

    def x_y_to_state_idx(self, x, y):
        """
        Returns state index given x and y coordinates
        """

        return y * self.num_cols + x

    def _create_numpy_grid(self):
        """
        0: unblocked walkable state with nothing in it
        1: agent's current location
        2: wall/blocked state
        3: door
        4: lemon
        5: melon
        6: apple
        7: lever
        8: terminal goal state
        9: lava terminal state
        """

        grid = np.zeros(self.world.shape).reshape((self.num_cols, self.num_rows)) # will look like transpose but it's ok

        grid[self.state_idx_to_x_y(self.current_state)] = 1
        for state in self.wall_indices:
            grid[self.state_idx_to_x_y(state)] = 2

        for state in self.goal_states:
            grid[self.state_idx_to_x_y(state)] = 8

        for state in self.lava_states:
            grid[self.state_idx_to_x_y(state)] = 9

        return grid

    def get_observation(self):
        """
        Returns the specific observation type depending on the current sensor_mode
        """

        if self.sensor_mode == 'whole_grid':
            return self._create_numpy_grid()
        return self.current_state

    def _step(self, action):
        """
        Moves the agent one step according to the given action.
        """
        self.previous_state = self.current_state
        self.current_state, reward, self.done = self.look_step_ahead(self.current_state, action)
        self.last_n_states.append(self.world[self.current_state])
        if len(self.last_n_states) > self.num_previous_states_to_store:
            self.last_n_states.pop(0)
        return self.get_observation(), reward, self.done, self.info

    def _reset(self):
        self.done = False
        self.previous_state = self.current_state = self.initial_state = random.choice(self.starting_states)
        self.last_n_states = []
        if self.viewer:
            self.viewer.change_face_sprite()
        return self.get_observation()

    def _render(self, mode='human', close=False):
        if close:
            if self.viewer is not None:
                self.viewer.close()
                self.viewer = None
            return

        if mode == 'human' or mode == 'ansi':
            new_world = np.fromiter(('o' for _ in np.nditer(np.arange(self.num_cols))
                                     for _ in np.nditer(np.arange(self.num_rows))), dtype='S1')
            new_world[self.current_state] = 'x'
            for t_state in self.goal_states:
                new_world[t_state] = 'G'

            for t_state in self.lava_states:
                new_world[t_state] = 'L'

            for w_state in self.wall_indices:
                new_world[w_state] = '#'

            outfile = StringIO() if mode == 'ansi' else sys.stdout
            for row in np.reshape(new_world, (self.num_rows, self.num_cols)):
                for state in row:
                    outfile.write((state.decode('UTF-8') + ' '))
                outfile.write('\n')
            outfile.write('\n')
            return outfile

        elif mode == 'graphic':
            if self.viewer is None:
                from core.envs import rendering
                self.viewer = rendering.Viewer(self, self.screen_width, self.screen_height)

            return self.viewer.render(return_rgb_array=mode == 'rgb_array')
        else:
            super(GridUniverseEnv, self).render(mode=mode)

    def render_policy_arrows(self, policy):
        if self.viewer is None:
            from core.envs import rendering
            self.viewer = rendering.Viewer(self, self.screen_width, self.screen_height)

        self.viewer.render_policy_arrows(policy)

    def _close(self):
        pass

    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def _create_custom_world_from_file(self, fp):
        with open(fp, 'r') as f:
            all_lines = [line.rstrip() for line in f.readlines()]
            all_lines = ["".join(line.split()) for line in all_lines if line] # remove empty lines and any whitespace

            self._create_custom_world_from_text(all_lines)

    def _create_custom_world_from_text(self, text_world_lines):
        """
        Creates the world from a rectangular text file in the format of:

        ooo#
        oxoo
        oooL
        oooG

        Where:
         "o" is an empty walkable area.
         "#" is a blocked "wall"
         "G" is a terminal goal state
         "L" is a lava terminal state
         "x" is a possible starting location. Chosen uniform randomly if multiple "x"s.
        """

        self.goal_states = []
        self.starting_states = []
        self.lava_states = []
        walls_indices = []

        curr_index = 0
        width_of_grid = len(text_world_lines[0])  # first row length will be width from now on
        for y, line in enumerate(text_world_lines):
            if len(line) != width_of_grid:
                raise ValueError("Input text file is not a rectangle")

            for char in line:
                if char == 'G':
                    self.goal_states.append(curr_index)
                elif char == 'L':
                    self.lava_states.append(curr_index)
                elif char == 'o':
                    pass
                elif char == '#':
                    walls_indices.append(curr_index)
                elif char == 'x':
                    self.starting_states.append(curr_index)
                else:
                    raise ValueError('Invalid Character "{}". Returning'.format(char))

                curr_index += 1

        if len(self.starting_states) == 0:
            raise ValueError("No starting states set in text file. Place \"x\" within grid. ")
        if len(self.goal_states) == 0:
            raise ValueError("No terminal goal states set in text file. Place \"T\" within grid. ")

        self.reset()

        self.num_rows = len(text_world_lines)
        self.num_cols = width_of_grid
        self.world = self._generate_world()

        self.wall_grid = np.zeros(self.world.shape)
        self.wall_indices = []
        self._generate_walls(walls_indices)

        self.reward_matrix = np.full(self.world.shape, -1)
        for terminal_state in self.goal_states:
            self.reward_matrix[terminal_state] = 10
        for terminal_state in self.lava_states:
            self.reward_matrix[terminal_state] = -10

    def _create_random_maze(self, width, height):
        all_textworld_lines = maze_generation.create_random_maze(width, height)

        self._create_custom_world_from_text(all_textworld_lines)
