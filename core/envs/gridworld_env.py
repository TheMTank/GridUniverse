import ast
import sys
import random
import time

import numpy as np
from six import StringIO
import gym
from gym import spaces
from gym.utils import seeding

from core.envs import maze_generation

class GridWorldEnv(gym.Env):
    metadata = {'render.modes': ['human', 'ansi', 'graphic', 'rgb_array']}

    def __init__(self, grid_shape=(4, 4), initial_state=0, terminal_states=None, walls=None, levers=None, textworld_fp=None, random_maze=False):
        """
        Main constructor to create a GridWorld environment. The default GridWorld is a square grid of 4x4 where the
        agent starts at the top left corner and the terminal state is at the bottom right corner.

        :param grid_shape: Tuple of size 2 to specify (width, height) of grid
        :param initial_state: int for single initial state or list of possible states chosen uniform randomly
        :param terminal_states: list of terminal states. If agent walks into any, done = True,
                                and no actions are possible
        :param walls: list of walls. These are blocked states where the agent can't walk
        :param levers: dictionary of with integer keys being the location of lever and
                        the values being the wall index (door) to be removed if lever is reached by agent
        :param textworld_fp: optional parameter to create the grid from a text file.
        :param random_maze: optional parameter to randomly generate a maze from the algorithm within maze_generation.py
                            This will override the terminal_states, initial_state, walls and textworld_fp params
        """
        # check state space params
        if terminal_states is not None and not isinstance(terminal_states, list):
            raise TypeError("terminal_states parameter must be a list of integer indices")
        if walls is not None and not isinstance(walls, list):
            raise TypeError("walls parameter must be a list of integer indices")
        if levers is not None and not isinstance(levers, dict):
            raise TypeError("levers parameter must be a dictionary with integer keys being location of lever state \
                             and values being wall index (door) to be removed.")
        if not isinstance(grid_shape, tuple) or len(grid_shape) != 2 or not isinstance(grid_shape[0], int):
            raise TypeError("grid_shape parameter must be tuple of two integers")
        self.x_max = grid_shape[0] # num columns
        self.y_max = grid_shape[1] # num rows
        self.world = self._generate_world()
        # set action space params
        self.action_space = spaces.Discrete(4)
        # main boundary check for edges of map done here.
        # To get to another row, we subtract or add the width of the grid (self.x_max) since the state is an integer
        self.action_state_to_next_state = [lambda s: s - self.x_max if self.world[s][1] > 0 else s,                # up
                                           lambda s: s + 1 if self.world[s][0] < (self.x_max - 1) else s,          # right
                                           lambda s: s + self.x_max if self.world[s][1] < (self.y_max - 1) else s, # down
                                           lambda s: s - 1 if self.world[s][0] > 0 else s]                         # left

        self.action_descriptors = ['UP', 'RIGHT', 'DOWN', 'LEFT']
        self.action_descriptor_to_int = {desc:idx for idx, desc in enumerate(self.action_descriptors)}
        # set observed params: [current state, world state]
        self.observation_space = spaces.Discrete(self.world.size)
        # set initial state for the agent. If initial_state is a list, choose randomly
        if isinstance(initial_state, int):
            initial_state = [initial_state] # convert to list
        self.starting_states = initial_state
        self.previous_state = self.current_state = self.initial_state = random.choice(self.starting_states)
        # set terminal state(s) and default terminal state if None given
        if terminal_states is None or len(terminal_states) == 0:
            self.terminal_states = [self.world.size - 1]
        else:
            self.terminal_states = terminal_states
        for t_s in self.terminal_states:
            if not isinstance(t_s, int):
                raise TypeError("Terminal state {} is not an integer".format(t_s))
            if t_s < 0 or t_s > (self.world.size - 1):
                raise ValueError("Terminal state {} is out of grid bounds".format(t_s))
        # set walls
        self.initial_walls = []
        self.wall_indices = []
        self.non_wall_blocked_states = []
        self.wall_grid = np.zeros(self.world.shape)
        self._setup_walls(walls)
        # set levers
        # lever dict contains key (int where lever is) and value (int of wall index/door)

        self.levers = {}
        self._setup_levers(levers)
        # set reward matrix
        self.reward_matrix = np.full(self.world.shape, -1)
        for terminal_state in self.terminal_states:
            self.reward_matrix[terminal_state] = 0
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

        if textworld_fp:
            self._create_textworld_from_file(textworld_fp)
        if random_maze:
            self._create_random_maze(self.x_max, self.y_max)

    def _generate_world(self):
        """
        Creates and returns the gridworld map as a numpy array.

        The states are defined by their index and contain a tuple of uint16 values that represent the
        coordinates (x,y) of a state in the grid.
        """
        world = np.fromiter(((x, y) for y in np.nditer(np.arange(self.y_max))
                             for x in np.nditer(np.arange(self.x_max))), dtype='int64, int64')
        return world

    def _setup_walls(self, walls):
        """
        Given a list of wall indices, fills in self.wall_indices list
        and places "1"s appropriately within self.walls numpy array

        self.walls: need index positioning for efficient check in _is_wall() but
        self.wall_indices: we also need list of indices to easily access each wall sequentially (e.g in render())
        """
        if not walls:
            self.initial_walls = []
        else:
            self.initial_walls = walls
            self.wall_grid = np.zeros(self.world.shape)
            self.wall_indices = []
            self.non_wall_blocked_states = []
            for wall_state in self.initial_walls:
                if not isinstance(wall_state, int):
                    raise TypeError("Wall state {} is not an integer".format(wall_state))
                if wall_state < 0 or wall_state > (self.world.size - 1):
                    raise ValueError("Wall state {} is out of grid bounds".format(wall_state))

                self.wall_grid[wall_state] = 1
                self.wall_indices.append(wall_state)

        self.non_wall_blocked_states = [x for x in range(self.world.size) if x not in self.wall_indices]

    def _setup_levers(self, lever_dict):
        """
        Setup levers
        """
        if not lever_dict:
            self.levers = {}
        else:
            self.levers = lever_dict

        self.unactivated_levers = {k: v for k, v in self.levers.items()}
        # Parameter checks to see if correct
        for lever_state in self.unactivated_levers.keys():
            # Check lever state can't equal a wall. Key and value can't equal the same if any of the below is raised
            # Check value is always wall
            if self.unactivated_levers[lever_state] not in self.wall_indices:
                raise ValueError("Wall linked to lever state {} is not a wall state".format(lever_state))
            # Check key is always non-wall
            if lever_state in self.wall_indices:
                raise ValueError("Lever state {} can not be placed on top of a wall".format(lever_state))
            if not isinstance(lever_state, int):
                raise TypeError("Lever state {} is not an integer".format(lever_state))
            # Check if within bounds
            if lever_state < 0 or lever_state > (self.world.size - 1):
                raise ValueError("Lever state {} is out of grid bounds".format(lever_state))

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

                if self.unactivated_levers:
                    if next_state in self.unactivated_levers.keys():
                        print('Stepped on lever at state {} to remove door at state {}'.format(next_state, self.unactivated_levers[next_state]))
                        if self.viewer:
                            wall_sprite_index = self.viewer.wall_indices_to_wall_sprite_index[self.unactivated_levers[next_state]]
                            self.viewer.wall_sprites[wall_sprite_index].visible = False

                            # Change lever sprite
                            lever_sprite_index = self.viewer.lever_indices_to_lever_sprite_index[next_state]
                            self.viewer.lever_sprites[lever_sprite_index].image = self.viewer.lever_on_img

                        self.wall_indices.remove(self.unactivated_levers[next_state])
                        self.wall_grid[self.unactivated_levers[next_state]] = 0
                        del self.unactivated_levers[next_state]
        else:
            # repeating code for now, but for good reason
            next_state = self.action_state_to_next_state[action](state)
            next_state = next_state if not self._is_wall(next_state) else state

        return next_state, self.reward_matrix[next_state], self.is_terminal(next_state)

    def _is_wall(self, state):
        """
        Checks if a given state is a wall or any other element that shall not be trespassed.
        """
        if self.wall_grid[state] == 1:
            return True
        return False

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
        # if self.done: # todo: render on done
        #     env.render(mode='graphic')
        # self.last_n_states.append(self.current_state)
        self.last_n_states.append(self.world[self.current_state])
        if len(self.last_n_states) > self.num_previous_states_to_store:
            self.last_n_states.pop(0)
        return self.current_state, reward, self.done, self.info

    def _reset(self):
        """
        Resets agent state (can be random if more than two set)
        Resets environment state (levers, replaces removed walls, changes sprite images)
        """

        self.done = False
        # Randomly choose starting location from list of starting states
        self.previous_state = self.current_state = self.initial_state = random.choice(self.starting_states)
        self.last_n_states = []
        # reset walls and levers sprites
        if self.viewer:
            self.viewer.change_face_sprite()
            for sprite in self.viewer.wall_sprites:
                sprite.visible = True
            for sprite in self.viewer.lever_sprites:
                sprite.image = self.viewer.lever_off_img
        # reset walls and levers states
        self.unactivated_levers = {k: v for k, v in self.levers.items()}
        self._setup_walls(self.initial_walls)
        return self.current_state

    def _render(self, mode='human', close=False):
        """
        Renders agent in either ASCII ('human', 'ansi') or pyglet OpenGL ('graphic', 'rgb_array') mode
        """

        if mode == 'human' or mode == 'ansi':
            new_world = np.fromiter(('o' for _ in np.nditer(np.arange(self.x_max))
                                     for _ in np.nditer(np.arange(self.y_max))), dtype='S1')
            new_world[self.current_state] = 'x'
            for t_state in self.terminal_states:
                new_world[t_state] = 'T'

            for w_state in self.wall_indices:
                new_world[w_state] = '#'

            if self.unactivated_levers:
                for lever_state in self.unactivated_levers.keys():
                    new_world[lever_state] = '\\'

            if self.levers:
                # render activated levers
                for lever_state in [k for k in self.levers.keys() if k not in self.unactivated_levers.keys()]:
                    new_world[lever_state] = '|'

            outfile = StringIO() if mode == 'ansi' else sys.stdout
            for row in np.reshape(new_world, (self.y_max, self.x_max)):
                for state in row:
                    outfile.write((state.decode('UTF-8') + ' '))
                outfile.write('\n')
            outfile.write('\n')
            return outfile

        elif mode == 'graphic' or mode == 'rgb_array':
            if close: # code needed for pressing x on window to close
                if self.viewer is not None:
                    self.viewer.close()
                    self.viewer = None
                return

            if self.viewer is None:
                from core.envs import rendering
                self.viewer = rendering.Viewer(self, self.screen_width, self.screen_height)

            return self.viewer.render(return_rgb_array=mode == 'rgb_array')
        else:
            super(GridWorldEnv, self).render(mode=mode)

    def render_policy_arrows(self, policy):
        """
        Only works if pyglet is installed. Removes previous policy arrows in pyglet rendering if already done.
        """
        if self.viewer is None:
            from core.envs import rendering
            self.viewer = rendering.Viewer(self, self.screen_width, self.screen_height)

        self.viewer.render_policy_arrows(policy)

    def _close(self):
        pass

    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def _create_textworld_from_file(self, fp):
        """
        Opens file fp and does a little cleaning before sending it to function _create_textworld_from_text()
        """

        with open(fp, 'r') as f:
            all_lines = [line.rstrip() for line in f.readlines()]
            all_lines = ["".join(line.split()) for line in all_lines if line] # remove empty lines and any whitespace

            self._create_textworld_from_text(all_lines)

    def _create_textworld_from_text(self, textworld_lines):
        """
        Creates the world from a rectangular text file in the format of:

        ooo#
        oxoo
        oooo
        oooT

        Where:
         "o" is an empty walkable area.
         "#" is a blocked "wall"
         "T" is a terminal state
         "x" is a possible starting location. Chosen uniform randomly if multiple "x"s.

        If you would like to include lever metadata add a line of dashes
        and then a line with a python dictionary with lever keys and values being doors (check constructor for more info)
        For example:
        ----
        {42: 22}

        This final line will be parsed with ast package and so is quite error prone to wrong syntax.
        """

        self.terminal_states = []
        self.starting_states = []
        walls_indices = []

        curr_index = 0
        width_of_grid = len(textworld_lines[0])  # first row length will be width from now on
        height_of_grid = len(textworld_lines)
        for y, line in enumerate(textworld_lines):
            if line[0] == '-':
                height_of_grid = y
                metadata_line_str = textworld_lines[y + 1]
                print('Lever metadata: ', metadata_line_str)

                try:
                    lever_metadata = ast.literal_eval(metadata_line_str)
                    if not isinstance(lever_metadata, dict):
                        raise(TypeError('Lever metadata line after converting to Python is not a dictionary'))
                except:
                    raise(TypeError('Lever metadata line is not in correct dictionary format. \
                                      \nKeys and values should be ints representing {lever_state_index: wall_state_index} e.g. {5: 3, 7: 6}. \nThis is how the whole text file should look: \nxooo\noooo\noooo\noooT\n----------\n{5: 3, 7: 6}'))
                break
            if len(line) != width_of_grid:
                raise ValueError("Input text file is not a rectangle")

            for char in line:
                if char == 'T':
                    self.terminal_states.append(curr_index)
                elif char == 'o':
                    pass
                elif char == '#':
                    walls_indices.append(curr_index)
                elif char == 'x':
                    self.starting_states.append(curr_index)
                else:
                    raise ValueError('Invalid Character "{}". Returning'.format(char))

                curr_index += 1

        if len(self.terminal_states) == 0:
            raise ValueError("No terminal states set in text file. Place \"T\" within grid. ")

        self.y_max = height_of_grid
        self.x_max = width_of_grid
        self.world = self._generate_world()

        self.initial_walls = []
        self._setup_walls(walls_indices)

        # No starting states set in file
        if len(self.starting_states) == 0:
            # One option (crash if no 'x'):
            # raise ValueError("No starting states set in text file. Place \"x\" within grid. ")
            # 2nd option. Random start in any place with no wall every reset
            self.starting_states = self.non_wall_blocked_states

        # Set agent location and common things
        self.reset()

        if 'lever_metadata' in locals():
            self._setup_levers(lever_metadata)

        self.reward_matrix = np.full(self.world.shape, -1)
        for terminal_state in self.terminal_states:
            self.reward_matrix[terminal_state] = 0

    def _create_random_maze(self, width, height):
        all_textworld_lines = maze_generation.create_random_maze(width, height)

        self._create_textworld_from_text(all_textworld_lines)
