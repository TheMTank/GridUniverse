import sys
import random
import time
import math
import os

from six import StringIO
import numpy as np
from numpy.random import random_integers as rand
import matplotlib.pyplot as pyplot

from gym.utils import reraise

import pyglet
try:
    from pyglet.gl import *
except ImportError as e:
    reraise(prefix="Error occured while running `from pyglet.gl import *`",suffix="HINT: make sure you have OpenGL install. On Ubuntu, you can run 'apt-get install python-opengl'. If you're running on a server, you may need a virtual frame buffer; something like this should work: 'xvfb-run -s \"-screen 0 1400x900x24\" python <your_script.py>'")

def odd_maze(width=81, height=51, complexity=.75, density=.75):
    # Only odd shapes
    shape = ((height // 2) * 2 + 1, (width // 2) * 2 + 1)
    # Adjust complexity and density relative to maze size
    complexity = int(complexity * (5 * (shape[0] + shape[1])))
    density    = int(density * ((shape[0] // 2) * (shape[1] // 2)))
    # Build actual maze
    Z = np.zeros(shape, dtype=bool)
    # Fill borders
    Z[0, :] = Z[-1, :] = 1
    Z[:, 0] = Z[:, -1] = 1
    # Make aisles
    for i in range(density):
        x, y = rand(0, shape[1] // 2) * 2, rand(0, shape[0] // 2) * 2
        Z[y, x] = 1
        for j in range(complexity):
            neighbours = []
            if x > 1:             neighbours.append((y, x - 2))
            if x < shape[1] - 2:  neighbours.append((y, x + 2))
            if y > 1:             neighbours.append((y - 2, x))
            if y < shape[0] - 2:  neighbours.append((y + 2, x))
            if len(neighbours):
                y_,x_ = neighbours[rand(0, len(neighbours) - 1)]
                if Z[y_, x_] == 0:
                    Z[y_, x_] = 1
                    Z[y_ + (y - y_) // 2, x_ + (x - x_) // 2] = 1
                    x, y = x_, y_
    return Z

def recursive_backtracker(current_cell, width=20, height=20):
    shape = (height, width)
    # print('Maze grid shape:', shape)
    # Build actual maze
    # Z = np.zeros(shape, dtype=bool)

    # todo don't make environment, create a new app, All it needs is Z
    # env = GridWorldEnv(grid_shape=shape)
    # render(mode='graphic') # needs so many changes to get this to work
    # time.sleep(3)

    # Z = np.ones(shape, dtype=bool) # begin everything as walls
    # Fill borders?
    # Z[0, :] = Z[-1, :] = 1
    # Z[:, 0] = Z[:, -1] = 1

    #visited = np.zeros(shape, dtype=bool)
    #stack = []

    # 1. Make the initial cell the current cell and mark it as visited
    # everything is x, y but indexing is y, x
    # current_cell = initial_cell = rand(0, shape[1] - 1), rand(0, shape[0] - 1) # inclusive
    # print('initial_cell:', initial_cell)
    visited[current_cell[1], current_cell[0]] = True  # must index by y, x

    # 2. While there are unvisited cells
    # while not visited.all():
    # 1. If the current cell has any neighbours which have not been visited
    x, y = current_cell
    # print('current_cell:', current_cell)
    options = []
    if x + 2 < width and y < height and not visited[(y, x + 2)]:
        options.append((x + 2, y))
    if x - 2 > -1 and y < height and not visited[(y, x - 2)]:
        options.append((x - 2, y))
    if y + 2 < height and x < width and not visited[(y + 2, x)]:
        options.append((x, y + 2))
    if y - 2 > -1 and x < width and not visited[(y - 2, x)]:
        options.append((x, y - 2))

    # print('Num options:', len(options))
    if len(options) > 0:
        # 1. Choose randomly one of the unvisited neighbours
        random_neighbour = random.choice(options)
        # print('random_neighbour:', random_neighbour)
        # 2. Push the current cell to the stack
        stack.append(current_cell)
        # 3. Remove the wall between the current cell and the chosen neighbour cell
        neighbour_x, neighbour_y = random_neighbour
        wall_x, wall_y = (neighbour_x + x) // 2, (neighbour_y + y) // 2
        # print('carving wall: ', wall_x, wall_y)

        # Carve current cell, wall and neighbour.
        Z[wall_y, wall_x] = 0
        Z[neighbour_y, neighbour_x] = 0
        Z[current_cell[1], current_cell[0]] = 0

        # todo set wall as visited too?
        # 4. Make the chosen cell the current cell and mark it as visited
        current_cell = random_neighbour
        visited[current_cell[1], current_cell[0]] = True
    else:
        # 2. Else if stack is not empty
        if len(stack) > 0:
            # 1. Pop a cell from the stack
            # 2. Make it the current cell
            print('popping') #todo doesn't work 100% of the time
            current_cell = stack.pop()
        else:
            Z, current_cell, True
            print('Finished generation of maze')


        # for printing out variations
        # time.sleep(0.4)
        # print(Z.astype(int))
        # print((visited == True).sum(), visited.size)

    return Z, current_cell, False

def create_random_maze(width, height):
    pyplot.figure(figsize=(10, 5))
    # maze1 = odd_maze(width, height)
    # maze1 = odd_maze(width, height, density=0.1) # for large open areas
    maze1 = recursive_backtracker(width, height)
    print(maze1.shape)

    all_lines = []

    outfile = sys.stdout
    for row in np.reshape(maze1, maze1.shape):
        all_lines.append([])
        for state in row:
            if state == 1:
                # outfile.write('# ')
                all_lines[-1].append('#')
            else:
                # outfile.write('o ')
                all_lines[-1].append('o')
        outfile.write('\n')

    all_o_idxs = []
    # randomly place 'x' and 'T' # todo sample 90th percentile euclidean distance between pairwise distances
    index = 0
    for line in all_lines:
        for state in line:
            if state == 'o':
                all_o_idxs.append(index)
            index += 1

    # random sample two places: one for starting location, one for terminal
    start_idx, T_idx = random.sample(all_o_idxs, 2)
    num_cols = len(all_lines[0])

    x = start_idx % num_cols  # state number 9 => 9 % 4 = 1
    y = start_idx // num_cols  # state number 9 = > 9 // 4 = 1
    all_lines[y][x] = 'x'
    x = T_idx % num_cols
    y = T_idx // num_cols
    all_lines[y][x] = 'T'

    for line in all_lines:
        for char in line:
            outfile.write(char + ' ')
        outfile.write('\n')

    return all_lines

    # todo can definitely do above better and cleaner

    # todo fix float error below or not.
    # pyplot.imshow(maze, cmap=pyplot.cm.binary, interpolation='nearest')
    # pyplot.imshow(maze, cmap=pyplot.cm.Greys, interpolation='nearest')
    # pyplot.xticks([]), pyplot.yticks([])
    # pyplot.show()

global window
width, height = 800, 600
window = pyglet.window.Window(width, height)

@window.event
def on_draw():
    global Z, called, current_cell, window

    glMatrixMode(GL_PROJECTION)
    glLoadIdentity()

    # Initialize Modelview matrix
    glMatrixMode(GL_MODELVIEW)
    glLoadIdentity()
    # Save the default modelview matrix
    glPushMatrix()

    glClearColor(0, 0, 0, 1)
    glOrtho(left, right, bottom, top, 1, -1)

    window.clear()
    # label.draw()

    batch.draw()
    # print('calling maze generator')
    Z, current_cell, done = recursive_backtracker(current_cell, x_max, y_max)

    for index, val in enumerate(Z.flatten()):
        x = i % x_max  # state number 9 => 9 % 4 = 1
        y = i // x_max  # state number 9 = > 9 // 4 = 1
        if val == 1:
            pass
        elif not visited[y, x]:
            all_sprites[index].image = ground_img # todo don't replace each time

    if done:
        # time.sleep(1) # todo doesn't work
        window.close() # todo doesn't work
        # event_loop.exit()
        sys.exit() # todo doesn't work

    glPopMatrix()

def update(*args):
    pass

if __name__ == '__main__':
    # create_random_maze(11, 11)
    
    x_max, y_max = 10, 10
    x_max, y_max = 30, 30
    # x_max, y_max = 50, 50
    # x_max, y_max = 100, 100
    global visited, stack, Z
    current_cell = initial_cell = rand(0, x_max - 1), rand(0, y_max - 1)  # inclusive
    visited = np.zeros((x_max, y_max), dtype=bool)
    stack = []
    Z = np.ones((x_max, y_max), dtype=bool)

    # recursive_backtracker(y_max, x_max)

    # width, height = 800, 600 # done globally
    # window = pyglet.window.Window(width, height)

    script_dir = os.path.dirname(__file__)
    resource_path = os.path.join(script_dir, '..', 'resources')
    pyglet.resource.path = [resource_path]
    pyglet.resource.reindex()

    ground_img = pyglet.resource.image('wbs_texture_05_resized.jpg')
    wall_img = pyglet.resource.image('wbs_texture_05_resized_red.jpg')

    # label = pyglet.text.Label('Hello, world',
    #                           font_name='Times New Roman',
    #                           font_size=36,
    #                           x=window.width // 2, y=window.height // 2,
    #                           anchor_x='center', anchor_y='center')

    padding = 1
    tile_dim = ground_img.width + padding

    all_sprites = []

    batch = pyglet.graphics.Batch()

    # have to flip pixel location. top-left is initial state = x, y = 0, 0 = state 0
    pix_grid_height = (y_max - 1) * tile_dim

    for i in range(x_max * y_max):
        x = i % x_max  # state number 9 => 9 % 4 = 1
        y = i // x_max  # state number 9 = > 9 // 4 = 1
        x_pix_loc, y_pix_loc = x * tile_dim, pix_grid_height - y * tile_dim
        if Z[y, x] == 1:  # todo totally wrong inverse?
            all_sprites.append(pyglet.sprite.Sprite(wall_img, x=x_pix_loc, y=y_pix_loc, batch=batch))
        else:
            all_sprites.append(pyglet.sprite.Sprite(ground_img, x=x_pix_loc, y=y_pix_loc, batch=batch))

    # must accommodate for the bigger dimension but also check smaller dimension so that it fits.
    # larger dimension check
    ind = np.argmax((x_max, y_max))
    larger_grid_dimension = np.max((x_max, y_max))
    if ind == 0:
        larger_pixel_dimension = width
        smaller_pixel_dimension = height
        smaller_grid_dimension = y_max
    elif ind == 1:
        larger_pixel_dimension = height
        smaller_pixel_dimension = width
        smaller_grid_dimension = x_max

    how_many_tiles_you_can_fit_in_larger_dim = math.floor(larger_pixel_dimension / tile_dim)
    zoom_level = larger_grid_dimension / how_many_tiles_you_can_fit_in_larger_dim  # + 5
    # smaller dimension check
    how_many_tiles_you_can_fit_in_smaller_dim = math.floor(smaller_pixel_dimension / tile_dim)
    other_zoom_level = smaller_grid_dimension / how_many_tiles_you_can_fit_in_smaller_dim
    # if other dimension still can't fit the tiles within the map, use its zoom level
    if other_zoom_level > zoom_level:
        zoom_level = other_zoom_level

    # if you can fit more tiles into the black space, then no need to zoom. # todo or maybe a bit and centre the maze?
    if how_many_tiles_you_can_fit_in_larger_dim > larger_grid_dimension and how_many_tiles_you_can_fit_in_smaller_dim > smaller_grid_dimension:
        zoom_level = 1

    zoomed_width = width * zoom_level
    zoomed_height = height * zoom_level

    left = 0
    right = zoomed_width
    bottom = 0
    top = zoomed_height

    print('tile_dim: {}. grid_shape: {}'.format(tile_dim, [x_max, y_max]))
    print('how_many_tiles_you_can_fit_in_smaller_dim: {}, how_many_tiles_you_can_fit_in_larger_dim: {}'.format(
        how_many_tiles_you_can_fit_in_smaller_dim, how_many_tiles_you_can_fit_in_larger_dim))
    print('zoom:', zoom_level)
    print('width: {}, height: {}, zoomed_width: {}, zoomed_height: {}'.format(width, height, zoomed_width,
                                                                              zoomed_height))
    glViewport(0, 0, width, height)

    # Set antialiasing
    glEnable(GL_LINE_SMOOTH)
    glEnable(GL_POLYGON_SMOOTH)
    glHint(GL_LINE_SMOOTH_HINT, GL_NICEST)

    glEnable(GL_BLEND)
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)

    print('before pyg')
    pyglet.clock.schedule_interval(update, 1.0 / 60.0)
    pyglet.app.run()
