import sys
import random
import time

from six import StringIO
import numpy as np
from numpy.random import random_integers as rand
import matplotlib.pyplot as pyplot

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

def recursive_backtracker(width=20, height=20):
    shape = (height, width)
    print('Maze grid shape:', shape)
    # Build actual maze
    # Z = np.zeros(shape, dtype=bool)

    # todo don't make environment, create a new app, All it needs is Z
    # env = GridWorldEnv(grid_shape=shape)
    # env.render(mode='graphic') # needs so many changes to get this to work
    # time.sleep(3)

    Z = np.ones(shape, dtype=bool) # begin everything as walls
    # Fill borders?
    # Z[0, :] = Z[-1, :] = 1
    # Z[:, 0] = Z[:, -1] = 1

    visited = np.zeros(shape, dtype=bool)
    stack = []

    # 1. Make the initial cell the current cell and mark it as visited
    # everything is x, y but indexing is y, x
    current_cell = initial_cell = rand(0, shape[1] - 1), rand(0, shape[0] - 1) # inclusive
    print('initial_cell:', initial_cell)
    visited[current_cell[1], current_cell[0]] = True  # must index by y, x

    # 2. While there are unvisited cells
    while not visited.all():
        # 1. If the current cell has any neighbours which have not been visited
        x, y = current_cell
        print('current_cell:', current_cell)
        options = []
        if x + 2 < width and y < height and not visited[(y, x + 2)]:
            options.append((x + 2, y))
        if x - 2 > -1 and y < height and not visited[(y, x - 2)]:
            options.append((x - 2, y))
        if y + 2 < height and x < width and not visited[(y + 2, x)]:
            options.append((x, y + 2))
        if y - 2 > -1 and x < width and not visited[(y - 2, x)]:
            options.append((x, y - 2))

        print('Num options:', len(options))
        if len(options) > 0:
            # 1. Choose randomly one of the unvisited neighbours
            random_neighbour = random.choice(options)
            print('random_neighbour:', random_neighbour)
            # 2. Push the current cell to the stack
            stack.append(current_cell)
            # 3. Remove the wall between the current cell and the chosen neighbour cell
            neighbour_x, neighbour_y = random_neighbour
            wall_x, wall_y = (neighbour_x + x) // 2, (neighbour_y + y) // 2
            print('carving wall: ', wall_x, wall_y)

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
                print('popping')
                current_cell = stack.pop()
            else:
                break

        # for printing out variations
        # time.sleep(0.4)
        # print(Z.astype(int))
        # print((visited == True).sum(), visited.size)
    return Z

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

if __name__ == '__main__':
    # create_random_maze(11, 11)
    # todo render maze while creating it?
    # create_random_maze(10, 10)
    recursive_backtracker(10, 10)
