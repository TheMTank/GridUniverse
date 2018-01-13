import sys
import random

from six import StringIO
import numpy as np
from numpy.random import random_integers as rand
import matplotlib.pyplot as pyplot

def maze(width=81, height=51, complexity=.75, density=.75):
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

def create_random_maze(width, height):
    pyplot.figure(figsize=(10, 5))
    maze1 = maze(width, height)
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
    all_lines[x][y] = 'x'
    x = T_idx % num_cols
    y = T_idx // num_cols
    all_lines[x][y] = 'T'

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
    create_random_maze(11, 11)
