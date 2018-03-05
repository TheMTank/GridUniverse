import random
import sys
import time
# from queue import *
from queue import Queue
# import queue

from core.envs.griduniverse_env import GridUniverseEnv


# vertex_visited = {} # needed for DFS?
# global_stack = []
# global_found = False

def depth_first_search_recursive(graph, state):
    vertex_visited[state] = 1

    for neighbour_state in graph[state]:
        if neighbour_state not in vertex_visited:
            if global_found:
                return True
            global_stack.append(neighbour_state)
            if env.is_terminal(neighbour_state):
                global_found = True
                return True
            depth_first_search_recursive(graph, neighbour_state)


def depth_first_search_iterative(graph, node):
    nodes_visited = {}
    stack = []
    stack.append(node)

    while len(stack) > 0:
        node = stack.pop()
        print('pop:', node)
        if node not in nodes_visited:
            nodes_visited[node] = 1
            for neighbour_state in graph[node]:
                if neighbour_state not in vertex_visited:
                    stack.append(neighbour_state)
                    if env.is_terminal(neighbour_state):
                        # break
                        return stack


def calculate_action(parent_state, next_state):
    # If you want to get from parent_state to next_state

    if parent_state - next_state == 1:
        return 'LEFT'
    if parent_state - next_state == -1:
        return 'RIGHT'
    if parent_state - next_state < -1:  # if next state is bigger
        return 'DOWN'
    if parent_state - next_state > 1:
        return 'UP'
    return 'Something Wrong'


def breadth_first_search(graph, start_state):
    """
    Breadth First Search checks depth of graph 1 level at a time
    """

    # a FIFO open_set
    open_set = []  # list queue
    # an empty set to maintain visited nodes
    closed_set = set()
    # a dictionary to maintain meta information (used for path formation)
    meta = dict()  # key -> (parent state, action to reach child)

    # initialize
    meta[start_state] = (None, None)  # todo rename to pathways?
    open_set.append(start_state)

    while len(open_set) > 0:
        parent_state = open_set.pop(0)

        print('Parent State: {}, edges: {}'.format(parent_state, graph[parent_state]))
        # print(meta)
        if env.is_terminal(parent_state):
            print('Calling construct_path. meta:', meta)
            return construct_path(parent_state, meta)
        # print('Within while: ', graph[parent_state])

        # todo check if path to goal is possible

        for child_state in graph[parent_state]:
            if child_state in closed_set:
                continue

            if child_state not in open_set:
                action = calculate_action(parent_state, child_state)
                meta[child_state] = (parent_state, action)
                open_set.append(child_state)

        closed_set.add(parent_state)

    construct_path(parent_state, meta)

def construct_path(state, meta):
    """
    Construct path from goal state to start state and then reverse it
    """

    action_list = []
    print('Action List: ')
    while True:
        print(state)
        row = meta[state]

        if len(row) == 2:  # todo get rid of
            state = row[0]
            if state is None:
                break
            # print(row[1] + ', ', end='')
            print(state, row[1])
            action = env.action_descriptor_to_int[row[1]]
            action_list.append(action)
        else:
            break

    action_list.reverse()
    print(action_list)
    return action_list

def create_graph(env):
    nodes_and_edges = {}
    for curr_state in range(env.world.size):
        if not env._is_wall(curr_state):  # and not env.is_terminal(curr_state):
            nodes_and_edges[curr_state] = []
            for action in actions:
                next_state, reward, done = env.look_step_ahead(curr_state, action, care_about_terminal=False)
                if next_state != curr_state:  # and next_state not in nodes_and_edges:
                    nodes_and_edges[curr_state].append(next_state)

    return nodes_and_edges

if __name__ == '__main__':
    print('\n' + '*' * 20 + 'Creating a random GridUniverse and running random agent on it' + '*' * 20 + '\n')

    first_time = True
    for i in range(3):
        env = GridUniverseEnv(grid_shape=(15, 15), random_maze=True)
        curr_state = initial_state = env.reset()

        actions = range(4)
        visited = {}

        # https: // en.wikipedia.org / wiki / A * _search_algorithm

        nodes_and_edges = create_graph(env)
        print(nodes_and_edges)

        # print(list(nodes_and_edges.keys()))
        root_vertex = env.initial_state
        # depth_first_search_recursive(nodes_and_edges, root_vertex)
        # node_path_to_terminal = depth_first_search_iterative(nodes_and_edges, root_vertex)
        print('Initial states edges:', nodes_and_edges[root_vertex])
        action_list_to_terminal = breadth_first_search(nodes_and_edges, root_vertex)
        print('Initial state:', root_vertex)
        print('Path to terminal:', action_list_to_terminal)

        for i_episode in range(2):
            observation = env.reset()
            for step_num, action in enumerate(action_list_to_terminal):
                env.render(mode='graphic')
                if first_time:
                    # time.sleep(5)
                    first_time = False
                # env.render()
                # time.sleep(0.2)
                # print('go ' + env.action_descriptors[action])
                observation, reward, done, info = env.step(action)
                if done or step_num == len(action_list_to_terminal) - 1:
                    env.render(mode='graphic')
                    env.render()
                    time.sleep(1)
                    print("Episode finished after {} timesteps. Done: {}".format(step_num + 1, done))
                    break
