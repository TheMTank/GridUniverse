import random
import sys
import time
# from queue import *
from queue import Queue
# import queue

from core.envs.gridworld_env import GridWorldEnv

# todo: could rename file to path_finding.py

if __name__ == '__main__':
    print('\n' + '*' * 20 + 'Creating a random GridWorld and running random agent on it' + '*' * 20 + '\n')


    first_time = True
    for i in range(10):
        env = GridWorldEnv(grid_shape=(15, 15), random_maze=True) # todo turn off terminal somehow to calculate recursively?
        curr_state = initial_state = env.reset()

        actions = range(4)

        all_valid_states = []

        visited = {}
        nodes_and_edges = {}

        # https: // en.wikipedia.org / wiki / A * _search_algorithm

        # def find_all_neighbouring_states(curr_state):
        #     nodes_and_edges[curr_state] = []
        #     for action in actions:
        #         # next_states = []
        #         next_state, reward, done = env.look_step_ahead(curr_state, action)
        #         # if next_state != curr_state and next_state not in nodes_and_edges: # todo can change to graph
        #         if next_state != curr_state and next_state not in visited: # todo can change to graph
        #             # next_states.append
        #             visited[next_state] = 1
        #             # nodes_and_edges[curr_state].append(next_state)
        #             all_valid_states.append(next_state)
        #             find_all_neighbouring_states(next_state)

        def create_graph():
            for curr_state in range(env.world.size):
                if not env._is_wall(curr_state):# and not env.is_terminal(curr_state):
                    nodes_and_edges[curr_state] = []
                    for action in actions:
                        next_state, reward, done = env.look_step_ahead(curr_state, action, False)
                        if next_state != curr_state: # and next_state not in nodes_and_edges:
                            nodes_and_edges[curr_state].append(next_state)

                # if len(nodes_and_edges[curr_state]) == 0:
                #     del nodes_and_edges[curr_state]

        # find_all_neighbouring_states(initial_state)
        create_graph()


        sys.stdout.flush()
        # time.sleep(1)
        print()
        print(all_valid_states)

        # for state in all_valid_states:
        #     nodes_and_edges[state] = []
        #     for action in actions:
        #         next_state, reward, done = env.look_step_ahead(curr_state, action)
        #         if next_state != curr_state:
        #             nodes_and_edges[state].append(next_state)

        print()
        print(nodes_and_edges)

        vertex_visited = {}
        global_stack = []
        global_found = False

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
                # return 'RIGHT'
                return 'LEFT'
            if parent_state - next_state == -1:
                # return 'LEFT'
                return 'RIGHT'
            if parent_state - next_state < -1: # if next state is bigger
                return 'DOWN'
            if parent_state - next_state > 1:
                return 'UP'
            return 'Something Wrong'

        def breadth_first_search(graph, start_state):
            # a FIFO open_set
            # open_set = Queue()
            open_set = [] # list queue
            # an empty set to maintain visited nodes
            closed_set = set()
            # a dictionary to maintain meta information (used for path formation)
            meta = dict()  # key -> (parent state, action to reach child)

            # initialize
            # start = problem.get_start_state()
            meta[start_state] = (None, None) # todo rename to pathways?
            # open_set.enqueue(start)
            # open_set.put(start)
            open_set.append(start_state)

            # while not open_set.is_empty():
            # while not open_set.empty():
            while len(open_set) > 0:
                # parent_state = open_set.dequeue()
                # parent_state = open_set.get()
                parent_state = open_set.pop(0)

                print('Parent State: {}, edges: {}'.format(parent_state, graph[parent_state]))
                # print(meta)
                #if problem.is_goal(parent_state):
                if env.is_terminal(parent_state):
                    print('Calling construct_path. meta:', meta)
                    return construct_path(parent_state, meta)
                # print('Within while: ', graph[parent_state])

                # for (child_state, action) in graph[parent_state]: #problem.get_successors(parent_state):
                for child_state in graph[parent_state]:
                    if child_state in closed_set:
                        continue

                    if child_state not in open_set:
                        action = calculate_action(parent_state, child_state)
                        meta[child_state] = (parent_state, action)
                        # meta[child_state] = (parent_state)
                        # open_set.enqueue(child_state)
                        # open_set.put(child_state)
                        open_set.append(child_state)

                closed_set.add(parent_state)

            construct_path(parent_state, meta)

        def construct_path(state, meta):
            action_list = []
            # print('Action List: ', end='')
            print('Action List: ')
            while True:
                print(state)
                row = meta[state]

                if len(row) == 2: #todo get rid of
                    state = row[0]
                    if state is None:
                        break
                    # print(row[1] + ', ', end='')
                    print(state, row[1])
                    action = env.action_descriptor_to_int[row[1]]
                    action_list.append(action)
                else:
                    break

                # print(action_list)
            action_list.reverse()
            print(action_list)
            return action_list

        # print(list(nodes_and_edges.keys()))
        #root_vertex = random.choice(list(nodes_and_edges.keys())) #todo should be 'x'
        root_vertex = env.initial_state # todo not same as x
        # depth_first_search_recursive(nodes_and_edges, root_vertex)
        # node_path_to_terminal = depth_first_search_iterative(nodes_and_edges, root_vertex)
        print('Initial states edges:', nodes_and_edges[root_vertex])
        action_list_to_terminal = breadth_first_search(nodes_and_edges, root_vertex)
        print('Initial state:', root_vertex)
        print('Path to terminal:', action_list_to_terminal)

        # for i_episode in range(2):
        for i_episode in range(1):
            observation = env.reset()
            for step_num, action in enumerate(action_list_to_terminal):
                env.render(mode='graphic')
                if first_time:
                    time.sleep(5)
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
