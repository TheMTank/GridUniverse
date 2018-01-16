import random
import sys
import time

from core.envs.gridworld_env import GridWorldEnv

if __name__ == '__main__':
    print('\n' + '*' * 20 + 'Creating a random GridWorld and running random agent on it' + '*' * 20 + '\n')
    env = GridWorldEnv(grid_shape=(5, 5), random_maze=True) # todo turn off terminal somehow to calculate recursively?
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
            if env._is_wall(curr_state):# and not env.is_terminal(curr_state):
                nodes_and_edges[curr_state] = []
                for action in actions:
                    next_state, reward, done = env.look_step_ahead(curr_state, action, False)
                    if next_state != curr_state:# and next_state not in nodes_and_edges:
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

    # print(list(nodes_and_edges.keys()))
    root_vertex = random.choice(list(nodes_and_edges.keys()))
    # depth_first_search_recursive(nodes_and_edges, root_vertex)
    node_path_to_terminal = depth_first_search_iterative(nodes_and_edges, root_vertex)
    print('Initial state:', root_vertex)
    print('Path to terminal:', node_path_to_terminal)

    # for i_episode in range(1):
    #     observation = env.reset()
    #     for t in range(1000):
    #         env.render(mode='graphic')
    #         action = env.action_space.sample()
    #         # print('go ' + env.action_descriptors[action])
    #         observation, reward, done, info = env.step(action)
    #         if done:
    #             print("Episode finished after {} timesteps".format(t + 1))
    #             break
