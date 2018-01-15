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

    def find_all_neighbouring_states(curr_state):
        nodes_and_edges[curr_state] = []
        for action in actions:
            # next_states = []
            next_state, reward, done = env.look_step_ahead(curr_state, action)
            # if next_state != curr_state and next_state not in nodes_and_edges: # todo can change to graph
            if next_state != curr_state and next_state not in visited: # todo can change to graph
                # next_states.append
                visited[next_state] = 1
                # nodes_and_edges[curr_state].append(next_state)
                all_valid_states.append(next_state)
                find_all_neighbouring_states(next_state)


    find_all_neighbouring_states(initial_state)

    sys.stdout.flush()
    # time.sleep(1)
    print()
    print(all_valid_states)

    for state in all_valid_states:
        nodes_and_edges[state] = []
        for action in actions:
            next_state, reward, done = env.look_step_ahead(curr_state, action)
            if next_state != curr_state:
                nodes_and_edges[state].append(next_state)

    print()
    print(nodes_and_edges)

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
