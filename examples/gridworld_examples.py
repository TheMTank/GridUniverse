import time

import numpy as np

from core.envs.gridworld_env import GridWorldEnv
from core.algorithms.monte_carlo import run_episode, monte_carlo_evaluation
from core.algorithms import utils
import core.algorithms.dynamic_programming as dp

def run_random_gridworld():
    print('\n' + '*' * 20 + 'Starting to run random agent on GridWorld' + '*' * 20 + '\n')
    env = GridWorldEnv()
    for i_episode in range(1):
        observation = env.reset()
        for t in range(100):
            env.render()
            action = env.action_space.sample()
            print('go ' + env.action_descriptors[action])
            observation, reward, done, info = env.step(action)

            if done:
                print("Episode finished after {} timesteps".format(t + 1))
                break

def run_and_create_gridworld_from_text_file():
    print('\n' + '*' * 20 + 'Creating a pre-made GridWorld from text file and running random agent on it' + '*' * 20 + '\n')
    # env = GridWorldEnv(custom_world_fp='../core/envs/test_env.txt')
    # env = GridWorldEnv(grid_shape=(10, 10))
    # env = GridWorldEnv(custom_world_fp='../core/envs/maze_text_files/maze_21x21.txt')
    env = GridWorldEnv(custom_world_fp='../core/envs/maze_text_files/maze_101x101.txt')
    for i_episode in range(1):
        observation = env.reset()
        for t in range(1000):
            env.render(mode='graphic')
            action = env.action_space.sample()
            # print('go ' + env.action_descriptors[action])
            observation, reward, done, info = env.step(action)
            if done:
                print("Episode finished after {} timesteps".format(t + 1))
                break

def run_random_maze():
    print('\n' + '*' * 20 + 'Creating a random GridWorld and running random agent on it' + '*' * 20 + '\n')
    # env = GridWorldEnv(grid_shape=(11, 11), random_maze=True)
    # env = GridWorldEnv(grid_shape=(50, 20), random_maze=True)
    env = GridWorldEnv(grid_shape=(20, 50), random_maze=True)
    for i_episode in range(1):
        observation = env.reset()
        for t in range(10000000):
            env.render(mode='graphic')
            env.step_num = t
            action = env.action_space.sample()
            # print('go ' + env.action_descriptors[action])
            observation, reward, done, info = env.step(action)
            if done:
                print("Episode finished after {} timesteps".format(t + 1))
                break

def run_policy_iteration_gridworld():
    print('\n' + '*' * 20 + 'Starting value and policy iteration' + '*' * 20 + '\n')
    # test policy evaluation
    world_shape = (4, 4)

    # env = GridWorldEnv(grid_shape=world_shape, terminal_states=[3, 12])
    # todo even maze gridsize mazes messes this up
    world_shape = (21, 31)  # odd mazes only
    world_shape = (11, 11)  # odd mazes only
    env = GridWorldEnv(grid_shape=world_shape, random_maze=True)
    policy0 = np.ones([env.world.size, len(env.action_state_to_next_state)]) / len(env.action_state_to_next_state)
    v0 = np.zeros(env.world.size)
    val_fun = v0
    for k in range(500):
        val_fun = utils.single_step_policy_evaluation(policy0, env, value_function=val_fun)
    print(utils.reshape_as_gridworld(val_fun, world_shape))

    # comment: mention that for DP there is no concept of an agent.
    # todo why do the walls of policy map have arrows? don't draw them?
    # todo print everthing below better

    env.render(mode='graphic')

    # test greedy policy
    policy1 = utils.greedy_policy_from_value_function(policy0, env, val_fun)
    policy_map1 = utils.get_policy_map(policy1, world_shape)
    print('Policy: (0=up, 1=right, 2=down, 3=left)\n', policy_map1)
    np.set_printoptions(linewidth=75 * 2, precision=4)
    print('Policy: (up, right, down, left)\n', utils.get_policy_map(policy1, world_shape)) #todo shouldn't have two outputs! better name or bettr handling
    np.set_printoptions(linewidth=75, precision=8)

    # test policy iteration
    # print('Policy iteration:')
    # policy0 = np.ones([env.world.size, len(env.action_state_to_next_state)]) / len(env.action_state_to_next_state)
    # optimal_value, optimal_policy = dp.policy_iteration(policy0, env, v0, threshold=0.001, max_steps=1000)
    # print('Value:\n', utils.reshape_as_gridworld(optimal_value, world_shape))
    # print('Policy: (0=up, 1=right, 2=down, 3=left)\n', utils.get_policy_map(optimal_policy, world_shape))
    # np.set_printoptions(linewidth=75 * 2, precision=4)
    # print('Policy: (up, right, down, left)\n', utils.get_policy_map(optimal_policy, world_shape))
    # np.set_printoptions(linewidth=75, precision=8)
    #
    # # test value iteration
    # print('Value iteration:')
    # policy0 = np.ones([env.world.size, len(env.action_state_to_next_state)]) / len(env.action_state_to_next_state)
    # optimal_value, optimal_policy = dp.value_iteration(policy0, env, v0, threshold=0.001, max_steps=100)
    # print('Value:\n', utils.reshape_as_gridworld(optimal_value, world_shape))
    # print('Policy: (0=up, 1=right, 2=down, 3=left)\n', utils.get_policy_map(optimal_policy, world_shape))
    # np.set_printoptions(linewidth=75 * 2, precision=4)
    # print('Policy: (up, right, down, left)\n', utils.get_policy_map(optimal_policy, world_shape))
    # np.set_printoptions(linewidth=75, precision=8)

    curr_state = env.reset()

    # has to be done after after env.render(mode='graphic')
    policy_arrow_array, policy_probabilities = utils.get_policy_map(policy1, world_shape)

    # todo make an env function not viewer.
    env.viewer.calculate_policy_lines(policy1)

    for t in range(100):
        env.render(mode='graphic')

        action = np.argmax(policy1[curr_state])
        print('go ' + env.action_descriptors[action])
        curr_state, reward, done, info = env.step(action)

        if done:
            print('DONE in {} steps'.format(t + 1))
            env.render(mode='graphic') # must render here to see agent in final state
            time.sleep(5)
            break

def run_monte_carlo():
    print('\n' + '*' * 20 + 'Starting Monte Carlo evaluation and greedy policy' + '*' * 20 + '\n')
    env = GridWorldEnv()
    # env = GridWorldEnv(grid_shape=(40, 40), random_maze=True)
    policy0 = np.ones([env.world.size, env.action_space.n]) / env.action_space.n

    print('Running an episode with a random agent (with initial policy)')
    st_history, rw_history = run_episode(policy0, env)
    print('States history: ' + str(st_history))
    print('Rewards history: ' + str(rw_history))

    print('Starting Monte-Carlo evaluation of random policy')
    value0 = monte_carlo_evaluation(policy0, env, every_visit=True, stationary_env=False)
    print(value0)
    for state, value in value0.items():
        print(state, value)

    # Create greedy policy from value function and run it on environment
    world_shape = (4, 4)
    policy1 = utils.greedy_policy_from_value_function(policy0, env, value0)
    print(policy1)

    print('Policy: (up, right, down, left)\n', utils.get_policy_map(policy1, world_shape))
    np.set_printoptions(linewidth=75, precision=8)

    print('Starting greedy policy episode')
    curr_state = env.reset()

    for t in range(100):
        env.render(mode='graphic')

        action = np.argmax(policy1[curr_state])
        print('go ' + env.action_descriptors[action])
        curr_state, reward, done, info = env.step(action)

        if done:
            print('DONE in {} steps'.format(t + 1))
            break

if __name__ == '__main__':
    # Run random agent on environment variations
    # run_and_create_gridworld_from_text_file()
    run_random_maze()

    # Run specific algorithms on gridworld
    # run_monte_carlo()
    # run_policy_iteration_gridworld()
