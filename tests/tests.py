import numpy as np

from core.envs.gridworld_env import GridWorldEnv
from core.algorithms.policy_iteration import (single_step_policy_evaluation, reshape_as_gridworld, get_policy_map,
                                greedy_policy_from_value_function, policy_iteration, value_iteration)


def test_policy_iteration():
    # test policy evaluation
    world_shape = (4, 4)
    gw_env = GridWorldEnv(grid_shape=world_shape, terminal_states=[3, 12])
    policy0 = np.ones([gw_env.world.size, len(gw_env.action_state_to_next_state)]) / len(gw_env.action_state_to_next_state)
    v0 = np.zeros(gw_env.world.size)
    val_fun = v0
    for k in range(500):
        val_fun = single_step_policy_evaluation(policy0, gw_env, value_function=val_fun)
    print(reshape_as_gridworld(val_fun, world_shape))

    # test greedy policy
    policy1 = greedy_policy_from_value_function(policy0, gw_env, val_fun)
    policy_map1 = get_policy_map(policy1, world_shape)
    print('Policy: (0=up, 1=right, 2=down, 3=left)\n', policy_map1)
    np.set_printoptions(linewidth=75 * 2, precision=4)
    print('Policy: (up, right, down, left)\n', get_policy_map(policy1, world_shape))
    np.set_printoptions(linewidth=75, precision=8)

    # test policy iteration
    print('Policy iteration:')
    policy0 = np.ones([gw_env.world.size, len(gw_env.action_state_to_next_state)]) / len(gw_env.action_state_to_next_state)
    optimal_value, optimal_policy = policy_iteration(policy0, gw_env, v0, threshold=0.001, max_steps=1000)
    print('Value:\n', reshape_as_gridworld(optimal_value, world_shape))
    print('Policy: (0=up, 1=right, 2=down, 3=left)\n', get_policy_map(optimal_policy, world_shape))
    np.set_printoptions(linewidth=75 * 2, precision=4)
    print('Policy: (up, right, down, left)\n', get_policy_map(optimal_policy, world_shape))
    np.set_printoptions(linewidth=75, precision=8)

    # test value iteration
    print('Value iteration:')
    policy0 = np.ones([gw_env.world.size, len(gw_env.action_state_to_next_state)]) / len(gw_env.action_state_to_next_state)
    optimal_value, optimal_policy = value_iteration(policy0, gw_env, v0, threshold=0.001, max_steps=100)
    print('Value:\n', reshape_as_gridworld(optimal_value, world_shape))
    print('Policy: (0=up, 1=right, 2=down, 3=left)\n', get_policy_map(optimal_policy, world_shape))
    np.set_printoptions(linewidth=75 * 2, precision=4)
    print('Policy: (up, right, down, left)\n', get_policy_map(optimal_policy, world_shape))
    np.set_printoptions(linewidth=75, precision=8)


def test_random_gridworld():
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


if __name__ == '__main__':
    test_random_gridworld()
    test_policy_iteration()
