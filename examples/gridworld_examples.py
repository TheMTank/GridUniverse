import numpy as np

from core.envs.gridworld_env import GridWorldEnv
from core.algorithms import utils
import core.algorithms.dynamic_programming as dp


def run_random_gridworld():
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

def run_policy_iteration_gridworld():
    # test policy evaluation
    world_shape = (4, 4)
    env = GridWorldEnv(grid_shape=world_shape, terminal_states=[3, 12])
    policy0 = np.ones([env.world.size, len(env.action_state_to_next_state)]) / len(env.action_state_to_next_state)
    v0 = np.zeros(env.world.size)
    val_fun = v0
    for k in range(500):
        val_fun = utils.single_step_policy_evaluation(policy0, env, value_function=val_fun)
    print(utils.reshape_as_gridworld(val_fun, world_shape))

    # test greedy policy
    policy1 = utils.greedy_policy_from_value_function(policy0, env, val_fun)
    policy_map1 = utils.get_policy_map(policy1, world_shape)
    print('Policy: (0=up, 1=right, 2=down, 3=left)\n', policy_map1)
    np.set_printoptions(linewidth=75 * 2, precision=4)
    print('Policy: (up, right, down, left)\n', utils.get_policy_map(policy1, world_shape))
    np.set_printoptions(linewidth=75, precision=8)

    # test policy iteration
    print('Policy iteration:')
    policy0 = np.ones([env.world.size, len(env.action_state_to_next_state)]) / len(env.action_state_to_next_state)
    optimal_value, optimal_policy = dp.policy_iteration(policy0, env, v0, threshold=0.001, max_steps=1000)
    print('Value:\n', utils.reshape_as_gridworld(optimal_value, world_shape))
    print('Policy: (0=up, 1=right, 2=down, 3=left)\n', utils.get_policy_map(optimal_policy, world_shape))
    np.set_printoptions(linewidth=75 * 2, precision=4)
    print('Policy: (up, right, down, left)\n', utils.get_policy_map(optimal_policy, world_shape))
    np.set_printoptions(linewidth=75, precision=8)

    # test value iteration
    print('Value iteration:')
    policy0 = np.ones([env.world.size, len(env.action_state_to_next_state)]) / len(env.action_state_to_next_state)
    optimal_value, optimal_policy = dp.value_iteration(policy0, env, v0, threshold=0.001, max_steps=100)
    print('Value:\n', utils.reshape_as_gridworld(optimal_value, world_shape))
    print('Policy: (0=up, 1=right, 2=down, 3=left)\n', utils.get_policy_map(optimal_policy, world_shape))
    np.set_printoptions(linewidth=75 * 2, precision=4)
    print('Policy: (up, right, down, left)\n', utils.get_policy_map(optimal_policy, world_shape))
    np.set_printoptions(linewidth=75, precision=8)


if __name__ == '__main__':
    run_random_gridworld()
    run_policy_iteration_gridworld()
