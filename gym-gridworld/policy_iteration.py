import numpy as np
from gym_gridworld.envs.gridworld_env import GridWorldEnv


def reshape_as_gridworld(input_matrix):
    """
    Helper function to reshape a gridworld state matrix into a visual representation of the gridworld with origin on the
    low left corner and x,y corresponding to cartesian coordinates.
    """
    return np.reshape(input_matrix, (world_shape[0], world_shape[1]))[:, ::-1].T


def single_step_policy_evaluation(policy, env, discount_factor=1.0, **kwargs):
    """
    Returns an update of the input value function using the input policy.
    """
    if 'value_function' in kwargs:
        v = kwargs['value_function']
    else:
        v = np.zeros(env.world.size)
    v_new = np.zeros(env.world.size)

    for state in range(env.world.size):
        v_new[state] += env.reward_matrix[state]
        for action, action_prob in enumerate(policy[state]):
            next_state, reward, done = env.look_step_ahead(state, action)
            v_new[state] += action_prob * (discount_factor * v[next_state])
    return v_new


def get_policy_map(policy):
    """
    Generates a visualization grid from the policy to be able to print which action is most likely from every state
    """
    policy_map = np.fromiter((np.argmax(policy[state]) for state in np.arange(policy.shape[0])), dtype=np.int64)
    return reshape_as_gridworld(policy_map)


def greedy_policy_from_value_function(policy, env, discount_factor=1.0, **kwargs):
    """
    Returns a greedy policy based on the input value function.

    If no value function was provided the defaults from a single step starting with a value function of zeros
    will be used.
    """
    v = single_step_policy_evaluation(policy, env, discount_factor, **kwargs)

    for state in range(env.world.size):
        action_values = np.zeros(env.action_space.n)
        for action in range(env.action_space.n):
            next_state, reward, done = env.look_step_ahead(state, action)
            action_values[action] += policy[state][action] * (reward + discount_factor * v[next_state])
        best_action = np.argmax(action_values)  # TODO: we have to select all max with the same prob, this is wrong
        policy[state] = np.eye(env.action_space.n)[best_action]
    return policy


def policy_iteration(policy, env, threshold=0.00001, **kwargs):
    """
    Policy iteration algorithm, which consists on iteratively evaluating a policy and updating it greedily with
    respect to the value function obtained from a single step evaluation.
    """
    delta = 0
    while True:
        policy_value = single_step_policy_evaluation(policy, env, **kwargs)
        greedy_policy = greedy_policy_from_value_function(policy_value, env, **kwargs)

        if delta < threshold:
            break
    return policy_value, greedy_policy


if __name__ == '__main__':
    # test policy evaluation
    world_shape = (4, 4)
    gw_env = GridWorldEnv(grid_shape=world_shape, terminal_states=[3, 12])
    policy0 = np.ones([gw_env.world.size, len(gw_env.actions_list)]) / len(gw_env.actions_list)
    v0 = np.zeros(gw_env.world.size)
    val_fun = v0
    for k in range(10):
        val_fun = single_step_policy_evaluation(policy0, gw_env, value_function=val_fun)
    print(reshape_as_gridworld(val_fun))

    # test greedy policy
    policy1 = greedy_policy_from_value_function(policy0, gw_env)
    policy_map1 = get_policy_map(policy1)
    print('Policy:\n', policy_map1)
