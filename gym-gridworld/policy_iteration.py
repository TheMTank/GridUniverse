import numpy as np
from gym_gridworld.envs.gridworld_env import GridWorldEnv
import warnings


def reshape_as_gridworld(input_matrix):
    """
    Helper function to reshape a gridworld state matrix into a visual representation of the gridworld with origin on the
    low left corner and x,y corresponding to cartesian coordinates.
    """
    return np.reshape(input_matrix, (world_shape[0], world_shape[1]))[:, ::-1].T


def single_step_policy_evaluation(policy, env, discount_factor=1.0, value_function=None):
    """
    Returns an update of the input value function using the input policy.
    """
    v = np.zeros(env.world.size) if value_function is None else value_function
    v_new = np.zeros(env.world.size)

    for state in range(env.world.size):
        v_new[state] += env.reward_matrix[state]
        for action, action_prob in enumerate(policy[state]):
            next_state, reward, done = env.look_step_ahead(state, action)
            v_new[state] += action_prob * (discount_factor * v[next_state])
    return v_new


def get_policy_map(policy, max_only=True):
    """
    Generates a visualization grid from the policy to be able to print which action is most likely from every state
    """
    if max_only:
        policy_map = np.fromiter((np.argmax(policy[state]) for state in np.nditer(np.arange(policy.shape[0]))),
                                 dtype=np.int64)
    else:
        policy_map = np.fromiter((policy[state] for state in np.nditer(np.arange(policy.shape[0]))),
                                 dtype='float64, float64, float64, float64')
    return reshape_as_gridworld(policy_map)


def greedy_policy_from_value_function(policy, env, value_function, discount_factor=1.0):
    """
    Returns a greedy policy based on the input value function.

    If no value function was provided the defaults from a single step starting with a value function of zeros
    will be used.
    """
    q_function = np.zeros((env.world.size, env.action_space.n))
    for state in range(env.world.size):
        for action in range(env.action_space.n):
            next_state, reward, done = env.look_step_ahead(state, action)
            q_function[state][action] += reward + discount_factor * value_function[next_state]
        max_value_actions = np.where(np.around(q_function[state], 8) == np.around(np.amax(q_function[state]), 8))[0]

        policy[state] = np.fromiter((1 / len(max_value_actions) if action in max_value_actions and
                                    not env.is_terminal(state) else 0
                                     for action in np.nditer(np.arange(env.action_space.n))), dtype=np.float)
    return policy


def value_iteration(policy, env, value_function=None, threshold=0.00001, max_steps=1000, **kwargs):
    """
    Value iteration algorithm, which consists on one sweep of policy evaluation (no convergence) and then one policy
    greedy update. These two steps are repeated until convergence.
    """
    value_function = np.zeros(env.world.size) if value_function is None else value_function
    greedy_policy = policy
    for step_number in range(max_steps):
        new_value_function = single_step_policy_evaluation(greedy_policy, env, value_function=value_function, **kwargs)
        delta = np.max(value_function - new_value_function)
        value_function = new_value_function

        greedy_policy = greedy_policy_from_value_function(greedy_policy, env, value_function=value_function, **kwargs)

        if delta < threshold:
            break
        elif step_number == max_steps - 1:
            warning_message = 'Value iteration did not reach the selected threshold. Finished after reaching ' \
                              'the maximum {} steps'.format(step_number + 1)
            warnings.warn(warning_message, UserWarning)
    return value_function, greedy_policy


def policy_iteration(policy, env, value_function=None, threshold=0.00001, max_steps=1000, **kwargs):
    """
    Policy iteration algorithm, which consists on iterative policy evaluation until convergence for the current policy
    (estimate over many sweeps until you can't estimate no more). And then finally updates policy to be greedy.
    """
    value_function = last_converged_v_fun = np.zeros(env.world.size) if value_function is None else value_function
    greedy_policy = policy
    for step_number in range(max_steps):
        new_value_function = single_step_policy_evaluation(greedy_policy, env, value_function=value_function, **kwargs)
        delta_eval = np.max(value_function - new_value_function)
        value_function = new_value_function
        if delta_eval < threshold:  # policy evaluation converged
            new_policy = greedy_policy_from_value_function(greedy_policy, env, value_function=value_function, **kwargs)
            delta = np.max(last_converged_v_fun - new_value_function)
            last_converged_v_fun = new_value_function
            if delta < threshold:  # last converged value functions difference converged
                break
            else:
                greedy_policy = new_policy

        elif step_number == max_steps - 1:
            greedy_policy = greedy_policy_from_value_function(greedy_policy, env, value_function=last_converged_v_fun,
                                                              **kwargs)
            warning_message = 'Policy iteration did not reach the selected threshold. Finished after reaching ' \
                              'the maximum {} steps with delta_eval {}'.format(step_number + 1, delta_eval)
            warnings.warn(warning_message, UserWarning)
    return last_converged_v_fun, greedy_policy


if __name__ == '__main__':
    # test policy evaluation
    world_shape = (4, 4)
    gw_env = GridWorldEnv(grid_shape=world_shape, terminal_states=[3, 12])
    policy0 = np.ones([gw_env.world.size, len(gw_env.actions_list)]) / len(gw_env.actions_list)
    v0 = np.zeros(gw_env.world.size)
    val_fun = v0
    for k in range(500):
        val_fun = single_step_policy_evaluation(policy0, gw_env, value_function=val_fun)
    print(reshape_as_gridworld(val_fun))

    # test greedy policy
    policy1 = greedy_policy_from_value_function(policy0, gw_env, val_fun)
    policy_map1 = get_policy_map(policy1)
    print('Policy: (0=up, 1=right, 2=down, 3=left)\n', policy_map1)
    np.set_printoptions(linewidth=75*2, precision=4)
    print('Policy: (up, right, down, left)\n', get_policy_map(policy1, max_only=False))
    np.set_printoptions(linewidth=75, precision=8)

    # test policy iteration
    print('Policy iteration:')
    policy0 = np.ones([gw_env.world.size, len(gw_env.actions_list)]) / len(gw_env.actions_list)
    optimal_value, optimal_policy = policy_iteration(policy0, gw_env, v0, threshold=0.001, max_steps=1000)
    print('Value:\n', reshape_as_gridworld(optimal_value))
    print('Policy: (0=up, 1=right, 2=down, 3=left)\n', get_policy_map(optimal_policy))
    np.set_printoptions(linewidth=75*2, precision=4)
    print('Policy: (up, right, down, left)\n', get_policy_map(optimal_policy, max_only=False))
    np.set_printoptions(linewidth=75, precision=8)

    # test value iteration
    print('Value iteration:')
    policy0 = np.ones([gw_env.world.size, len(gw_env.actions_list)]) / len(gw_env.actions_list)
    optimal_value, optimal_policy = value_iteration(policy0, gw_env, v0, threshold=0.001, max_steps=100)
    print('Value:\n', reshape_as_gridworld(optimal_value))
    print('Policy: (0=up, 1=right, 2=down, 3=left)\n', get_policy_map(optimal_policy))
    np.set_printoptions(linewidth=75*2, precision=4)
    print('Policy: (up, right, down, left)\n', get_policy_map(optimal_policy, max_only=False))
    np.set_printoptions(linewidth=75, precision=8)
