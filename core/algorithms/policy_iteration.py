import numpy as np
from core.envs.gridworld_env import GridWorldEnv
from six import StringIO
import sys
import warnings
from core.algorithms import utils


def value_iteration(policy, env, value_function=None, threshold=0.00001, max_steps=1000, **kwargs):
    """
    Value iteration algorithm, which consists on one sweep of policy evaluation (no convergence) and then one policy
    greedy update. These two steps are repeated until convergence.
    """
    value_function = np.zeros(env.world.size) if value_function is None else value_function
    greedy_policy = policy
    for step_number in range(max_steps):
        new_value_function = utils.single_step_policy_evaluation(greedy_policy, env, value_function=value_function, **kwargs)
        delta = np.max(value_function - new_value_function)
        value_function = new_value_function

        greedy_policy = utils.greedy_policy_from_value_function(greedy_policy, env, value_function=value_function, **kwargs)

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
        new_value_function = utils.single_step_policy_evaluation(greedy_policy, env, value_function=value_function, **kwargs)
        delta_eval = np.max(value_function - new_value_function)
        value_function = new_value_function
        if delta_eval < threshold:  # policy evaluation converged
            new_policy = utils.greedy_policy_from_value_function(greedy_policy, env, value_function=value_function, **kwargs)
            delta = np.max(last_converged_v_fun - new_value_function)
            last_converged_v_fun = new_value_function
            if delta < threshold:  # last converged value functions difference converged
                break
            else:
                greedy_policy = new_policy

        elif step_number == max_steps - 1:
            greedy_policy = utils.greedy_policy_from_value_function(greedy_policy, env, value_function=last_converged_v_fun,
                                                              **kwargs)
            warning_message = 'Policy iteration did not reach the selected threshold. Finished after reaching ' \
                              'the maximum {} steps with delta_eval {}'.format(step_number + 1, delta_eval)
            warnings.warn(warning_message, UserWarning)
    return last_converged_v_fun, greedy_policy
