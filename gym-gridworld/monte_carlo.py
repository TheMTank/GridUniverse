import numpy as np
from gym_gridworld.envs.gridworld_env import GridWorldEnv
import warnings


def run_episode(policy):
    return np.zeros(1), 0


def monte_carlo_first_visit(policy, env, visit_flag='first-visit'):
    """"
    Monte Carlo algorithm, which solves the MDP learning from full episodes without the need of a model.

    It is implemented in three variants depending on how the value function is calculated.
    """
    value_function = visit_counter = np.zeros(env.world.size)
    states_history, total_reward = run_episode(policy)
    for state in np.nditer(np.arange(value_function.size)):
        if visit_flag == 'first-visit':
            visit_counter[state] = 1 if state in states_history else 0
            value_function += value_function[(visit_counter == 1)]
        elif visit_flag == 'every-visit':
            raise NotImplementedError
        elif visit_flag == 'incremental':
            raise NotImplementedError
        else:
            raise ValueError('Invalid visit flag. Please choose one of the following values: \'first-visit\', '
                             '\'every-visit\' or \'incremental\'')
    return value_function


