import numpy as np
from gym_gridworld.envs.gridworld_env import GridWorldEnv
import warnings


def run_episode(policy):
    return np.zeros(1), np.zeros(1)


def monte_carlo(policy, env, every_visit=False, incremental_mean=True, stationary_env=False, num_episodes=100):
    """"
    Monte Carlo algorithm, which solves the MDP learning from full episodes without the need of a model.

    It is implemented in three variants depending on how the value function is calculated.
    """
    visit_counter = value_function = total_return = np.zeros(env.world.size)
    for episode in range(num_episodes):
        states_history, reward_history = run_episode(policy)
        # count visits
        if not every_visit:
            states_history = set(states_history)

        visit_counter += np.fromiter((1 if state in states_history else 0
                                      for state in np.nditer(np.arange(value_function.size))), dtype='unit16')

        if incremental_mean:
            # add observation values dynamically
            """
            V(St) ← V(St) + 1/ N(St) * (Gt − V(St))"""
        else:  # TODO: Do we want to have this possiblity?
            """
            Increment total return S(s) ← S(s) + Gt
            """
            # add observations to the total return for a static mean
            total_return[(visit_counter > 0)] += reward_history[(visit_counter > 0)] # TODO: wrong, we need bellman equation

        if not stationary_env:
            """
            In non - stationary problems, it can be useful to track a running mean, i.e.forget old episodes.
            V(St) ← V(St) + α(Gt − V(St))
            """

    if not incremental_mean:
        """
        Value is estimated by mean return V(s) = S(s) / N(s)
        """
        total_return[(visit_counter > 0)] += reward_history[(visit_counter > 0)]
        value_function += total_return/visit_counter

    return value_function


