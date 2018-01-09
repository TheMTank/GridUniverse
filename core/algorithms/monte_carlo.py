import sys

import numpy as np
from six import StringIO


def run_episode(policy, env, max_steps_per_episode=100):
    """
    Generates an agent and runs actions until the agent either gets to a terminal state or executes a number of 
    max_steps_per_episode steps.
    
    Assumes a stochastic policy and takes an action sample taken from a distribution with the probabilities given
    by the policy.
    """
    states_hist = []
    rewards_hist = []
    observation = env.reset()
    for step in range(max_steps_per_episode):
        action = np.random.choice(policy[observation].size, p=policy[observation])
        observation, reward, done, info = env.step(action)
        states_hist.append(observation)
        rewards_hist.append(reward)
        if done:
            break
    return states_hist, rewards_hist


def monte_carlo_evaluation(policy, env, every_visit=False, incremental_mean=True, stationary_env=True,
                           discount_factor=0.01, threshold=0.0001, alpha=0.001, n_episodes=100):
    """"
    Monte Carlo algorithm, which solves the MDP learning from full episodes without the need of a model.

    It is implemented in three variants depending on how the value function is calculated.

    --- Monte-Carlo
    1. Store visit counts for each state from last episode (First Visit MC vs Every Visit MC count differently)
    2. If incremental mean, update total_visit_counter and using that and returns from last episode update
       value function at the end of each episode.
    3. If not using incremental mean store total return and update total_visit_counter for each state over all episodes
       (again depending on First Visit MC vs Every Visit MC)
       and using this, update value function after all episodes have been run
    4. If non-stationary environment use alpha instead of (1 / (total_visit_counter[state]))
    """

    # could do "total_visit_counter = np.zeros(env.world.size)" but that isn't model-free
    # This way we can run on procedurally generated environments that grown in size
    total_visit_counter = {}
    total_return = {}
    value_function = {}
    for episode in range(n_episodes):
        episode_state_hist, episode_reward_hist = run_episode(policy, env)

        visits_from_last_episode = {}
        returns_from_last_episode = {}
        # Store visit counts and returns from each state from last episode depending on whether first or every visit MC
        for idx, (state, reward) in enumerate(zip(episode_state_hist, episode_reward_hist)):
            if state not in visits_from_last_episode:
                visits_from_last_episode[state] = 0
                returns_from_last_episode[state] = 0.0
            else:
                if not every_visit:
                    # State was already seen before in episode, don't add more return to that state if First Visit MC
                    # Also don't increase it's visit count
                    continue

            # If first-visit MC, the code below only runs for the first encounter of a state within an episode
            # Calculation depends heavily if whether it is first or every visit MC.
            # Update visit counter for last episode
            visits_from_last_episode[state] += 1
            # Update return for last episode for a specific state
            return_from_state = sum([(discount_factor ** i) * r for i, r in enumerate(episode_reward_hist[idx:])
                                                  if (discount_factor ** i) < threshold])
            returns_from_last_episode[state] += return_from_state

        # Update total_visit_counter, total_return, and value function for each state
        for state in visits_from_last_episode:
            # if state hasn't been seen before, setup defaults
            if state not in total_visit_counter:
                total_visit_counter[state] = 0
                total_return[state] = 0.0
                value_function[state] = 0.0

            # Update visits from last episode
            total_visit_counter[state] += visits_from_last_episode[state]
            # Update total return. Only used for non-incremental mean.
            if not incremental_mean:
                # Increment total return S(s) ← S(s) + Gt
                total_return[state] += returns_from_last_episode[state]
            else:
                # Update value function only if incremental mean. Otherwise do at end
                if stationary_env:
                    # V(St) ← V(St) + 1/ N(St) * (Gt − V(St))
                    value_function[state] += (1 / total_visit_counter[state]) * (returns_from_last_episode[state] - value_function[state])
                else:
                    """
                    In non-stationary problems, it can be useful to track a running mean, i.e. forget old episodes.
                    V(St) ← V(St) + α(Gt − V(St))
                    """
                    value_function[state] += alpha * (returns_from_last_episode[state] - value_function[state])

    if not incremental_mean:
        # Value is estimated by mean return V(s) = S(s) / N(s) if not incremental mean
        for state in total_visit_counter:
            value_function[state] = total_return[state] / total_visit_counter[state]

    return value_function
