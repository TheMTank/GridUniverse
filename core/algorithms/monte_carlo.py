import sys

import numpy as np
from six import StringIO


def run_episode(policy, env, max_steps_per_episode=1000):
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
    return states_hist, rewards_hist, done


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

    total_visit_counter = np.zeros(env.world.size)
    total_return = np.zeros(env.world.size)
    value_function = np.zeros(env.world.size)
    for episode in range(n_episodes):
        episode_state_hist, episode_reward_hist, done = run_episode(policy, env)
        print('Episode: {}, terminal found: {}'.format(episode, done))

        visits_from_last_episode = np.zeros(env.world.size)
        returns_from_last_episode = np.zeros(env.world.size)
        # Store visit counts and returns from each state from last episode depending on whether first or every visit MC
        for idx, (state, reward) in enumerate(zip(episode_state_hist, episode_reward_hist)):
            if visits_from_last_episode[state] == 0:
                pass # calculate first-first below
            elif not every_visit:
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
        for state in range(visits_from_last_episode.size):
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
