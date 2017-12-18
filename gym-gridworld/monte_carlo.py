import numpy as np
from gym_gridworld.envs.gridworld_env import GridWorldEnv
import warnings


def run_episode(policy, env, max_steps_per_episode=100):
    """
    Generates an agent and runs actions until the agent either gets to a terminal state or executes a number of 
    max_steps_per_episode steps.
    
    Assumes a stochastic policy and acts with a sample taken from a normal distribution with the probabilities given
    by the policy.
    """
    states_history = np.empty([1], dtype=np.int)
    rewards_history = np.empty([1], dtype=np.int32)
    observation = env.reset()
    for step in range(max_steps_per_episode):
        action = np.random.choice(policy[observation].size, p=policy[observation])
        observation, reward, done, info = env.step(action)
        np.append(states_history, observation)
        np.append(rewards_history, reward)
        if done:
            break
    return states_history, rewards_history


def monte_carlo_evaluation(policy, env, every_visit=False, incremental_mean=True, stationary_env=True,
                           discount_factor=0.01, n_episodes=100):
    """"
    Monte Carlo algorithm, which solves the MDP learning from full episodes without the need of a model.

    It is implemented in three variants depending on how the value function is calculated.
    """
    visit_counter = value_function = total_return = np.zeros(env.world.size)
    episodes_states_history = np.empty([1], dtype=np.int)
    episodes_reward_history = np.empty([1], dtype=np.int32)
    for episode in range(n_episodes):
        episodes_states, episodes_reward = run_episode(policy, env)
        np.append(episodes_states_history, episodes_states)
        np.append(episodes_reward_history, episodes_reward)
        # count visits
        if not every_visit:
            raise NotImplementedError
            # TODO: this part is wrong
            # episodes_states = set(episodes_states)
            # unique_episodes_states, indeces = np.unique(episodes_states, return_inverse=True)

        visit_counter += np.fromiter((1 if state in episodes_states else 0
                                      for state in np.nditer(np.arange(value_function.size))), dtype=np.uint16)

        if incremental_mean:
            # add observation values dynamically
            """
            V(St) ← V(St) + 1/ N(St) * (Gt − V(St))
            
            Currently only working for every_visit MC
            """
            for idx, state in enumerate(episodes_states):
                for i in range(episodes_states.size - idx):
                    # TODO: add threshold for multiplication (discount_factor gets lower every time)
                    total_return[state] += (discount_factor ** i) * episodes_reward[i]
                value_function[state] += (total_return[state] - value_function[state]) / visit_counter[state]
        else:
            """
            Increment total return S(s) ← S(s) + Gt
            """
            raise NotImplementedError
            #  TODO: wrong, we need bellman eq
            # add observations to the total return for a static mean
            # total_return[(visit_counter > 0)] += reward_history[(visit_counter > 0)]

        if not stationary_env:
            """
            In non - stationary problems, it can be useful to track a running mean, i.e.forget old episodes.
            V(St) ← V(St) + α(Gt − V(St))
            """
            raise NotImplementedError

    if not incremental_mean:
        """
        Value is estimated by mean return V(s) = S(s) / N(s)
        """
        raise NotImplementedError
        # total_return[(visit_counter > 0)] += reward_history[(visit_counter > 0)]  # TODO: wrong, we need bellman eq
        # value_function += total_return/visit_counter

    return value_function


if __name__ == '__main__':
    gw_env = GridWorldEnv()
    policy0 = np.ones([gw_env.world.size, gw_env.action_space.n]) / gw_env.action_space.n
    st_history, rw_history = run_episode(policy0, gw_env)
    print('States history: ' + str(st_history))
    print('Rewards history: ' + str(rw_history))
    value0 = monte_carlo_evaluation(policy0, gw_env, True)
    print(value0)