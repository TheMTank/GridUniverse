import numpy as np
from gym_gridworld.envs.gridworld_env import GridWorldEnv
import warnings


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
                           discount_factor=0.01, threshold=0.0001, n_episodes=100):
    """"
    Monte Carlo algorithm, which solves the MDP learning from full episodes without the need of a model.

    It is implemented in three variants depending on how the value function is calculated.
    """
    # visit_counter = total_return = np.zeros(env.world.size) # do it model-free
    visit_counter = {}
    total_return = {}
    value_function = {}
    # episodes_states_history = np.empty([1], dtype=np.int)
    # episodes_reward_history = np.empty([1], dtype=np.int32)
    all_episode_state_hist = []
    all_episode_reward_hist = []
    for episode in range(n_episodes):
        episode_state_hist, episode_reward_hist = run_episode(policy, env)
        # np.append(all_episode_state_hist, episode_state_hist)
        # np.append(all_episode_reward_hist, episode_reward_hist)
        # all_episode_state_hist.extend(episode_state_hist)
        # all_episode_reward_hist.extend(episode_reward_hist)

        # todo after each episode: every visit
        # todo 1. store visit counts for each state
        # todo 2. store total return for each visit
        # todo 3. Update V(St)

        # Store visit counts for each state from last episode
        for state in episode_state_hist:
            if state not in visit_counter:
                visit_counter[state] = 1
                total_return[state] = 0
            else:
                visit_counter[state] += 1

        returns_from_last_episode = {}
        # Store returns from each state from last episode
        for idx, (state, reward) in enumerate(zip(episode_state_hist, episode_reward_hist)):
            # return_from_episode_from_state = sum([(discount_factor ** i) * r for i, r in enumerate(episode_reward_hist[idx:])
            #                                     if (discount_factor ** i) < threshold])
            # total_return[state] += return_from_episode_from_state
            return_from_episode_from_state = sum([(discount_factor ** i) * r for i, r in enumerate(episode_reward_hist[idx:])
                                                if (discount_factor ** i) < threshold])
            if state not in returns_from_last_episode:
                return_from_episode_from_state[state] = 0.0
            else:
                if not every_visit:
                    continue # State was already seen before in episode, don't add more return to that state

            returns_from_last_episode[state] += return_from_episode_from_state # Add if statement if first visit


            # for i in range(episodes_states.size - idx):
            #     # break the loop if the next state contribution is below the threshold
            #     if (discount_factor ** i) > threshold: # bug should be less than?
            #         total_return[state] += (discount_factor ** i) * episodes_reward[i]
            #     else:
            #         break



        # if over: V(s) = S(s) / N(s)


    #     # count visits
    #     if not every_visit:
    #         raise NotImplementedError
    #         # TODO: this part is wrong
    #         # episodes_states = set(episodes_states)
    #         # unique_episodes_states, indeces = np.unique(episodes_states, return_inverse=True)
    #
    #     visit_counter += np.fromiter((1 if state in episodes_states else 0
    #                                   for state in np.nditer(np.arange(value_function.size))), dtype=np.uint16)
    #
    #     if incremental_mean:
    #         # add observation values dynamically
    #         """
    #         V(St) ← V(St) + 1/ N(St) * (Gt − V(St))
    #
    #         Currently only working for every_visit MC
    #         """
    #         for idx, state in enumerate(episodes_states):
    #             for i in range(episodes_states.size - idx):
    #                 # break the loop if the next state contribution is below the threshold
    #                 if (discount_factor ** i) > threshold:
    #                     total_return[state] += (discount_factor ** i) * episodes_reward[i]
    #                 else:
    #                     break
    #             value_function[state] += (total_return[state] - value_function[state]) / visit_counter[state]
    #     else:
    #         """
    #         Increment total return S(s) ← S(s) + Gt
    #         """
    #         raise NotImplementedError
    #         #  TODO: wrong, we need bellman eq
    #         # add observations to the total return for a static mean
    #         # total_return[(visit_counter > 0)] += reward_history[(visit_counter > 0)]
    #
    #     if not stationary_env:
    #         """
    #         In non - stationary problems, it can be useful to track a running mean, i.e.forget old episodes.
    #         V(St) ← V(St) + α(Gt − V(St))
    #         """
    #         raise NotImplementedError
    #
    # if not incremental_mean:
    #     """
    #     Value is estimated by mean return V(s) = S(s) / N(s)
    #     """
    #     raise NotImplementedError
    #     # total_return[(visit_counter > 0)] += reward_history[(visit_counter > 0)]  # TODO: wrong, we need bellman eq
    #     # value_function += total_return/visit_counter

    return value_function


if __name__ == '__main__':
    env = GridWorldEnv()
    policy0 = np.ones([env.world.size, env.action_space.n]) / env.action_space.n
    st_history, rw_history = run_episode(policy0, env)
    print('States history: ' + str(st_history))
    print('Rewards history: ' + str(rw_history))
    # value0 = monte_carlo_evaluation(policy0, gw_env, True)
    # print(value0)
