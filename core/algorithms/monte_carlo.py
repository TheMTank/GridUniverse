import time
import sys
import math

import numpy as np
from six import StringIO


import torch
import torch.nn as nn
from torch.autograd import Variable
import matplotlib.pyplot as plt

from core.algorithms import utils

# Linear Regression Model
class LinearRegression(nn.Module):
    def __init__(self, input_size, output_size):
        super(LinearRegression, self).__init__()
        self.linear = nn.Linear(input_size, output_size)

    def forward(self, x):
        out = self.linear(x)
        return out


def create_one_hot_state_vector(env, state):
    vec = np.zeros(env.world.size)
    vec[state] = 1
    return vec

def test_linear_approx(env):
    value_function = np.zeros(env.world.size)
    w = (np.random.rand(env.world.size) - 0.5) * 0.01

    threshold = 0.0001
    discount_factor = 1.0
    lr = 0.001

    print(w)
    # print(w.shape)
    # first_state = env.reset()
    # ohe_vec = create_one_hot_state_vector(env, first_state)
    # output = w.dot(ohe_vec) # understand why no transpose

    # print(output)
    # print(output.shape)

    policy = np.ones([env.world.size, len(env.action_state_to_next_state)]) / len(env.action_state_to_next_state)

    policy = utils.greedy_policy_from_value_function(policy, env, value_function, discount_factor=1.0)
    # print(policy)

    # todo get {S1, G1}, {S2, G2}, ..., {St, Gt} from each single episode

    all_episode_losses = []

    # for episode in range(50):
    for episode in range(50):
        episode_state_hist, episode_reward_hist, done = run_episode(policy, env)

        # print(episode_reward_hist)
        # print(np.cumsum(episode_reward_hist))
        # print(episode_state_hist)

        loss_for_episode = 0.0

        for idx, state in enumerate(episode_state_hist):
            ohe_state_vec = create_one_hot_state_vector(env, state)
            # predicted_value = w.dot(ohe_state_vec)
            predicted_value = ohe_state_vec.dot(w)

            return_from_state = sum([(discount_factor ** i) * r for i, r in enumerate(episode_reward_hist[idx:])
                                                       if (discount_factor ** i) > threshold])

            loss = math.pow(abs(return_from_state - predicted_value), 2)
            # delta_w = lr * abs(return_from_state - predicted_value) * ohe_state_vec # todo is it absolute?
            delta_w = lr * (return_from_state - predicted_value) * ohe_state_vec
            delta_w = np.clip(delta_w, -1, 1)

            # w = w - delta_w
            w = w + delta_w
            new_predicted_value = w.dot(ohe_state_vec)
            new_loss = (return_from_state - new_predicted_value) ** 2.0

            # print('\nidx:', idx)
            # print('state:', state)
            # print('ohe:', ohe_state_vec)
            # print('predicted_value:', predicted_value)
            # print('return_from_state:', return_from_state)
            # print('delta_w:', delta_w)
            # print('w:', w)
            # print('New predicted value:', new_predicted_value)
            # print('loss: {}, new_loss: {}'.format(loss, new_loss))

            loss_for_episode += loss

        all_episode_losses.append(loss_for_episode)
        # policy = utils.greedy_policy_from_value_function(policy, env, w, discount_factor=1.0)

    policy = utils.greedy_policy_from_value_function(policy, env, w, discount_factor=1.0)
    print(all_episode_losses)
    plt.plot(all_episode_losses)
    plt.show()
    return policy


###########################

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
    states_hist.append(observation)
    for step in range(max_steps_per_episode):
        action = np.random.choice(policy[observation].size, p=policy[observation])
        observation, reward, done, info = env.step(action)
        states_hist.append(observation)
        rewards_hist.append(reward)
        if done:
            break
    return states_hist, rewards_hist, done


def monte_carlo_evaluation(policy, env, every_visit=False, incremental_mean=True, stationary_env=True,
                           discount_factor=0.99, threshold=0.0001, alpha=0.001, num_episodes=100):
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
    for episode in range(num_episodes):
        episode_state_hist, episode_reward_hist, done = run_episode(policy, env)
        print('Episode: {}, terminal found: {}'.format(episode, done))

        visits_from_last_episode = np.zeros(env.world.size)
        returns_from_last_episode = np.zeros(env.world.size)
        # Store visit counts and returns from each state from last episode depending on whether first or every visit MC
        for idx, state in enumerate(episode_state_hist):
            if visits_from_last_episode[state] == 0:
                pass  # calculate first-first below
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
                                     if (discount_factor ** i) > threshold])
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
                    if total_visit_counter[state] > 0.0:  # don't divide by 0
                        value_function[state] += (1 / total_visit_counter[state]) * (
                                                    returns_from_last_episode[state] - value_function[state])
                else:
                    """
                    In non-stationary problems, it can be useful to track a running mean, i.e. forget old episodes.
                    V(St) ← V(St) + α(Gt − V(St))
                    """
                    value_function[state] += alpha * (returns_from_last_episode[state] - value_function[state])

    if not incremental_mean:
        # Value is estimated by mean return V(s) = S(s) / N(s) if not incremental mean
        for state in range(total_visit_counter.size):
            if total_visit_counter[state] > 0.0:  # don't divide by 0
                value_function[state] = total_return[state] / total_visit_counter[state]

    return value_function

if __name__ == '__main__':
    from core.envs.gridworld_env import GridWorldEnv
    env = GridWorldEnv()
    # env = GridWorldEnv((10, 10))
    env = GridWorldEnv((10, 10), random_maze=True)

    policy = test_linear_approx(env)

    curr_state = env.reset()
    env.render_policy_arrows(policy)

    print(policy)

    for t in range(500):
        env.render(mode='graphic')

        action = np.argmax(policy[curr_state])
        # print('go ' + env.action_descriptors[action])
        curr_state, reward, done, info = env.step(action)

        if done:
            print('DONE in {} steps'.format(t + 1))
            env.render(mode='graphic')
            time.sleep(8)
            break
