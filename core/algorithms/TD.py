import warnings

import numpy as np

from core.envs.gridworld_env import GridWorldEnv
from core.algorithms.utils import run_episode


def td_n_step_evaluation(policy, env, num_steps, value_function=None, gamma=0.9, alpha=0.01, num_episodes=100):
    """
    TD n-step algorithm for policy evaluation in num_episodes number of episodes

    The initial state for every episode is set to the value provided in the creation of the environment.
    """
    value_function = np.zeros(env.world.size) if value_function is None else value_function

    for _ in range(num_episodes):
        curr_state = env.curr_state
        states_history = [curr_state]
        rewards_history = []
        state_update_idx = 0
        done = False
        while not done:
            # move n-steps. Store states and rewards
            while len(states_history) < state_update_idx + num_steps:
                action = np.random.choice(policy[curr_state].size, p=policy[curr_state])
                curr_state, step_reward, done = env.look_step_ahead(curr_state, action)
                states_history.append(curr_state)
                rewards_history.append(step_reward)

            # update value function for state_update_idx
            return_value = 0
            for step in range(num_steps):
                return_value += rewards_history[state_update_idx + step] * gamma ** step
            td_target = return_value + (gamma ** num_steps) * value_function[states_history[state_update_idx + num_steps]]
            td_error = td_target - value_function[states_history[state_update_idx]]
            value_function[states_history[state_update_idx]] += alpha * td_error
            state_update_idx += 1
        env.reset()
    return value_function


def td_lambda_update(value_function, states_hist, rewards_hist, lambda_val, gamma, alpha, backward_view):
    """
    Helper function for TD lambda
    """
    updated_value_function = value_function.copy()
    eligibility_traces = np.zeros(env.world.size) if backward_view else np.ones(env.world.size)
    cum_return = np.cumsum(rewards_hist)

    for step_num, curr_state in enumerate(states_hist[:-1]):
        # calculate lambda return
        lambda_return = cum_return[-1] * lambda_val ** (len(rewards_hist) - 1)
        for step in range(len(states_hist[:-1]) - step_num - 1):
            lambda_return += cum_return[step_num] * (1 - lambda_val) * (lambda_val ** step)
        # calculate td values
        td_target = lambda_return + gamma * value_function[curr_state]
        td_error = td_target - value_function[curr_state]
        if backward_view:
            eligibility_traces *= gamma * lambda_val
            eligibility_traces[curr_state] += 1
        # update value function
        updated_value_function[curr_state] += alpha * td_error * eligibility_traces[curr_state]
    return updated_value_function


def td_lambda_evaluation(policy, env, curr_state=None, value_function=None, gamma=0.9, alpha=0.01,
                         max_steps_per_episode=1000, lambda_val=0.9, backward_view=False, execution='offline',
                         num_episodes=100):
    """
    TD lambda

    execution: 'offline', 'online', 'exact_online'
    """
    valid_execution_modes = ['offline', 'online', 'exact_online']
    if execution not in valid_execution_modes:
        raise ValueError('The execution mode must be one of the elements form the '
                         'following list {}'.format(valid_execution_modes))
    value_function = np.zeros(env.world.size) if value_function is None else value_function
    updated_value_function = value_function.copy()
    curr_state = env.current_state if curr_state is None else curr_state

    for episode in range(num_episodes):
        value_function = updated_value_function.copy()
        if execution == 'offline':
            # Compute full episode before updating
            states_hist, rewards_hist, done = run_episode(policy, env, max_steps_per_episode=max_steps_per_episode)
            updated_value_function = td_lambda_update(value_function, states_hist, rewards_hist, lambda_val, gamma,
                                                      alpha, backward_view)
        else:
            # Evaluate lambda return step by step
            states_hist = [curr_state]
            rewards_hist = []
            for episode_step in range(max_steps_per_episode):
                # move 1 step. Store states and rewards
                action = np.random.choice(policy[curr_state].size, p=policy[curr_state])
                curr_state, step_reward, done = env.look_step_ahead(curr_state, action)
                states_hist.append(curr_state)
                rewards_hist.append(step_reward)
                updated_value_function = td_lambda_update(value_function, states_hist, rewards_hist, lambda_val, gamma,
                                                          alpha, backward_view)
                if execution is not 'exact_online':
                    value_function[curr_state] = updated_value_function[curr_state].copy()
                if done:
                    break
        env.reset()
    return value_function


if __name__ == '__main__':
    # TD Evaluation
    world_shape = (4, 4)
    env = GridWorldEnv(world_shape=world_shape)
    value_func = np.zeros(env.world.size)
    policy0 = np.ones([env.world.size, len(env.action_state_to_next_state)]) / len(env.action_state_to_next_state)
    episodes = 1000
    print('Initial value function:', value_func)
    np.set_printoptions(linewidth=75 * 2, precision=4)

    # TD 3-step return for 2 episodes
    value_func_td5step = td_n_step_evaluation(policy=policy0, env=env, num_steps=3, num_episodes=episodes)
    print("Value function for TD 3 step return run on 1 episode:\n", value_func_td5step)
    # TD lambda forward view offline
    env.reset()
    value_func_td_lambda_fwd_off = td_lambda_evaluation(policy=policy0, env=env, num_episodes=episodes)
    print("Value function for TD lambda forward view offline in {} episodes:\n".format(episodes),
          value_func_td_lambda_fwd_off)
    # TD lambda backward view offline
    env.reset()
    value_func_td_lambda_bwd_off = td_lambda_evaluation(policy=policy0, env=env, backward_view=True,
                                                        num_episodes=episodes)
    print("Value function for TD lambda backward view offline in {} episodes:\n".format(episodes),
          value_func_td_lambda_bwd_off)
    # TD lambda forward view online
    env.reset()
    value_func_td_lambda_fwd_on = td_lambda_evaluation(policy=policy0, env=env, execution='online',
                                                       num_episodes=episodes)
    print("Value function for TD lambda forward view online in {} episodes:\n".format(episodes),
          value_func_td_lambda_fwd_on)
    # TD lambda backward view online
    env.reset()
    value_func_td_lambda_bwd_on = td_lambda_evaluation(policy=policy0, env=env, backward_view=True,
                                                       execution='online', num_episodes=episodes)
    print("Value function for TD lambda backward view online in {} episodes:\n".format(episodes),
          value_func_td_lambda_bwd_on)
    # TD lambda backward view exact online
    env.reset()
    value_func_td_lambda_bwd_on = td_lambda_evaluation(policy=policy0, env=env, backward_view=True,
                                                       execution='exact_online', num_episodes=episodes)
    print("Value function for TD lambda backward view  exact online in {} episodes:\n".format(episodes),
          value_func_td_lambda_bwd_on)
    # TD lambda invalid execution mode
    # env.reset()
    # value_func_td_lambda_bwd_on = td_lambda_evaluation(policy=policy0, env=env, backward_view=True,
    #                                                    execution='ham', num_episodes=100)
