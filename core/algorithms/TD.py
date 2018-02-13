import warnings


import numpy as np


from core.envs.gridworld_env import GridWorldEnv


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


def n_step_return(policy, env, n_steps, curr_state=None):
    """
    Moves the agent n_steps and returns the sum of the rewards experienced on those steps.

    Assumes a stochastic policy and takes an action sample taken from a distribution with the probabilities given
    by the policy.
    """
    reward_experienced = 0  # Gt according to the equations
    curr_state = env.current_state if curr_state is None else curr_state

    for step in range(n_steps):
        action = np.random.choice(policy[curr_state].size, p=policy[curr_state])
        curr_state, step_reward, done = env.look_step_ahead(curr_state, action)
        if done:
            warning_message = 'Terminal state {} reached after {} steps'.format(curr_state, step + 1)
            warnings.warn(warning_message, UserWarning)
            break
        reward_experienced += step_reward
    return reward_experienced, curr_state


def td_single_n_step_evaluation(policy, env, n_steps, curr_state=None, value_function=None, gamma=0.9, alpha=0.01):
    """
    TD n-step algorithm for policy evaluation in a single n-step
    """
    value_function = np.zeros(env.world.size) if value_function is None else value_function
    curr_state = env.current_state if curr_state is None else curr_state
    action = np.random.choice(policy[curr_state].size, p=policy[curr_state])

    return_value, last_state = n_step_return(policy, env, n_steps, curr_state)
    next_state, *_ = env.look_step_ahead(last_state, action)
    td_target = return_value + gamma * value_function[next_state]
    td_error = td_target - value_function[curr_state]

    value_function[curr_state] += alpha * td_error
    return value_function


def td_episodic_n_step_evaluation(policy, env, n_steps, curr_state=None, value_function=None, gamma=0.9, alpha=0.01,
                                  n_episodes=100):
    """
    TD n-step algorithm for policy evaluation in n_episodes number of episodes
    """
    value_function = np.zeros(env.world.size) if value_function is None else value_function
    curr_state = env.current_state if curr_state is None else curr_state

    for episode in range(n_episodes):
        # TODO: recheck n_step execution
        value_function = td_single_n_step_evaluation(policy, env, n_steps, curr_state, value_function=value_function,
                                                     gamma=gamma, alpha=alpha)
    return value_function


def td_lambda_episodic_evaluation(policy, env, curr_state=None, value_function=None, gamma=0.9, alpha=0.01,
                                  max_steps_per_episode=1000, lambda_val=0.9, backward_view=False, execution='offline',
                                  n_episodes=100):
    """
    TD lambda

    execution: 'offline', 'online', 'exact_online'
    """
    valid_execution_modes = ['offline', 'online', 'exact_online']
    if execution not in valid_execution_modes:
        raise ValueError('The execution mode must be one of the elements form the '
                         'following list {}'.format(valid_execution_modes))
    value_function = np.zeros(env.world.size) if value_function is None else value_function
    updated_value_function = value_function
    eligibility_traces = np.zeros(env.world.size) if backward_view else np.ones(env.world.size)
    curr_state = env.current_state if curr_state is None else curr_state

    for episode in range(n_episodes):
        value_function = updated_value_function
        if execution == 'offline':
            # TODO: recheck forward view offline execution
            # Compute full episode before updating
            states_hist, rewards_hist, done = run_episode(policy, env, max_steps_per_episode=max_steps_per_episode)
            step_returns = np.cumsum(rewards_hist)
            lambda_return = np.fromiter(((1 - lambda_val) * lambda_val ** (step_n + 1) * cum_return
                                         for step_n, cum_return in enumerate(step_returns)), dtype='float64')

            for step_n, curr_state in enumerate(states_hist[:-1]):
                td_target = lambda_return[step_n] + gamma * value_function[states_hist[step_n + 1]]
                td_error = td_target - value_function[curr_state]
                if backward_view:
                    eligibility_traces *= gamma * lambda_val
                    eligibility_traces[curr_state] += 1

                updated_value_function[curr_state] += alpha * td_error * eligibility_traces[curr_state]

        else:
            # Evaluate lambda return step by step
            cum_return = 0
            for step_n in range(max_steps_per_episode):
                action = np.random.choice(policy[curr_state].size, p=policy[curr_state])
                curr_state, reward, done = env.look_step_ahead(curr_state, action)
                next_state, *_ = env.look_step_ahead(env.current_state, action)
                if done:
                    break

                cum_return += reward
                lambda_return = (1 - lambda_val) * lambda_val ** (step_n + 1) * cum_return
                td_target = lambda_return + gamma * value_function[next_state]
                td_error = td_target - value_function[curr_state]
                if backward_view:
                    eligibility_traces *= gamma * lambda_val
                    eligibility_traces[curr_state] += 1

                updated_value_function[curr_state] += alpha * td_error * eligibility_traces[curr_state]
                if execution is not 'exact_online':
                    value_function[curr_state] = updated_value_function[curr_state]

    return value_function


if __name__ == '__main__':
    # TD Evaluation
    world_shape = (4, 4)
    env = GridWorldEnv(world_shape=world_shape)
    value_func = np.zeros(env.world.size)
    policy0 = np.ones([env.world.size, len(env.action_state_to_next_state)]) / len(env.action_state_to_next_state)

    print('Initial value function:', value_func)
    np.set_printoptions(linewidth=75 * 2, precision=4)

    # TD 5-step return for 2 episodes
    value_func_td5step = td_episodic_n_step_evaluation(policy=policy0, env=env, n_steps=5, n_episodes=2)
    print("Value function for TD 5 step return run on 2 episodes:\n", value_func_td5step)
    # TD lambda forward view offline
    value_func_td_lambda_fwd_off = td_lambda_episodic_evaluation(policy=policy0, env=env, value_function=value_func,
                                                                 n_episodes=2)
    print("Value function for TD lambda forward view offline in 2 episodes:\n", value_func_td_lambda_fwd_off)
    # TD lambda backward view offline
    value_func_td_lambda_bwd_off = td_lambda_episodic_evaluation(policy=policy0, env=env, value_function=value_func,
                                                                 backward_view=True, n_episodes=2)
    print("Value function for TD lambda backward view offline in 2 episodes:\n", value_func_td_lambda_bwd_off)
    # TD lambda forward view online
    value_func_td_lambda_fwd_on = td_lambda_episodic_evaluation(policy=policy0, env=env, value_function=value_func,
                                                                execution='online', n_episodes=2)
    print("Value function for TD lambda forward view online in 2 episodes:\n", value_func_td_lambda_fwd_on)
    # TD lambda backward view online
    value_func_td_lambda_bwd_on = td_lambda_episodic_evaluation(policy=policy0, env=env, value_function=value_func,
                                                                backward_view=True, execution='online', n_episodes=2)
    print("Value function for TD lambda backward view online in 2 episodes:\n", value_func_td_lambda_bwd_on)
    # TD lambda backward view exact online
    value_func_td_lambda_bwd_on = td_lambda_episodic_evaluation(policy=policy0, env=env, value_function=value_func,
                                                                backward_view=True, execution='exact_online',
                                                                n_episodes=2)
    print("Value function for TD lambda backward view  exact online in 2 episodes:\n", value_func_td_lambda_bwd_on)
    # TD lambda invalid execution mode
    value_func_td_lambda_bwd_on = td_lambda_episodic_evaluation(policy=policy0, env=env, value_function=value_func,
                                                                backward_view=True, execution='ham', n_episodes=2)
