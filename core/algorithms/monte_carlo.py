import sys

import numpy as np
from six import StringIO

from core.envs.gridworld_env import GridWorldEnv


def reshape_as_gridworld(input_matrix, world_shape):
    """
    Helper function to reshape a gridworld state matrix into a visual representation of the gridworld with origin on the
    low left corner and x,y corresponding to cartesian coordinates.
    """
    return np.reshape(input_matrix, (world_shape[0], world_shape[1]))[:, ::-1].T

def greedy_policy_from_value_function(env, value_function, discount_factor=1.0):
    """
    Returns a greedy policy based on the input value function.

    If no value function was provided the defaults from a single step starting with a value function of zeros
    will be used.
    """
    policy = np.zeros([env.world.size, len(env.action_state_to_next_state)])
    q_function = np.zeros((env.world.size, env.action_space.n))
    for state in range(env.world.size):
        for action in range(env.action_space.n):
            next_state, reward, done = env.look_step_ahead(state, action)
            q_function[state][action] += reward + discount_factor * value_function[next_state]
        max_value_actions = np.where(np.around(q_function[state], 8) == np.around(np.amax(q_function[state]), 8))[0]

        policy[state] = np.fromiter((1 / len(max_value_actions) if action in max_value_actions and
                                    not env.is_terminal(state) else 0
                                    for action in np.nditer(np.arange(env.action_space.n))), dtype=np.float)
    return policy

def get_policy_map(policy, world_shape, mode='human'):
    """
    Generates a visualization grid from the policy to be able to print which action is most likely from every state
    """
    unicode_arrows = np.array([u'\u2191', u'\u2192', u'\u2193', u'\u2190' # up, right, down, left
                      u'\u2194', u'\u2195'], dtype='<U1')  # left-right, up-down
    policy_arrows_map = np.empty(policy.shape[0], dtype='<U4')
    for state in np.nditer(np.arange(policy.shape[0])):
        # find index of actions where the probability is > 0
        optimal_actions = np.where(np.around(policy[state], 8) > np.around(np.float64(0), 8))[0]
        # match actions to unicode values of the arrows to be displayed
        for action in optimal_actions:
            policy_arrows_map[state] = np.core.defchararray.add(policy_arrows_map[state], unicode_arrows[action])
    policy_probabilities = np.fromiter((policy[state] for state in np.nditer(np.arange(policy.shape[0]))),
                                       dtype='float64, float64, float64, float64')
    outfile = StringIO() if mode == 'ansi' else sys.stdout
    for row in reshape_as_gridworld(policy_arrows_map, world_shape):
        for state in row:
            outfile.write((state + u'  '))
        outfile.write('\n')
    outfile.write('\n')

    return policy_arrows_map, reshape_as_gridworld(policy_probabilities, world_shape)

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
        action = np.random.choice(policy[observation].size, p=policy[observation]) # todo: change to random number in env action space
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

    total_visit_counter = {} # could do "total_visit_counter = np.zeros(env.world.size)" but that isn't model-free
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

            # Calculation depends heavily if whether it is first or every visit MC.
            # Update visit counter for last episode
            visits_from_last_episode[state] += 1
            # Update return for last episode for a specific state
            return_from_state = sum([(discount_factor ** i) * r for i, r in enumerate(episode_reward_hist[idx:])
                                                  if (discount_factor ** i) < threshold]) # todo mention bug in commit
            returns_from_last_episode[state] += return_from_state # Add if statement if first visit
            # todo check that return is just summed up together in every visit MC and then only at end of episode do you do error term update (not for each update)

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
                """
                Increment total return S(s) ← S(s) + Gt
                """
                total_return[state] += returns_from_last_episode[state]
            else:
                # Update value function only if incremental mean. Otherwise do at end
                if stationary_env:
                    """
                    V(St) ← V(St) + 1/ N(St) * (Gt − V(St))
                    """
                    value_function[state] += (1 / total_visit_counter[state]) * (returns_from_last_episode[state] - value_function[state])
                else:
                    """
                    In non-stationary problems, it can be useful to track a running mean, i.e. forget old episodes.
                    V(St) ← V(St) + α(Gt − V(St))
                    """
                    # todo stationary with non-incremental mean? And shouldn't store visit count? Not necessary
                    value_function[state] += alpha * (returns_from_last_episode[state] - value_function[state])


    if not incremental_mean:
        """
        Value is estimated by mean return V(s) = S(s) / N(s) if not incremental mean
        """
        for state in total_visit_counter:
            value_function[state] = total_return[state] / total_visit_counter[state]

    return value_function


if __name__ == '__main__':
    env = GridWorldEnv()
    # todo non-model free. 4 actions is all we should know.
    policy0 = np.ones([env.world.size, env.action_space.n]) / env.action_space.n
    st_history, rw_history = run_episode(policy0, env)
    print('States history: ' + str(st_history))
    print('Rewards history: ' + str(rw_history))
    value0 = monte_carlo_evaluation(policy0, env, every_visit=True, stationary_env=False)
    print(value0)
    for state, value in value0.items():
        print(state, value)

    # Create greedy policy from value function and run it on environment
    world_shape = (4, 4)
    policy1 = greedy_policy_from_value_function(env, value0)
    print(policy1)

    policy_map1 = get_policy_map(policy1, world_shape)
    print('Policy: (up, right, down, left)\n', get_policy_map(policy1, world_shape))
    np.set_printoptions(linewidth=75, precision=8)

    print('Starting greedy policy run')
    curr_state = env.reset()

    for t in range(100):
        env.render()

        action = np.argmax(policy1[curr_state])
        print('go ' + env.action_descriptors[action])
        curr_state, reward, done, info = env.step(action)

        if done:
            print('DONE in {} steps'.format(t + 1))
            break
