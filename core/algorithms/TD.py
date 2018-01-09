import sys

import numpy as np

from core.envs.gridworld_env import GridWorldEnv
from core.algorithms import utils

alpha = 0.1
discount_factor = 0.99
lambda_factor = 0.9
n = 5 # set to -1 for Monte Carlo

def calculate_n_step_return(n_steps, from_idx, all_states, all_rewards):
    """
    :param n_steps:
    :param from_idx:
    :param all_states:
    :param all_rewards:
    :return: n_step_return
    """
    # todo pass in value func.
    # todo pass in values better. stop using globals

    # if len(all_rewards) > n:
        # print(all_rewards)
        # if n == -1:
        #     if not done:
        #         continue
        #     else:
        #         last_n_rewards = all_rewards # todo bug. Only assigns credit to first/one state. TD(1) must assign values to every state on path???
        # else:
        #     last_n_rewards = all_rewards[-n:]

    # todo instead of complex from_idx, n indexing, I could just pass the right arrays as parameters
    # todo assert that that from_idx + final n == number of steps in episode

    # last_n_rewards = all_rewards[-n:]
    # if


    first_n_rewards = all_rewards[from_idx:from_idx + n_steps]

    print('\nFrom step index: {}'.format(from_idx))
    print('First N = {} rewards: {}'.format(n_steps, first_n_rewards))
    print('Value of St+{}: {}, value of St: {}'.format(n_steps, value_func[all_states[from_idx + n_steps]], value_func[all_states[from_idx]]))


    # todo check V(St + 1) This is the future look.
    # if from_idx + 2 > len(all_states): # stupid jump out for now. it was from_idx + 1 but was still breaking...
    #     return 0.0
    # try:
    #     discounted_future_value = (discount_factor ** n) * value_func[all_states[from_idx + 1]] # todo value_func[curr_state] == this was a bug. think more. prev vs curr?
    # except:
    #     raise EnvironmentError('{} {}'.format(from_idx, len(all_states)))

    discounted_immediate_rewards = sum([(discount_factor ** i) * r for i, r in enumerate(first_n_rewards)])
    discounted_future_value = (discount_factor ** n_steps) * value_func[all_states[from_idx + n_steps]]
    n_step_return = discounted_immediate_rewards + discounted_future_value
    print('discounted_immediate_rewards:', discounted_immediate_rewards)
    print('discounted_future_value:', discounted_future_value)
    print('{}_step_return: {}'.format(n_steps, n_step_return))
    return n_step_return
    #TD_error = n_step_return - value_func[all_states[from_idx]] # todo check. This is current state
    # TD_error = n_step_return - value_func[prev_state]
    # value_func[prev_state] = value_func[prev_state] + alpha * TD_error


    #return all_states[from_idx] + alpha * TD_error

if __name__ == '__main__':
    # TD Learning
    # V(St) ← V(St) + alpha * (Rt+1 + lambda * V(St+1) − V(St))

    world_shape = (4, 4)
    env = GridWorldEnv(world_shape=world_shape)

    value_func = np.zeros(env.world.size)
    print('Initial value function:', value_func)


    # Forward-view TD(lambda)
    # 1. Store each state and reward in episode
    # 2. At end of episode (offline) compute, for each step, lambda-return
    #    2.1. 1-step return, 2-step return, ..., all-step return (MC)

    # Make agent act randomly and evaluate policy
    for i_episode in range(14):
        curr_state = env.reset()
        prev_state = curr_state

        all_states = [curr_state]
        all_rewards = []

        print('\n\n\nSTARTING NEXT EPISODE!!!!!!!!!!!!!!!!!!!!!!!!! \n\n\n')
        print(value_func)

        for t in range(100):
            action = env.action_space.sample()
            curr_state, reward, done, info = env.step(action)
            all_states.append(curr_state)
            all_rewards.append(reward)


            # G(n)t = Rt+1 + γRt+2 + ... + γ ^ n−1 * Rt+n + γ ^ n V(St+n)
            # V(St) ← V(St) + α * (G(n)t − V(St))

            # n = 1 = TD(0)
            # G(1)t = Rt+1 + γ * V(St+1)
            # V(St) ← V(St) + α * (Rt+1 + γ * V(St+1) − V(St))

            # n = 2
            # G(2)t = Rt+1 + γ * Rt+2 + γ ^ 2 * V(St+2)

            # so we have to wait 2 steps at the start before calculating anything.
            # And then every step, we do the above.

            # todo
            # n_step_return = reward + discount_factor * value_func[curr_state]

            # if len(all_rewards) > n:
            #     # print(all_rewards)
            #     # if n == -1:
            #     #     if not done:
            #     #         continue
            #     #     else:
            #     #         last_n_rewards = all_rewards # todo bug. Only assigns credit to first/one state. TD(1) must assign values to every state on path???
            #     # else:
            #     #     last_n_rewards = all_rewards[-n:]
            #     last_n_rewards = all_rewards[-n:]
            #     print('N last rewards:', last_n_rewards)
            #
            #     discounted_immediate_rewards = sum([(discount_factor ** i) * r for i, r in enumerate(last_n_rewards)])
            #     discounted_future_value = (discount_factor ** n) * value_func[curr_state]
            #     n_step_return = discounted_immediate_rewards + discounted_future_value
            #     print('discounted_immediate_rewards:', discounted_immediate_rewards)
            #     print('n_step_return:', n_step_return)
            #     TD_error = n_step_return - value_func[prev_state]
            #     value_func[prev_state] = value_func[prev_state] + alpha * TD_error

            # if (discount_factor ** i) < threshold]) todo or not

            prev_state = curr_state
            if done:
                # 2. At end of episode (offline) compute, for each step, calculate lambda-return
                #    2.1. 1-step return, 2-step return, ..., all-step (inf) return (MC)
                # todo test case for averaging 1-step and 2-step return

                print('episode over, calculating lambda_return')
                print('All states: {}'.format(all_states))
                print('All rewards: {}'.format(all_rewards))

                new_value_func = np.zeros(value_func.shape)

                for start_idx, state in enumerate(all_states):
                    lambda_return = 0
                    for n_steps, s in enumerate(all_states[start_idx:]):
                        # todo no such thing as 0-step
                        if n_steps == 0:
                            continue
                        # 2nd parameter: n is 1-indexed e.g. 1-step, 2-step etc.
                        # lambda_return +=  (lambda_factor ** step_idx) * calculate_n_step_return(step_idx + 1, start_idx, all_states, all_rewards)
                        lambda_return += (lambda_factor ** (n_steps - 1)) * calculate_n_step_return(n_steps, start_idx, all_states, all_rewards)

                    # todo shouldn't update here.... should make new value function. confirm.

                    lambda_return = lambda_return * (1 - lambda_factor) # outside sum
                    TD_error = lambda_return - value_func[state]
                    new_value_func[state] = value_func[state] + alpha * TD_error # update towards error
                    #break

                print("Episode finished after {} timesteps".format(t + 1))
                value_func = new_value_func
                break # next episode

        #print(i_episode, '\n', value_function)

    print('Final value function:', value_func)
    #sys.exit()
    # Now test algorithm
    # Create greedy policy from value function
    # todo change so we don't need or always fill policy till end. Or... always have value as 0 for unseen states.
    policy0 = np.ones([env.world.size, len(env.action_state_to_next_state)]) / len(env.action_state_to_next_state)
    policy1 = utils.greedy_policy_from_value_function(policy0, env, value_func)
    # print(policy1)

    policy_map1 = utils.get_policy_map(policy1, world_shape)
    # print('Policy: (up, right, down, left)\n', policy_map1)
    np.set_printoptions(linewidth=75, precision=8)

    print('Starting greedy policy run')
    curr_state = env.reset()
    prev_state = curr_state
    for t in range(10):
        env.render()

        # action = np.argmax(policy1[curr_state])
        action = np.random.choice(policy1[curr_state].size, p=policy1[curr_state]) # take uniform choice if equal
        print('go ' + env.action_descriptors[action])
        curr_state, reward, done, info = env.step(action)

        if done:
            print('DONE in {} steps'.format(t + 1))
            break
