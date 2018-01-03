import sys

import numpy as np

from core.envs.gridworld_env import GridWorldEnv
from core.algorithms import utils

alpha = 0.1
discount_factor = 0.99

if __name__ == '__main__':
    # TD Learning
    # V(St) ← V(St) + alpha * (Rt+1 + lambda * V(St+1) − V(St))

    world_shape = (4, 4)
    env = GridWorldEnv(world_shape=world_shape)

    value_func = np.zeros(env.world.size)
    print('Initial value function:', value_func)
    n = -1 # 5 # set to -1 for Monte Carlo

    # act randomly and evaluate policy
    for i_episode in range(50):
        curr_state = env.reset()
        prev_state = curr_state

        all_rewards = []
        for t in range(100):
            action = env.action_space.sample()
            curr_state, reward, done, info = env.step(action)
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

            # n_step_return = reward + discount_factor * value_func[curr_state]

            if len(all_rewards) > n:
                # print(all_rewards)
                if n == -1:
                    if not done:
                        continue
                    else:
                        last_n_rewards = all_rewards # todo bug. Only assigns credit to first/one state. TD(1) must assign values to every state on path???
                else:
                    last_n_rewards = all_rewards[-n:]
                print('N last rewards:', last_n_rewards)

                discounted_immediate_rewards = sum([(discount_factor ** i) * r for i, r in enumerate(last_n_rewards)])
                discounted_future_value = (discount_factor ** n) * value_func[curr_state]
                n_step_return = discounted_immediate_rewards + discounted_future_value
                print('discounted_immediate_rewards:', discounted_immediate_rewards)
                print('n_step_return:', n_step_return)
                TD_error = n_step_return - value_func[prev_state]
                value_func[prev_state] = value_func[prev_state] + alpha * TD_error

            # if (discount_factor ** i) < threshold]) todo or not

            prev_state = curr_state
            if done:
                print("Episode finished after {} timesteps".format(t + 1))
                break

        #print(i_episode, '\n', value_function)

    print('Final value function:', value_func)
    # sys.exit()
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
    for t in range(30):
        env.render()

        action = np.argmax(policy1[curr_state])
        print('go ' + env.action_descriptors[action])
        curr_state, reward, done, info = env.step(action)

        if done:
            print('DONE in {} steps'.format(t + 1))
            break
