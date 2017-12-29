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

    value_function = np.zeros(env.world.size)
    print('Initial value function:', value_function)

    # act randomly and evaluate policy
    for i_episode in range(15):
        curr_state = env.reset()
        prev_state = curr_state
        for t in range(1000):
            #env.render()
            action = env.action_space.sample()
            #print('go ' + env.action_descriptors[action])
            curr_state, reward, done, info = env.step(action)

            # V(St) ← V(St) + alpha * (Rt+1 + lambda * V(St+1) − V(St))

            value_function[prev_state] = value_function[prev_state] + \
                                         alpha * (reward +
                                         discount_factor * value_function[curr_state]
                                          - value_function[prev_state])

            prev_state = curr_state

            if done:
                print("Episode finished after {} timesteps".format(t + 1))
                break

        #print(i_episode, '\n', value_function)

    # Now test algorithm
    # Create greedy policy from value function
    # todo change so we don't need or always fill policy till end. Or... always have value as 0 for unseen states.
    policy0 = np.ones([env.world.size, len(env.action_state_to_next_state)]) / len(env.action_state_to_next_state)
    policy1 = utils.greedy_policy_from_value_function(policy0, env, value_function)
    # print(policy1)

    policy_map1 = utils.get_policy_map(policy1, world_shape)
    print('Policy: (up, right, down, left)\n', policy_map1)
    np.set_printoptions(linewidth=75, precision=8)

    print('Starting greedy policy run')
    curr_state = env.reset()
    prev_state = curr_state
    for t in range(100):
        env.render()

        action = np.argmax(policy1[curr_state])
        print('go ' + env.action_descriptors[action])
        curr_state, reward, done, info = env.step(action)

        if done:
            print('DONE in {} steps'.format(t + 1))
            break
