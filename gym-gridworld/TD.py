import numpy as np
from gym_gridworld.envs.gridworld_env import GridWorldEnv
import warnings


def reshape_as_gridworld(input_matrix):
    """
    Helper function to reshape a gridworld state matrix into a visual representation of the gridworld with origin on the
    low left corner and x,y corresponding to cartesian coordinates.
    """
    return np.reshape(input_matrix, (world_shape[0], world_shape[1]))[:, ::-1].T

def get_policy_map(policy):
    """
    Generates a visualization grid from the policy to be able to print which action is most likely from every state
    """
    unicode_arrows = [u'\u2191', u'\u2192', u'\u2193', u'\u2190' # up, right, down, left
                      u'\u2194', u'\u2195']  # left-right, up-down
    policy_arrows_map = np.full(policy.shape[0], ' ')
    for state in np.nditer(np.arange(policy.shape[0])):
        # find index of actions where the probability is > 0
        optimal_actions = np.where(np.around(policy[state], 8) > np.around(np.float64(0), 8))[0]
        # match actions to unicode values of the arrows to be displayed
        for action in optimal_actions:
            policy_arrows_map[state] = ' '.join((policy_arrows_map[state], unicode_arrows[action]))
    # TODO: Problem on policy_arrows_map definition and update
    policy_probabilities = np.fromiter((policy[state] for state in np.nditer(np.arange(policy.shape[0]))),
                                       dtype='float64, float64, float64, float64')

    return reshape_as_gridworld(policy_arrows_map), reshape_as_gridworld(policy_probabilities)

def greedy_policy_from_value_function(env, value_function, discount_factor=1.0):
    """
    Returns a greedy policy based on the input value function.

    If no value function was provided the defaults from a single step starting with a value function of zeros
    will be used.
    """
    #policy =
    #policy = np.ones([env.world.size, len(env.action_state_to_next_state)]) / len(env.action_state_to_next_state)
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

# V(St) ← V(St) + alpha * (Gt − V(St))
# V(St) ← V(St) + alpha * (Rt+1 + lambda * V(St+1) − V(St))

alpha = 0.1
discount_factor = 0.99

if __name__ == '__main__':
    world_shape = (4, 4)
    env = GridWorldEnv(world_shape=world_shape)

    value_function = np.zeros(env.world.size)

    print(value_function)

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
                #print("Episode finished after {} timesteps".format(t + 1))
                break

        print(i_episode, '\n', value_function)

    # Create greedy policy from value function
    policy1 = greedy_policy_from_value_function(env, value_function)
    print(policy1)

    policy_map1 = get_policy_map(policy1)
    print('Policy: (0=up, 1=right, 2=down, 3=left)\n', policy_map1)
    np.set_printoptions(linewidth=75 * 2, precision=4)
    print('Policy: (up, right, down, left)\n', get_policy_map(policy1))
    np.set_printoptions(linewidth=75, precision=8)

    print('Starting greedy policy run')
    curr_state = env.reset()
    prev_state = curr_state
    for t in range(100):
        env.render()

        action = np.argmax(policy1[curr_state])
        print(policy1[curr_state])
        print(action)
        #print('go ' + env.action_descriptors[action])
        curr_state, reward, done, info = env.step(action)

        if done:
            print('DONE in {} steps'.format(t + 1))
            break
