import time
from collections import deque
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import gym

from core.envs.griduniverse_env import GridUniverseEnv
from core.algorithms.monte_carlo import run_episode, monte_carlo_evaluation
from core.algorithms import utils
import core.algorithms.dynamic_programming as dp


def run_policy_and_value_iteration():
    """
    Majority of code is within utils.py and dynamic_programming.py for this function
    This function does 4 things:

    1. Evaluate the value function of a random policy a number of times
    2. Create a greedy policy created from from this value function
    3. Run Policy Iteration
    4. Run Value Iteration
    5. Run agent on environment on policy found from Value Iteration
    """
    print('\n' + '*' * 20 + 'Starting value and policy iteration' + '*' * 20 + '\n')

    # 1. Evaluate the value function of a random policy a number of times
    world_shape = (4, 4)
    # env = GridUniverseEnv(grid_shape=world_shape, goal_states=[3, 12]) # Sutton and Barlo/David Silver example
    # specific case with lava and path true it
    # env = GridUniverseEnv(grid_shape=world_shape, lava_states=[i for i in range(15) if i not in [0, 4, 8, 12, 13, 14, 15]])
    world_shape = (11, 11)
    env = GridUniverseEnv(grid_shape=world_shape, random_maze=True)
    policy0 = np.ones([env.world.size, len(env.action_state_to_next_state)]) / len(env.action_state_to_next_state)
    v0 = np.zeros(env.world.size)
    val_fun = v0
    for k in range(500):
        val_fun = utils.single_step_policy_evaluation(policy0, env, value_function=val_fun)
    print(utils.reshape_as_griduniverse(val_fun, world_shape))

    # 2. Create a greedy policy created from from this value function
    policy1 = utils.greedy_policy_from_value_function(policy0, env, val_fun)
    policy_map1 = utils.get_policy_map(policy1, world_shape)
    print('Policy: (0=up, 1=right, 2=down, 3=left)\n', policy_map1)
    np.set_printoptions(linewidth=75 * 2, precision=4)
    print('Policy: (up, right, down, left)\n', utils.get_policy_map(policy1, world_shape))
    np.set_printoptions(linewidth=75, precision=8)

    # 3. Run Policy Iteration
    print('Policy iteration:')
    policy0 = np.ones([env.world.size, len(env.action_state_to_next_state)]) / len(env.action_state_to_next_state)
    optimal_value, optimal_policy = dp.policy_iteration(policy0, env, v0, threshold=0.001, max_steps=1000)
    print('Value:\n', utils.reshape_as_griduniverse(optimal_value, world_shape))
    print('Policy: (0=up, 1=right, 2=down, 3=left)\n', utils.get_policy_map(optimal_policy, world_shape))
    np.set_printoptions(linewidth=75 * 2, precision=4)
    print('Policy: (up, right, down, left)\n', utils.get_policy_map(optimal_policy, world_shape))
    np.set_printoptions(linewidth=75, precision=8)

    # 4. Run Value Iteration
    print('Value iteration:')
    policy0 = np.ones([env.world.size, len(env.action_state_to_next_state)]) / len(env.action_state_to_next_state)
    optimal_value, optimal_policy = dp.value_iteration(policy0, env, v0, threshold=0.001, max_steps=100)
    print('Value:\n', utils.reshape_as_griduniverse(optimal_value, world_shape))
    print('Policy: (0=up, 1=right, 2=down, 3=left)\n', utils.get_policy_map(optimal_policy, world_shape))
    np.set_printoptions(linewidth=75 * 2, precision=4)
    print('Policy: (up, right, down, left)\n', utils.get_policy_map(optimal_policy, world_shape))
    np.set_printoptions(linewidth=75, precision=8)

    # 5. Run agent on environment on policy found from Value Iteration
    print('Starting to run agent on environment with optimal policy')
    curr_state = env.reset()
    env.render_policy_arrows(optimal_policy)

    # Dynamic programming doesn't necessarily have the concept of an agent.
    # But you can create an agent to run on the environment using the found policy
    for t in range(100):
        env.render(mode='graphic')

        action = np.argmax(optimal_policy[curr_state])
        print('go ' + env.action_descriptors[action])
        curr_state, reward, done, info = env.step(action)

        if done:
            print('Terminal state reached in {} steps'.format(t + 1))
            env.render(mode='graphic') # must render here to see agent in final state
            time.sleep(6)
            env.render(close=True)
            break


def run_monte_carlo_evaluation():
    """
    Run Monte Carlo evaluation on random policy and then act greedily with respect to the value function
    after the evaluation is complete
    """

    print('\n' + '*' * 20 + 'Starting Monte Carlo evaluation and greedy policy' + '*' * 20 + '\n')
    world_shape = (8, 8)
    # env = GridUniverseEnv(grid_shape=world_shape) # Default GridUniverse
    env = GridUniverseEnv(world_shape, random_maze=True)
    policy0 = np.ones([env.world.size, env.action_space.n]) / env.action_space.n

    print('Running an episode with a random agent (with initial policy)')
    st_history, rw_history, done = run_episode(policy0, env)

    print('Starting Monte-Carlo evaluation of random policy')
    value0 = monte_carlo_evaluation(policy0, env, every_visit=True, num_episodes=30)
    print(value0)

    # Create greedy policy from value function and run it on environment
    policy1 = utils.greedy_policy_from_value_function(policy0, env, value0)
    print(policy1)

    print('Policy: (up, right, down, left)\n', utils.get_policy_map(policy1, world_shape))
    np.set_printoptions(linewidth=75, precision=8)

    print('Starting greedy policy episode')
    curr_state = env.reset()
    env.render_policy_arrows(policy1)

    for t in range(500):
        env.render(mode='graphic')

        action = np.argmax(policy1[curr_state])
        print('go ' + env.action_descriptors[action])
        curr_state, reward, done, info = env.step(action)

        if done:
            print('Terminal state found in {} steps'.format(t + 1))
            env.render(mode='graphic')
            time.sleep(5)
            break







class DQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DQN, self).__init__()

        self.linear1 = nn.Linear(input_dim, 120)
        self.linear2 = nn.Linear(120, output_dim)

    def forward(self, input):
        out = self.linear1(input)
        out = F.relu(out)
        out = self.linear2(out)
        return out


EPISODES = 1000


class DQNAgent:
    def __init__(self, state_size, action_size):

        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=2000)
        self.gamma = 0.95  # discount rate
        self.epsilon = 1.0  # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001

        # Our DQN
        self.model = DQN(state_size, action_size)
        self.criteria = nn.MSELoss()
        self.opt = optim.Adam(self.model.parameters(), lr=self.learning_rate)

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):

        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)

        act_values = self.model(Variable(torch.Tensor(state))).data.numpy()
        return np.argmax(act_values[0])  # returns action

    def replay(self, batch_size):
        minibatch = random.sample(self.memory, batch_size)

        for state, action, reward, next_state, done in minibatch:

            target = reward

            if not done:
                next_state_v = Variable(torch.Tensor(next_state))
                target = (reward + self.gamma * np.amax(self.model(next_state_v).data.numpy()[0]))

            target_actual = self.model(Variable(torch.Tensor(state))).data.numpy()
            target_actual[0][action] = target

            self.opt.zero_grad()
            out = self.model(Variable(torch.Tensor(state)))
            loss = self.criteria(out, Variable(torch.Tensor(target_actual)))
            loss.backward()
            self.opt.step()

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def load(self, name):
        self.model.load_weights(name)

    def save(self, name):
        self.model.save_weights(name)


def run_lemon_or_apple():
    """
    Run a random agent on an environment that was save via ascii text file
    """

    print('\n' + '*' * 20 + 'Creating the lemon_or_apple map and running DQN agent on it' + '*' * 20 + '\n')
    env = GridUniverseEnv(custom_world_fp='../core/envs/maze_text_files/lemon_or_apple.txt', task_mode=True)
    # env = gym.make('CartPole-v1')
    print(env.observation_space)
    print(env.observation_space.shape)
    state_size = 2 #env.observation_space.shape[0]
    action_size = env.action_space.n
    agent = DQNAgent(state_size, action_size)
    # agent.load("./save/cartpole-dqn.h5")
    done = False
    batch_size = 32

    for e in range(EPISODES):
        state = env.reset()
        state = state.reshape((1, state_size))
        print(state)

        for t in range(500):
            env.render(mode='graphic')
            action = agent.act(state)
            next_state, reward, done, _ = env.step(action)
            state = next_state.reshape((1, state_size))
            agent.remember(state, action, reward, next_state, done)
            state = next_state

            if done:
                print("episode: {}/{}, score: {}, e: {:.2}"
                      .format(e, EPISODES, t, agent.epsilon))
                time.sleep(1)
                break
        if len(agent.memory) > batch_size:
            agent.replay(batch_size)

if __name__ == '__main__':
    # Run specific algorithms on GridUniverse
    # run_policy_and_value_iteration()
    # run_monte_carlo_evaluation()

    run_lemon_or_apple()
