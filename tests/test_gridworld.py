import unittest

from core.envs.gridworld_env import GridWorldEnv


class TestGridWorld(unittest.TestCase):
    def test_gridworld_wall_not_trespassed(self):
        """
        Test whether agent is still in the same place after moving into a wall
        """
        env = GridWorldEnv(walls=[1])
        env.render()
        action = 1 # go right

        observation, reward, done, info = env.step(action)
        print('go ' + env.action_descriptors[action])

        env.render()
        self.assertTrue(observation == 0) # check if in same place

    def test_default_gridworld_completion_in_six_steps(self):
        """
        Test whether the agent reaches a terminal state within the
        default square GridWorld within six steps by going right 3 times
        and then going down 3 times.
        """
        env = GridWorldEnv()

        actions_to_take = [1, 1, 1, 2, 2, 2] # 3 rights and 3 downs
        for t in range(100):
            env.render()
            action = actions_to_take[t]
            print('go ' + env.action_descriptors[action])
            observation, reward, done, info = env.step(action)

            if done:
                print("Episode finished after {} timesteps".format(t + 1))
                self.assertTrue((t + 1) == 6)
                break

    def test_large_gridworld_completion_in_53_steps(self):
        """
        Test whether the agent completes the a large rectangular GridWorld in the expected 53 steps
        """
        env = GridWorldEnv(grid_shape=(25, 30))

        actions_to_take = [1] * 24 + [2] * 29 # 24 steps right + 24 steps down
        num_actions = len(actions_to_take)
        print('Num actions to take to get to terminal state: {}'.format(num_actions))
        for t in range(100):
            action = actions_to_take[t]
            observation, reward, done, info = env.step(action)

            if done:
                print("Episode finished after {} timesteps".format(t + 1))
                break
        self.assertTrue((t + 1) == num_actions and done)

    def test_custom_gridworld_from_text_file(self):
        """
        Test whether we can complete the GridWorld created from the text file within
        """

        env = GridWorldEnv(custom_world_fp='../core/envs/maze_text_files/test_env.txt')
        actions_to_take = [2, 2, 2, 2, 2, 2, 2, 1]
        for step_no, action in enumerate(actions_to_take):
            env.render()
            print('go ' + env.action_descriptors[action])
            observation, reward, done, info = env.step(action)
            if done:
                print("Episode finished after {} timesteps".format(step_no + 1))

        self.assertTrue((step_no + 1) == len(actions_to_take) and done)

    def test_each_boundary_within_default_env(self):
        """
        The agent follows a sequence of steps to check each boundary acts as expected (the current
        observation should be the same as the previous if you move into a boundary).
        The agent tries the top-left, top-right and bottom-right corners while avoid the Terminal state.
        The step numbers where the agent ends up in the same state as previously
        are stored and then compared to the expected values and if exactly the same the test passes.
        """
        env = GridWorldEnv()

        # self.action_descriptors = ['up', 'right', 'down', 'left']
        actions_to_take = [3, 0, 1, 1, 1, 1, 2, 2, 3, 2, 2, 3, 3, 3]
        boundary_test_step_numbers = [0, 1, 5, 10, 13]
        collected_boundary_step_numbers = []
        prev_observation = env.reset()
        for step_no, action in enumerate(actions_to_take):
            env.render()
            print('go ' + env.action_descriptors[action])
            observation, reward, done, info = env.step(action)

            if observation == prev_observation:
                collected_boundary_step_numbers.append(step_no)

            prev_observation = observation

        print('collected_boundary_steps:', collected_boundary_step_numbers)
        print('boundary_test_steps', boundary_test_step_numbers)
        boolean_elementwise_comparison = [a == b for a, b in zip(collected_boundary_step_numbers, boundary_test_step_numbers)]
        print(boolean_elementwise_comparison)
        print(all(boolean_elementwise_comparison))
        self.assertTrue(all(boolean_elementwise_comparison))

    def test_lava(self):
        """
        Run agent into lava, test to see if episode ends with negative reward
        """

        env = GridWorldEnv(lava_states=[1])

        env.render()
        action = env.action_descriptor_to_int['RIGHT']
        observation, reward, done, info = env.step(action)

        self.assertTrue(reward == -10 and done)

    def test_lava_works_from_text_file(self):
        """
        Test whether we can end the episode by making the agent travel into lava in environment created from text file
        """

        env = GridWorldEnv(custom_world_fp='../core/envs/maze_text_files/test_env.txt')
        actions_to_take = [env.action_descriptor_to_int[action_desc] for action_desc in ['DOWN', 'DOWN', 'DOWN', 'RIGHT', 'RIGHT']]
        for step_no, action in enumerate(actions_to_take):
            env.render()
            print('go ' + env.action_descriptors[action])
            observation, reward, done, info = env.step(action)
            if done:
                print("Episode finished after {} timesteps".format(step_no + 1))

        self.assertTrue(reward == -10 and done)

    def test_lemons_melons_apples(self):
        """
        Test whether correct cumulative reward is received for collecting certain number of melons, lemons and apples.
        In places where there are no fruit, we lose -1 immediate reward (self.MOVEMENT_REWARD)
        """

        env = GridWorldEnv(custom_world_fp='../core/envs/maze_text_files/melons_lemons_apples_env.txt')
        actions_to_take = [env.action_descriptor_to_int[action_desc] for action_desc in
                           ['RIGHT', 'RIGHT', 'RIGHT', 'RIGHT', 'RIGHT', 'RIGHT',
                            'DOWN', 'LEFT', 'LEFT', 'DOWN', 'RIGHT', 'RIGHT']]

        num_lemons = num_apples = num_melons = 3
        expected_total_reward = num_apples * env.APPLE_REWARD + num_lemons * env.LEMON_REWARD + \
                                num_melons * env.MELON_REWARD + len(actions_to_take) * self.MOVEMENT_REWARD  # -1 for immediate reward
        cumulative_reward = 0

        for step_no, action in enumerate(actions_to_take):
            env.render()
            print('go ' + env.action_descriptors[action])
            observation, reward, done, info = env.step(action)
            cumulative_reward += reward

        print(cumulative_reward, expected_total_reward)
        self.assertTrue(cumulative_reward == expected_total_reward)

    def test_fruit_disappears(self):
        """
        Test that if you collect fruit, it disappears. Apples in this case
        """

        env = GridWorldEnv(apples=[1])

        env.render()
        action = env.action_descriptor_to_int['RIGHT']
        observation, reward1, done, info = env.step(action)

        env.render()
        action = env.action_descriptor_to_int['LEFT']
        observation, reward2, done, info = env.step(action)

        env.render()
        action = env.action_descriptor_to_int['RIGHT']
        observation, reward3, done, info = env.step(action)
        env.render()

        self.assertTrue(reward1 == (env.APPLE_REWARD - 1) and reward3 != (env.APPLE_REWARD - 1))

    def test_fruit_reset(self):
        """
        Test that after collecting fruit and the environment is reset, fruit reappears
        and that you still get same reward. Melon in this case
        """

        env = GridWorldEnv(melons=[1])

        env.render()
        action = env.action_descriptor_to_int['RIGHT']
        observation, reward1, done, info = env.step(action)
        env.render()

        print('Resetting environment')
        env.reset()
        env.render()
        observation, reward2, done, info = env.step(action)
        env.render()

        self.assertTrue(reward1 == reward2 and reward1 == (env.MELON_REWARD - 1))

if __name__ == '__main__':
    unittest.main()
