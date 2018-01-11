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

    def test_each_boundary(self):
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

if __name__ == '__main__':
    unittest.main()
