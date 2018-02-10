import time
import unittest

from core.envs.gridworld_env import GridWorldEnv


class TestGridWorld(unittest.TestCase):
    def test_lever_parameter_raise_correct_exceptions(self):
        """
        Test the numerous ways in which levers should raise exceptions
        """
        # Check if crashes if not a dictionary
        with self.assertRaises(TypeError):
            GridWorldEnv(levers=[5, 3])

        # Need walls for levers
        with self.assertRaises(ValueError):
            GridWorldEnv(levers={5: 3})

        # Wall (3) linked to lever state (5) is not a wall state
        with self.assertRaises(ValueError):
            GridWorldEnv(walls=[4], levers={5: 3})

        # How it should work
        try:
            GridWorldEnv(walls=[3], levers={5: 3})
        except:
            self.fail("Should not crash here. Wall state 3 is the value of lever state 5. ")

        # Lever state can not be placed on top of a wall
        with self.assertRaises(ValueError):
            GridWorldEnv(walls=[4, 8], levers={4: 8})

        # non-int key
        with self.assertRaises(TypeError):
            GridWorldEnv(walls=[4, 8], levers={'a': 8})

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
        Test whether we can complete the GridWorld created from the text file within maze_text_files folder
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

    def test_custom_gridworld_from_text_file_with_lever(self):
        """
        Test whether we can complete the GridWorld created from the text file
        within maze_text_files folder. This level contains levers so it also tests that the
        functionality of levers works correctly. If anything changes anywhere, this test will fail.
        """

        env = GridWorldEnv(custom_world_fp='../core/envs/maze_text_files/lever_level_2.txt')
        env.render()

        # Very particular path to take for agent to go to each lever and finally get to terminal state
        actions_to_take = [env.action_descriptor_to_int['DOWN']] + [env.action_descriptor_to_int['RIGHT']] * 6 + \
                          [env.action_descriptor_to_int['LEFT']] * 5 + [env.action_descriptor_to_int['DOWN']] * 5 + \
                          [env.action_descriptor_to_int['RIGHT']] * 3 + [env.action_descriptor_to_int['UP']] * 2 + \
                          [env.action_descriptor_to_int['RIGHT']] * 2 + [env.action_descriptor_to_int['LEFT']] * 2 + \
                          [env.action_descriptor_to_int['DOWN']] * 2 + [env.action_descriptor_to_int['RIGHT']] * 2

        for step_no, action in enumerate(actions_to_take):
            # env.render()  # set mode='graphic' for pyglet render
            observation, reward, done, info = env.step(action)

            if done:
                print("Episode finished after {} timesteps".format(step_no + 1))
                break

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
