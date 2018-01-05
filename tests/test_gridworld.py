import unittest

from core.envs.gridworld_env import GridWorldEnv


class TestGridWorld(unittest.TestCase):
    def test_gridworld_wall(self):
        env = GridWorldEnv(walls=[1])
        env.render()
        action = 1 # go right

        observation, reward, done, info = env.step(action)
        print('go ' + env.action_descriptors[action])

        env.render()
        self.assertTrue(observation == 0) # check if in same place

    def test_default_gridworld_completion_in_six_steps(self):
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

    def test_large_gridworld_completion_in_48_steps(self):
        env = GridWorldEnv(grid_shape=(25, 25))

        actions_to_take = [1] * 24 + [2] * 24 # 24 steps right + 24 steps down
        num_actions = len(actions_to_take)
        print('Num actions to take to get to terminal state: {}'.format(num_actions))
        for t in range(100):
            action = actions_to_take[t]
            observation, reward, done, info = env.step(action)

            if done:
                print("Episode finished after {} timesteps".format(t + 1))
                self.assertTrue((t + 1) == num_actions)
                break

if __name__ == '__main__':
    unittest.main()
