import unittest

from core.envs.gridworld_env import GridWorldEnv


class TestGridWorld(unittest.TestCase):
    def test_gridworld_wall(self):
        env = GridWorldEnv(walls=[4])
        observation = env.reset()
        env.render()
        action = 1 # go right
        print('go ' + env.action_descriptors[action])

        observation, reward, done, info = env.step(action)

        env.render()
        self.assertTrue(observation == 0)

if __name__ == '__main__':
    unittest.main()
