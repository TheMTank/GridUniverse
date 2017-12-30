import unittest

from core.envs.gridworld_env import GridWorldEnv


class TestGridWorld(unittest.TestCase):
    def test_gridworld_wall(self):
        env = GridWorldEnv(walls=[4])
        observation = env.reset()
        env.render()
        action = 1 # go right

        observation, reward, done, info = env.step(action)

        env.render()
        self.assertTrue(observation == 0)

    def test_default_gridworld_completion_in_six_steps(self):
        env = GridWorldEnv()

        actions_to_take = [1, 1, 1, 0, 0, 0] # 3 rights and 3 ups
        observation = env.reset()
        for t in range(100):
            env.render()
            action = actions_to_take[t]
            observation, reward, done, info = env.step(action)

            if done:
                print("Episode finished after {} timesteps".format(t + 1))
                self.assertTrue((t + 1) == 6)
                break

if __name__ == '__main__':
    unittest.main()
