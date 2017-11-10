from random import randint


class GridWorld:
    """
    Gridworld environment.
    """
    UP = 0
    RIGHT = 1
    DOWN = 2
    LEFT = 3

    def __init__(self, x_y_dim=3):
        self.x_y_dim = x_y_dim

        self.done = False
        self.curr_state = [0, 0]
        self.terminal_state = (self.x_y_dim, self.x_y_dim)
        self.info = 'No info'

    def step(self, selected_action):

        if selected_action == self.UP:
            self.curr_state[0] -= 1
        elif selected_action == self.DOWN:
            self.curr_state[0] += 1
        elif selected_action == self.LEFT:
            self.curr_state[1] -= 1
        elif selected_action == self.RIGHT:
            self.curr_state[1] += 1

        if self.curr_state[0] < 0:
            self.curr_state[0] = 0
        elif self.curr_state[0] > self.x_y_dim:
            self.curr_state[0] = self.x_y_dim
        if self.curr_state[1] < 0:
            self.curr_state[1] = 0
        elif self.curr_state[1] > self.x_y_dim:
            self.curr_state[1] = self.x_y_dim

        observation = self.curr_state
        reward = 0
        if self.curr_state[0] == self.terminal_state[0] and self.curr_state[1] == self.terminal_state[1]:
            reward = 10
            self.done = True

        return observation, reward, self.done, self.info

    def render(self):
        grid = [['o', 'o', 'o', 'o'], ['o', 'o', 'o', 'o'], ['o', 'o', 'o', 'o'], ['o', 'o', 'o', 'o']]

        grid[self.curr_state[0]][self.curr_state[1]] = 'x'
        grid[self.terminal_state[0]][self.terminal_state[1]] = 'T'

        for row in grid:
            for el in row:
                print(el, end=' ')
            print()
        print()

    def reset(self):
        self.done = False
        self.curr_state = [0, 0]
        self.terminal_state = (self.x_y_dim, self.x_y_dim)
        return self.curr_state


if __name__ == '__main__':
    env = GridWorld()
    for i_episode in range(1):
        observation = env.reset()
        for t in range(100):
            env.render()
            # print(observation)
            action = randint(0, 3)
            observation, reward, done, info = env.step(action)

            if done:
                print("Episode finished after {} timesteps".format(t + 1))
                env.render()
                break
