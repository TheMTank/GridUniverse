import time

from core.envs.gridworld_env import GridWorldEnv

def run_default_gridworld():
    """
    Run a random agent on the default gridworld.
    This piece of code shows the main interface to the environment. This runs in ascii format
    """

    print('\n' + '*' * 20 + 'Starting to run random agent on default GridWorld' + '*' * 20 + '\n')
    env = GridWorldEnv()
    for i_episode in range(1):
        observation = env.reset()
        for t in range(100):
            env.render()  # set mode='graphic for pyglet render
            action = env.action_space.sample()
            print('go ' + env.action_descriptors[action])
            observation, reward, done, info = env.step(action)

            if done:
                print("Episode finished after {} timesteps".format(t + 1))
                break

def run_gridworld_from_text_file():
    """
    Run a random agent on an environment tat was save via ascii text file.
    Check core/envs/maze_text_files for examples or the _create_custom_world_from_text() function within the environment.
    """

    print('\n' + '*' * 20 + 'Creating a pre-made GridWorld from text file and running random agent on it' + '*' * 20 + '\n')
    env = GridWorldEnv(custom_world_fp='../core/envs/maze_text_files/test_env.txt')
    # env = GridWorldEnv(custom_world_fp='../core/envs/maze_text_files/maze_21x21.txt')
    # env = GridWorldEnv(custom_world_fp='../core/envs/maze_text_files/maze_101x101.txt')
    for i_episode in range(1):
        observation = env.reset()
        for t in range(1000):
            env.render(mode='graphic')
            action = env.action_space.sample()
            # print('go ' + env.action_descriptors[action])
            # time.sleep(0.1) # uncomment to watch slower
            observation, reward, done, info = env.step(action)
            if done:
                print("Episode finished after {} timesteps".format(t + 1))
                break

def run_random_maze():
    """
    Run a random agent on a randomly generated maze. If random_maze parameter is set to True,
    a maze generation algorithm will place walls to form the maze in the requested shape.
    """

    print('\n' + '*' * 20 + 'Creating a random GridWorld and running random agent on it' + '*' * 20 + '\n')
    env = GridWorldEnv(grid_shape=(11, 11), random_maze=True)
    # env = GridWorldEnv(grid_shape=(101, 101), random_maze=True)
    # env = GridWorldEnv(grid_shape=(49, 51), random_maze=True)
    # env = GridWorldEnv(grid_shape=(51, 49), random_maze=True)
    # todo print to user how long is left, so they can get comfortable with how constrained random search works. Step number out of step number?
    # todo exiting shouldn't crash everything
    for i_episode in range(1):
        observation = env.reset()
        for t in range(1000):
            env.render(mode='graphic')
            env.step_num = t
            action = env.action_space.sample()
            # print('go ' + env.action_descriptors[action])
            observation, reward, done, info = env.step(action)
            if done:
                print("Episode finished after {} timesteps".format(t + 1))
                break

def run_gridworld_with_lava():
    """
    Run a random agent on an environment with lava
    """

    print('\n' + '*' * 20 + 'Starting to run random agent on GridWorld with lava' + '*' * 20 + '\n')
    env = GridWorldEnv(grid_shape=(10, 10), lava_states=[4, 14, 24, 34, 44, 54, 64, 74])
    for i_episode in range(5):
        observation = env.reset()
        for t in range(100):
            env.render(mode='graphic')  # set mode='graphic for pyglet render
            action = env.action_space.sample()
            observation, reward, done, info = env.step(action)

            if done:
                print("Episode finished after {} timesteps".format(t + 1))
                print('Final states reward: ', reward)
                break

def run_gridworld_with_fruit():
    """
    Run a random agent on an environment with fruit. Fill whole grid with fruit.
    """

    print('\n' + '*' * 20 + 'Starting to run random agent on GridWorld with fruit' + '*' * 20 + '\n')
    #env = GridWorldEnv(grid_shape=(10, 10), apples=[4, 14, 24, 34, 44, 54, 64, 74])
    # env = GridWorldEnv(grid_shape=(10, 10), apples=[i for i in range(100)])

    world_shape = (10, 10)
    # world_shape = (30, 30)
    # world_shape = (100, 100)
    world_size = world_shape[0] * world_shape[1]

    apples = [i for i in range(world_size // 3)]
    lemons = [i for i in range(world_size // 3, int(2 * world_size // 3))]
    melons = [i for i in range(int(2 * world_size // 3), world_size)]

    # first_time = True

    # env = GridWorldEnv(grid_shape=world_shape, apples=[5, 4, 3], melons=[10, 15, 22], lemons=[55, 33, 22])
    env = GridWorldEnv(grid_shape=world_shape, apples=apples, melons=melons, lemons=lemons)
    string_actions_to_take = (['DOWN'] * world_shape[1] + ['RIGHT'] + ['UP'] * world_shape[1] + ['RIGHT']) * world_shape[1]

    actions_to_take = [env.action_descriptor_to_int[action_desc] for action_desc in string_actions_to_take]
    for i_episode in range(5):
        observation = env.reset()
        # for t in range(1000):
        for t, action in enumerate(actions_to_take):
            env.render(mode='graphic')  # set mode='graphic for pyglet render

            # if first_time:
            #     first_time = False
            #     time.sleep(7)
            # action = env.action_space.sample()
            observation, reward, done, info = env.step(action)
            # time.sleep(1)
            if done:
                print("Episode finished after {} timesteps".format(t + 1))
                print('Final states reward: ', reward)
                break

if __name__ == '__main__':
    # Run random agent on environment variations
    # run_default_gridworld()
    # run_gridworld_from_text_file()
    # run_random_maze()
    # run_gridworld_with_lava()

    run_gridworld_with_fruit()
