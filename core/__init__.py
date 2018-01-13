from gym.envs.registration import register

register(
    id='gridworld-v0',
    entry_point='core.envs:GridWorldEnv',
)
