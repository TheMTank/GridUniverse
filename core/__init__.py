from gym.envs.registration import register

register(
    id='griduniverse-v0',
    entry_point='core.envs:GridUniverseEnv',
)
