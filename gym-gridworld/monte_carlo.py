import numpy as np
from gym_gridworld.envs.gridworld_env import GridWorldEnv
import warnings


def monte_carlo_first_visit():
    """"
    To evaluate state s
    The first time-step t that state s is visited in an episode,
    Increment counter N(s) ← N(s) + 1
    Increment total return S(s) ← S(s) + Gt
    Value is estimated by mean return V(s) = S(s)/N(s)
    By law of large numbers, V(s) → vπ(s) as N(s) → ∞
    """
    pass


def monte_carlo_every_visit():
    """
    To evaluate state s
    Every time-step t that state s is visited in an episode,
    Increment counter N(s) ← N(s) + 1
    Increment total return S(s) ← S(s) + Gt
    Value is estimated by mean return V(s) = S(s)/N(s)
    Again, V(s) → vπ(s) as N(s) → ∞
    """
    pass

