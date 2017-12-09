import numpy as np
from gridworld_env import GridWorldEnv

gw_env = GridWorldEnv()
policy0 = np.ones([gw_env.world.size, len(gw_env.actions_list)]) / len(gw_env.actions_list)
v0 = np.zeros(gw_env.world.size)


def policy_eval(policy, env, discount_factor=1.0, threshold=0.00001, **kwargs):
    threshold = 0.01
    if 'value_function' in kwargs:
        v = kwargs['value_function']
    else:
        v = np.zeros(env.world.size)

    while True:
        delta = 0
        for state in range(env.world.size):
            v_update = 0
            for action, action_prob in enumerate(policy[state]):
                next_state, reward, done = env.look_step_ahead(state, action)
                v_update += action_prob * (reward + discount_factor * v[next_state])
            delta = max(delta, np.abs(v_update - v[state]))
            v[state] = v_update
        if delta < threshold:
            break
    return v


v1 = policy_eval(policy0, gw_env, value_function=v0)


def policy_improvement(policy, env, discount_factor=1.0):
    while True:
        v = policy_eval(policy, env, discount_factor)
        policy_stable = True

        for state in range(env.world.size):
            chosen_action = np.argmax(policy[state])
            action_values = np.zeros(env.action_space_size)
            for action in range(env.action_space_size):
                for next_state, reward in env.look_step_ahead(state, action):
                    action_values[action] += reward + discount_factor * v[next_state]
            best_action = np.argmax(action_values)

            if chosen_action != best_action:
                policy_stable = False
            policy[state] = np.eye(env.action_space_size)[best_action]

        if policy_stable:
            return policy, v


policy1 = policy_improvement(policy0, gw_env)
v2 = policy_eval(policy1, gw_env, value_function=v1)
# and keep alternating between policy evaluation and improvement until convergence