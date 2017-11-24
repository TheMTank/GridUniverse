"""
Implementation of Policy Iteration in Gridworld as from Sutton examples explained in David Silver's course.
"""
import numpy as np


""" 
Generate transition matrix. 25% of moving up, left, down or right. If moving against the wall you end up on the
previous state
"""
transition_matrix0 = np.zeros((4, 4, 4, 4))
for state_row in range(transition_matrix0.shape[0]):
    for state_col in range(transition_matrix0.shape[1]):
        for transition_row in range(transition_matrix0.shape[0]):
            for transition_col in range(transition_matrix0.shape[1]):
                if (np.absolute(state_row - transition_row) + np.absolute(state_col - transition_col)) == 1:
                    transition_matrix0[state_row, state_col, transition_row, transition_col] = 0.25
                elif state_row == transition_row and state_col == transition_col:
                    if state_row == 0 or state_row == 3:
                        transition_matrix0[state_row, state_col, transition_row, transition_col] += 0.25
                    if state_col == 0 or state_col == 3:
                        transition_matrix0[state_row, state_col, transition_row, transition_col] += 0.25
# print(transition_matrix0[0, 1])
""""
Reward matrix is -1 for every step except the two extreme corners
"""
reward_matrix = np.full((4, 4), -1)
reward_matrix[3, 3] = 0
reward_matrix[0, 0] = 0
# print(reward_matrix)
"""
First iteration from policy iteration
"""
v0 = np.zeros((4, 4))
for state_row in range(transition_matrix0.shape[0]):
    for state_col in range(transition_matrix0.shape[1]):
        v0 += np.multiply(transition_matrix0[state_row, state_col], v0)
v0 += reward_matrix
# print(v0)
""""
After this we should change our policy greedily with respect to V
"""
transition_matrix1 = np.zeros((4, 4, 4, 4))
for state_row in range(transition_matrix1.shape[0]):
    for state_col in range(transition_matrix1.shape[1]):
        valid_transition_values = {}
        for transition_row in range(transition_matrix1.shape[0]):
            for transition_col in range(transition_matrix1.shape[1]):
                if (np.absolute(state_row - transition_row) + np.absolute(state_col - transition_col)) == 1:
                    valid_transition_values[(transition_row, transition_col)] = v0[transition_row, transition_col]
                elif state_row == transition_row and state_col == transition_col:
                    if state_row == 0 or state_row == 3:
                        valid_transition_values['out_of_rows'] = v0[transition_row, transition_col]
                    if state_col == 0 or state_col == 3:
                        valid_transition_values["out_of_columns"] = v0[transition_row, transition_col]
        max_value = max(valid_transition_values.values())
        max_transition_list = [transition for transition, value in valid_transition_values.items()
                               if value == max_value]
        for transition in max_transition_list:
            if isinstance(transition, tuple):
                transition_matrix1[state_row, state_col, transition[0], transition[1]] = 1/len(max_transition_list)
            elif isinstance(transition, str):
                transition_matrix1[state_row, state_col, state_row, state_col] += 1 / len(max_transition_list)
            else:
                raise TypeError("Invalid transition type encountered")
# print(transition_matrix1[0, 3])
"""
And continue iterating until satisfied. For that purpose we have the following functions.
"""


def single_policy_evaluation(value_function, new_transition_matrix, terminal_states=None):
    """updates the new value function from the current value function and policy (defined by the transition matrix)"""
    if terminal_states is None:
        terminal_states = [(0, 0), (3, 3)]
    new_value_function = np.zeros((4, 4))
    for state_row in range(new_transition_matrix.shape[0]):
        for state_col in range(new_transition_matrix.shape[1]):
            for transition_row in range(transition_matrix1.shape[0]):
                for transition_col in range(transition_matrix1.shape[1]):
                    new_value_function[state_row, state_col] += np.multiply(new_transition_matrix[state_row,
                                                                                              state_col,
                                                                                              transition_row,
                                                                                              transition_col],
                                                                        value_function[transition_row, transition_col])
    new_value_function += reward_matrix
    for terminal_state in terminal_states:
        new_value_function[terminal_state[0], terminal_state[1]] = 0
    return new_value_function


def greedy_policy_from_value_function(value_function):
    transition_matrix = np.zeros((4, 4, 4, 4))
    for state_row in range(transition_matrix.shape[0]):
        for state_col in range(transition_matrix.shape[1]):
            valid_transition_values = {}
            for transition_row in range(transition_matrix.shape[0]):
                for transition_col in range(transition_matrix.shape[1]):
                    if (np.absolute(state_row - transition_row) + np.absolute(state_col - transition_col)) == 1:
                        valid_transition_values[(transition_row, transition_col)] = value_function[transition_row,
                                                                                                   transition_col]
                    elif state_row == transition_row and state_col == transition_col:
                        if state_row == 0 or state_row == 3:
                            valid_transition_values['out_of_rows'] = value_function[transition_row, transition_col]
                        if state_col == 0 or state_col == 3:
                            valid_transition_values["out_of_columns"] = value_function[transition_row, transition_col]
            max_value = max(valid_transition_values.values())
            max_transition_list = [transition for transition, value in valid_transition_values.items()
                                   if value == max_value]
            for transition in max_transition_list:
                if isinstance(transition, tuple):
                    transition_matrix[state_row, state_col, transition[0], transition[1]] = 1 / len(
                        max_transition_list)
                elif isinstance(transition, str):
                    transition_matrix[state_row, state_col, state_row, state_col] += 1 / len(max_transition_list)
                else:
                    raise TypeError("Invalid transition type encountered")
    return transition_matrix


""""
And we continue iterating until satisfied
"""
print("Policy iteration algorithm")
print("Initial value function:")
print(v0)
n_iterations = 3
v_n = v0
for iteration in range(n_iterations):
    transition_matrix_n = greedy_policy_from_value_function(v_n)
    v_n = single_policy_evaluation(v_n, transition_matrix_n)
    print("Iteration" + str(iteration))
    print(v_n)
"""
On the contrary, policy evaluation would always iterate on the same policy
"""
print("Policy evaluation algorithm:")
transition_matrix = transition_matrix0  # same probability of going anywhere nearby
n_evaluations = 10
v_n = v0
for iteration in range(1, n_evaluations):
    v_n = single_policy_evaluation(v_n, transition_matrix)
    print(v_n)
