# RL_problems

This repository was created with the goal of developing and reproducing many different Reinforcement Learning algorithms 
and then being able to run large scale experiments using these algorithms on different open-source 
environments as well as our own custom environments.  

Currently working exclusively on OpenAI gym and a GridWorld environment made by us.

## GridWorld

We have created a highly customisable GridWorld OpenAI gym environment to enable research and large scale testing, evaluation and comparison of new and old RL algorithms. 
It will also allow large scale experimentation in exciting new areas like language grounding, Meta-Learning and Exploration.

![GIF](docs/maze_solver_BFS_10_times.gif)

## GridWorld features and plans section

This environment defines a bidimensional grid representing the world as a discrete state and action space which includes the following entities that can be customized:
- [x] Walls. States forbidden to enter to or cross to reach other states.
- [x] Lava. Terminal states with a high negative reward.
- [x] Random maze generator, with multiple different maze generation algorithms.
- [x] “Classic” pathfinding/graph search algorithms to compare against RL algorithms.
- [x] Ascii and OpenGL based rendering (using pyglet)

Additionally, we plan to include the following features:
- [] 3 different “fruits” that can be collected. Objects that can provide positive and negative rewards in different amounts. 
- [] Levers and keys. Elements that modify the environment by removing particular walls/doors.
- [] Wind
- [] Human control of the agent
- [] Different sensor configurations for the agent so it can be defined whether the agent field of view is constrained to the current state, surrounding states or the complete grid.
- [] Natural language grounding. Implementation of algorithms to follow natural language instructions e.g. “Collect the lemon and then the apple in that order”
- [] Meta-Learning/Multi-task learning environment interface and algorithms run on a large number of levels of varying difficulty
- [] Sokoban extension. An unsolved state of the art AI problem based on a classic video game.

Algorithms created from scratch on the environment:
- [x] Policy Iteration
- [x] Value Iteration
- [x] Monte-Carlo (MC) Learning with variations
- [] Temporal Difference (TD) Learning with variations
- [] Value Approximation
- [] SARSA + Q-Learning
- [] Policy Gradients
- [] Actor Critic

## Examples of use

To run examples of policy/value iteration or Monte Carlo evaluation on our GridWorld environment run:  

`python examples/gridworld_alg_examples.py`

To run random agents on random or preset grids run:

`python examples/gridworld_env_examples.py`

To run tests:  

`python tests/test_gridworld.py`

## Installation

Follow the instructions at this link: [Installation instructions](https://github.com/beduffy/RL_problems/tree/master/docs/Installation.md)

## Running OpenAI gym

Now you should be able to run: 

`python examples/atari_example.py`

You should see a small window that automatically plays "Space Invaders" if everything is working correctly.

For a general documentation on how the environment works refer to the [official documentation](https://gym.openai.com/docs).

## Running GridWorld

To see how the environment is used explore the code within:

`examples/gridworld_alg_examples.py`

`examples/gridworld_env_examples.py`

The `run_default_gridworld()` function shows the simplest way to use the environment.  
For more info check the [GridWorld Documentation](https://github.com/beduffy/RL_problems/tree/master/docs/GridWorld.md)

## API Reference

You can look within the docs, examples and tests folders for more info on all aspects of this repo. 

## Contributors

Currently not looking for help from contributors. Further information to be added in the future.

## License

For information about the license of this code please refer to the corresponding file "license.txt"
