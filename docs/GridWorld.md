## GridWorld

GridWorld is an environment with discrete states and actions which is suited to simple tabular RL algorithms.  
Most of the inspiration for this environment comes from Lectures 1-5 of David Silver's 
[RL course](http://www0.cs.ucl.ac.uk/staff/d.silver/web/Teaching.html) and the great [Sutton and Barto book](http://ufal.mff.cuni.cz/~straka/courses/npfl114/2016/sutton-bookdraft2016sep.pdf). 

It is a great environment to test beginner RL algorithms like Dynamic Programming (Policy and Value Iteration), 
other algorithms like Monte Carlo and TD learning, as well as non-RL algorithms like search and maze solving.

## Constructor and initialisation

You can create the default environment like this:

`env = GridWorldEnv()`

The default grid_shape is (4, 4) but you can specify any rectangle shape you want e.g.:

`env = GridWorldEnv(grid_shape=(55, 7))`

Other options include where to place terminal states and walls:

`env = GridWorldEnv(walls=[5, 10], terminal_states=[15])`

List format is expected e.g. walls=[5. 10] means that two walls (unwalkable areas) will be placed within the grid, at state indices 5 and 10.
Similar for terminal_states, If an agent ends up in a terminal state, the episode is over.

## The main interface: env.step()

`action = env.action_space.sample()`  
`print(env.action_descriptors[action])`  

This code first samples a random action from the action_space which is just an integer from 0-3 
representing the actions of going:

`self.action_descriptors = ['up', 'right', 'down', 'left'] # 0, 1, 2, 3` 

We can then feed the environment an action:

`observation, reward, done, info = env.step(action)`  

This will return the new state of the agent in "observation", the "reward" received, 
if the environment has terminated ("done") and other "info".  
This is exactly the same way that OpenAI gym's environment interface is.

See below at "Agent state" for more details on how it is represented.

## Render

`env.render()`

Will render the environment in ascii format.  
A version of the environment that can be rendered using [pyglet](https://bitbucket.org/pyglet/pyglet/wiki/Home) 
is being worked on at the moment. 

Currently, even if env.render() is never called by the user, it will still be called by the superclass when the environment is closed. 

## Other important info

`observation = env.reset()`

Will reset the environment and the agent's position to the start again and return this state.

`def look_step_ahead(self, state, action):`

This function takes as parameters a state and an action and the next state is returned by the environment. 
This can be used for implementing Dynamic Programming algorithms.

**Agent state**

The agent's state is represented by a single number between 0-N where N is the number of states in the grid.
This makes it easy for tabular matrix algorithms (e.g. Dynamic Programming) to run on the environment.

## Code and Examples

Explore the self-contained environment code within `core/envs/gridworld_env.py`.

Run the code within `examples/gridworld_examples.py` and `core/algorithms/policy_iteration.py` 
to see how to use the environment more clearly.
