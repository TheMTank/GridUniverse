## GridWorld

GridWorld is an environment with discrete states and actions which is suited to simple tabular RL algorithms.  
Most of the inspiration for this environment comes from Lectures 1-5 of David Silver's 
[RL course](http://www0.cs.ucl.ac.uk/staff/d.silver/web/Teaching.html) and the great [Sutton and Barto book](http://ufal.mff.cuni.cz/~straka/courses/npfl114/2016/sutton-bookdraft2016sep.pdf). 

It is a great environment to test beginner RL algorithms like Dynamic Programming (Policy and Value Iteration) 
and other algorithms like Monte Carlo and TD learning.

## Constructor and initialisation

You can create the default environment like this:

`env = GridWorldEnv()`

The default grid_shape is (4, 4) but you can specify any square shape you want e.g.:

`env = GridWorldEnv(grid_shape=(55, 7))`

Other options include where to place terminal states and walls:

`env = GridWorldEnv(walls=[5, 10], terminal_states=[15])`

## The main interface: env.step()

`action = env.action_space.sample()`  
`print(env.action_descriptors[action])`  

This code first samples a random action from the action_space which is just an integer from 0-3 
representing the actions of going:

`self.action_descriptors = ['up', 'right', 'down', 'left']`

We can then feed the environment an action:

`observation, reward, done, info = env.step(action)`  

This will return the new state of the agent in "observation", the "reward" received, 
if the environment has terminated ("done") and other "info".  
This is exactly the same way how OpenAI gym's environment interface is.

## Render

`env.render()`

Will render the environment in ascii format.  
A version is being worked on to render using [pyglet](https://bitbucket.org/pyglet/pyglet/wiki/Home). 

## Other important info

`env.reset()`

Will reset the environment and the agent's position to the start again.

`def look_step_ahead(self, state, action):`

This function takes as parameters a state and an action and will return state returned by the environment. 
This can be used for implementing Dynamic Programming algorithms.

**Agent state**

The agent's state is represented by a single number between 0-N where N is the number of states in the grid.
This makes it easy for tabular matrix algorithms (e.g. Dynamic Programming) to run on the environment.
