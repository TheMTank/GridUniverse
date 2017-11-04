# RL_problems

Repository created with the goal of developing reinforcement learning algorithms and be able to reproduce 
experiments for different environments. Currently working exclusively on openAI gym.

## Example of use
Currently no example available

## Installation

### Create conda environment

First you need to install Anaconda. Select your corresponding installation file for Python 3.6. version following 
the instructions that can be found on the official Mkdocs documentation.
https://conda.io/docs/user-guide/install/index.html

Make sure you add conda to your system environment variables during installation, otherwise do it manually afterwards.
You can check that the installation was successfull by opening a terminal and enter one of the following commands:

```
conda --help
conda -h
```

### If it is your first time using this repository, or you would like to create a new environment for using it:

open a terminal, go to the main directory of the repository and enter the following command:

`conda env export -p c:/path/to/installation/folder/<new_environment_name> -f repository_dir/requirements.yml`

For more information on this command, please check the official documentation.
https://conda.io/docs/commands/env/conda-env-export.html

### If you already have an existing conda environment and only want to update the corresponding packages:

`conda env update -n <name_of_environment_to_update> -f repository_dir/requirements.yml`

For more information on this command, please check the official documentation.
https://conda.io/docs/commands/env/conda-env-update.html

## Installation of openAI gym

Currently OpenAI Gym is supported only in Linux distributions. 
For windows installation everything works fine except from the "atari" games package. 
To be able to run this package on windows 10, follow these steps:

### Activate the conda environment and install the required packages entering the following commands:

```
activate <name_of_environment>
pip install gym
```

Now you should be able to run "simple_test.py" to simulate "MsPacman-v0" or any other atari example.

For a general documentation on how the environment works refer to the official documentation on https://gym.openai.com/docs

## API Reference

The documentation is contained in the docs folder.

## Tests

How to run simple_test.py

## Contributors

Currently not looking for help from contributors.

## License

For information about the license of this code please refer to the corresponding file "license.md"
