## RL_problems

Repository created with the goal of developing reinforcement learning algorithms and be able to reproduce 
experiments for different environments. Currently working exclusively on openAI gym.

# Example of use
Currently no example available

# Installation

1. Create conda environment
1.1 First you need to install Anaconda. Select your corresponding installation file for Python 3.6. 
version following the instructions that can be found on the official Mkdocs documentation.
https://conda.io/docs/user-guide/install/index.html

Make sure you add conda to your system environment variables during installation, otherwise do it manually afterwards.
You can check that the installation was successfull by opening a terminal and enter one of the following commands
 conda --help
 conda -h

1.2. If it is your first time using this repository, or you would like to create a new environment for using
it open a prompt and enter the following command:

 conda env export -p c:/path/to/installation/folder/ -f repository_dir/requirements.yml

For more information on this command, please check the official documentation.
https://conda.io/docs/commands/env/conda-env-export.html

2. Installation of openAI gym
Currently OpenAI Gym is supported only in Linux distributions. 
For windows installation everything works fine except from the "atari" games package. 
To be able to run this package on windows 10, follow these steps:

3. Activate the conda environment and enter the following command
 pip install gym
 pip install git+https://github.com/Kojoley/atari-py.git

Now you should be able to run "simple_test.py" to simulate "MsPacman-v0" or any other atari example.

For a general documentation on how the environment works refer to the official documentation on https://gym.openai.com/docs

# API Reference

The documentation is contained in the docs folder.

# Tests

How to run simple_test.py

# Contributors

Currently not looking for help from contributors.

# License

For information about the license of this code please refer to the corresponding file "license.md"
