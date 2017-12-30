## Installation

### Requirements
You need to have installed the following packages before starting with the set up of the environment for this project:   
cmake, zlib1g-dev, libjpeg-dev, xvfb, libav-tools, xorg-dev, libboost-all-dev, libsdl2-dev, swig

To install them you can follow the instructions given in the official repository:
https://github.com/openai/gym#installation

If you already have them you can skip this step. An explanation on how to set up your virtual environment is given
provided that you are using Anaconda.

### Create conda environment

First you need to install Anaconda. Select your corresponding installation file for Python 3.6. version following 
the instructions that can be found on the official documentation:
https://conda.io/docs/user-guide/install/index.html

Make sure you add conda to your system environment variables during installation, otherwise do it manually afterwards.
You can check that the installation was successful by opening a terminal and entering:

```
conda --help
```

### If this is your first time using this repository, or you would like to create a new environment for using it:

Open a terminal, go to the main directory of the repository and enter one of the following commands:

For installation on default path (/home/<username>/anaconda3/envs/<new_environment_name>):  
`conda env create -n <new_environment_name> -f repository_dir/requirements.yml`

For installation on another path:  
`conda env create -p c:/path/to/installation/folder/<new_environment_name> -f repository_dir/requirements.yml`

For more information on this command, please check the official documentation:  
https://conda.io/docs/user-guide/tasks/manage-environments.html#creating-an-environment-from-an-environment-yml-file

### If you already have an existing conda environment and only want to update the corresponding packages:

`conda env update -n <name_of_environment_to_update> -f repository_dir/requirements.yml`

For more information on this command, please check the official documentation.
https://conda.io/docs/commands/env/conda-env-update.html

### Final steps

Activate the conda environment:  

`source activate <name_of_environment>`

And install OpenAI Gym

`pip install gym`

Now you should be able to run `atari_example.py`.
