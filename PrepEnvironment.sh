#!/bin/bash

source ~/.bashrc

# Create Conda Environment
conda create -p $PWD/Python_Env python=3.12

# Start Conda Environment
conda activate $PWD/Python_Env

# Install Ollama Python package
pip install ollama

# Deactivate Environment
conda deactivate
