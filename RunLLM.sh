#!/bin/bash

# Load Ollama Module
module load ollama

# Redirect Ollama Model Storage
export OLLAMA_MODELS=$PWD/Ollama_Models

# Start Conda Environment
conda activate $PWD/Python_Env

# Start Ollama Server
ollama serve &

# Run python script
python HPC_QA.py