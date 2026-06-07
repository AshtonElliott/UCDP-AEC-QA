#!/bin/bash

# Allocate Resources

#SBATCH --job-name=ollama_QA
#SBATCH --partition=a30
#SBATCH --gres=gpu:1
#SBATCH --mem=40G
#SBATCH --time=01:00:00
#SBATCH --output=ollama_job_%j.log

# Load Ollama & Conda Module
module load ollama
module load miniconda

# Ensure GPU Visibility
export CUDA_VISIBLE_DEVICES=0

# Redirect Ollama Model Storage
export OLLAMA_MODELS=$PWD/Ollama_Models

# Start Conda Environment
source ~/.bashrc
conda activate $PWD/Python_Env

# Start Ollama Server
ollama serve &

# Wait for Server
sleep 20

# Run python script
python HPC_QA.py
