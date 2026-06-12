#!/bin/bash

# Allocate Resources

#SBATCH --job-name=ollama_QA
#SBATCH --partition=h100
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=ollama_job_%j.log

# Load apptainer
ml apptainer

# Unset Proxies
unset http_proxy
unset https_proxy

# Load Ollama Models
export OLLAMA_MODELS=/groups/pbrandt/aee230007/Ollama_Models
export APPTAINERENV_OLLAMA_MODELS=/groups/pbrandt/aee230007/Ollama_Models

# Start Ollama
apptainer run --nv -B /groups/pbrandt/aee230007:/groups/pbrandt/aee230007 ollama.sif serve &

# Allow server to intialize
sleep 180

# Start Conda Environment
source ~/.bashrc
conda activate $PWD/Python_Env

# Run python script
python3 HPC_QA.py
