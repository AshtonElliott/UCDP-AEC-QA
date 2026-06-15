# Ollama CLI Installation and serving on Terminal 1

**Mind the "<YOUR_PROJECT_DIR>" placeholder.**

```bash
# Enter the project folder
cd <YOUR_PROJECT_DIR>

# Download the Ollama Linux archive
curl -fL https://ollama.com/download/ollama-linux-amd64.tar.zst -o ollama.tar.zst

# Create a user-local install folder and extract Ollama there
mkdir -p ~/.local
tar -I zstd -xf ollama.tar.zst -C ~/.local

# Load Ollama from your user install and store models in the project folder. 
# MAKE SURE THE MODEL WEIGHTS FILES ARE STORED IN A FOLDER THAT DOES NOT HAVE TOO MUCH LIMITS ON THE SIZE.
export PATH="$HOME/.local/bin:$PATH"
export OLLAMA_MODELS="<YOUR_PROJECT_DIR>/ollama_models"

# Check Ollama and start the server
ollama --version
ollama serve
```

# Running the script on Terminal 2

```bash
# Enter the project folder
cd <YOUR_PROJECT_DIR>

# Load Ollama from your user install and store models in the project folder
export PATH="$HOME/.local/bin:$PATH"
export OLLAMA_MODELS="<YOUR_PROJECT_DIR>/ollama_models"

# Create and activate a project Python environment. I am using venv here for testing but you can use other if you like.
# python3 -m venv .venv
source .venv/bin/activate

# Install the Python packages used by the scripts
# python -m pip install --upgrade pip
# python -m pip install ollama tqdm

# Pull the model from the running Ollama server, using the smaller model for testing
ollama pull gemma3:4b
ollama pull gemma4:12b
ollama pull gemma4:31b

# Run the sync script or the async script
python HPC_QA.py
python HPC_QA_async.py
```

# Checking status in terminal 3

```bash
nvidia-smi
ps aux | grep ollama
ps aux | grep python
watch -n 2 nvidia-smi
```