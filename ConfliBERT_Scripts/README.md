# ConfliBERT AEC-QA
Since ConfliBERT cannot natively be found on Ollama, we will have to make a manual Modelfile for Ollama. To do so, we will need a .gguf file of ConfliBERT which is doesn't exist on the model card on HuggingFace. 

I have already gone through the process to create the .gguf file for ConfliBERT and will provide in this part of the repo. That said, I will also provide instructions as to how acquire the .gguf for replication purposes. If you don't wish to know how the .gguf was made, you can skip down to the "Build Modelfile" section.

## Create .gguf for ConfliBERT
The ConfliBERT model on HuggingFace does not have a .gguf file, so we will have to make our own. To do so, we will need to clone the following repository:
  - A fork of the llama.cpp* repo that was modified to exclusively convert ConfliBERT. - https://github.com/AshtonElliott/llama.cpp-ConfliBERT-Conversion-

We will also need a HuggingFace Token which will require a HuggingFace Account.
After logging or creating an account, create a token by clicking on your profile and then click "Access Tokens".

<img width="1552" height="868" alt="HG_Access_Tokens" src="https://github.com/user-attachments/assets/563b6c87-9d5e-407f-a946-85153116fa3a" />

You will have the ability to create a token in the top right.
The conversion was done with a Fine-grained token with the Read-Only preset, so I recommend to do that.

<img width="1115" height="803" alt="HG_Create_Token" src="https://github.com/user-attachments/assets/df8c7a3d-3bd8-4554-94ab-a7121f2fbad9" />

Once the token is created, copy the token-ID as we will need that later.

After creating the token, you will then need to set up the Python Environment.
This can be done by doing the following**:
```bash
  python -m venv ConfliBERT_Conversion
```

After creating and activating the python environment, upgrade pip using the following:
```bash
  python -m pip install --upgrade pip setuptools wheel
```

Then, navigate to the directory of the cloned github page on the command prompt. Then, run the following:
```bash
  # Navigate to the requirements directory.
  cd requirements

  # Install requirements and dependencies.
  pip install -r requirements-convert_hf_to_gguf.txt
  pip install -r requirements-convert_hf_to_gguf_update.txt

  # Revert a directory to use the Python Scripts.
  cd ..

  # Run the conversion scripts.
  # Use the HuggingFace token from earlier here.
  python3 convert_hf_to_gguf_update.py <token_id> --full
  python3 convert_hf_to_gguf.py models/tokenizers/conflibert-scr-uncased/ --outfile models/ggml-conflibert-scr-uncased.gguf
```

After running all the above, you should have a file of ConfliBERT in .gguf.

## Create ConfliBERT Modelfile
W.I.P.


## Notes
*Contrary to the name of the repo, we are not using C or C++ here. While the original repo is designed to do so, the only part of the repo we are using is the conversion scripts which are in python.

**It is highly advised to create a virtual environment through Python instead of through Anaconda as the pre-installed packages of Anaconda cause additional installation issues while downloading from the requirements.txt files from the llama.cpp repo.
