# ConfliBERT AEC-QA
Since ConfliBERT cannot natively be found on Ollama, we will have to make a manual Modelfile for Ollama. To do so, we will need a .gguf file of ConfliBERT which is doesn't exist on the model card on HuggingFace. 

I have already gone through the process to create the .gguf file for ConfliBERT and will provide in this part of the repo. That said, I will also provide instructions as to how acquire the .gguf for replication purposes. If you don't wish to know how the .gguf was made, you can skip down to the "Build Modelfile" section.

## Create .gguf for ConfliBERT
The ConfliBERT model on HuggingFace does not have a .gguf file, so we will have to make our own. To do so, we will need to clone two repositories:
  1. A fork of the llama.cpp* repo that was modified to exclusively convert ConfliBERT. - https://github.com/AshtonElliott/llama.cpp-ConfliBERT-Conversion-
  2. ConfliBERT HuggingFace Model Card - https://huggingface.co/snowood1/ConfliBERT-scr-uncased

We will also need a HuggingFace Token which will require a HuggingFace Account.
After logging or creating an account, create a token by clicking on your profile and then click "Access Tokens".
*Insert Access Token Image here*

You will have the ability to create a token in the top right.
The conversion was done with a Fine-grained token with the Read-Only preset, so I recommend to do that.
*Insert image of Token Creation Here*

After creating the token, you will then need to set up the Python Environment.
This can be done by doing the following**:
```bash
  python -m venv ConfliBERT_Conversion
```



## Create ConfliBERT Modelfile
W.I.P.


## Notes
*Contrary to the name of the repo, we are not using C or C++ here. While the original repo is designed to do so, the only part of the repo we are using is the conversion scripts which are in python.

**It is highly advised to create a virtual environment through Python instead of through Anaconda as the pre-installed packages of Anaconda cause additional installation issues while downloading from the requirements.txt files from the llama.cpp repo.
