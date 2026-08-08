# ConfliBERT AEC-QA
Since ConfliBERT cannot natively be found on Ollama, we will have to make a manual Modelfile for Ollama. To do so, we will need a .gguf file of ConfliBERT which is doesn't exist on the model card on HuggingFace. 

I have already gone through the process to create the .gguf file for ConfliBERT and will provide in this part of the repo. That said, I will also provide instructions as to how acquire the .gguf for replication purposes. If you don't wish to know how the .gguf was made, you can skip down to the "Build Modelfile" section.

## Create .gguf for ConfliBERT
The ConfliBERT model on HuggingFace does not have a .gguf file, so we will have to make our own. To do so, we will need to clone two repositories:
  1. A fork of the llama.cpp repo that was modified to exclusively convert ConfliBERT. - https://github.com/AshtonElliott/llama.cpp-ConfliBERT-Conversion-
  2. ConfliBERT HuggingFace Model Card - https://huggingface.co/snowood1/ConfliBERT-scr-uncased
