import json
import os
import re
import sys
import asyncio
from tqdm.asyncio import tqdm
import tensorflow
import torch
import transformers
import numpy
import sklearn
import pandas
from simpletransformers.model import TransformerModel
from transformers import TFBertForQuestionAnswering, BertTokenizer, pipeline


# Unset proxies
os.environ.pop("http_proxy", None)
os.environ.pop("https_proxy", None)

# Load HuggingFace Token
os.environ["HF_TOKEN"] = "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"

script_dir = os.path.dirname(os.path.abspath(__file__))
input_path = os.path.join(script_dir, 'trainsample.json')
output_path = os.path.join(script_dir, 'confliBERT_results_cb.json')

# Load dataset
with open(input_path, 'r') as f:
    dataset = json.load(f)

if isinstance(dataset, dict):
    dataset = [dataset]

sem = asyncio.Semaphore(5) # max concurrency

# Creating a function for one single async action
async def process_entry(idx, entry):
    async with sem:
        try:
            print(f"Processing entry {idx+1}/{len(dataset)}...")
            context = entry.get('source_article', '')
            question = entry.get('question', '')
            chunks = [context[i:i+1500] for i in range(0, len(context), 1500)]
            retrieved_context = "\n".join(chunks[:5])
            
            # Change ConfliBERT Model Here
            model_name = "salsarra/ConfliBERT-QA"
            
            # Load the model specifically with a Question Answering head
            model = TFBertForQuestionAnswering.from_pretrained(model_name, from_tf=True)
            tokenizer = BertTokenizer.from_pretrained(model_name)
            
            # Initialize the pipeline task
            qa_pipe = pipeline("question-answering", model=model, tokenizer=tokenizer, from_tf=True)
            
            # Acquire Results
            result = qa_pipe(context=retrieved_context, question=question, from_tf=True)
            
            spans = []
            for result in results:
                    if result['answer'] != "":
                        spans.append({
                            "end": result['end'],
                            "text": result['answer'],
                            "start": result['start'],
                            "labels": ["Answer"]
                        })
            # Check AFTER processing all labels
            if len(spans) == 0:
                entry['no_answer'] = "No arms or methods mentioned"
            else:
                entry['answer_labels'] = spans
        except Exception as e:
            print(f"Error processing entry {idx}: {e}", file=sys.stderr)
            entry['error'] = str(e)

# Main function to use async later on. 
# Use await instead of for loop for asyncio.
async def main():
    
    await tqdm.gather(*[
        process_entry(idx, entry)
        for idx, entry in enumerate(dataset)
    ])
    # Write ONCE after all entries processed
    print("Writing results...")
    with open(output_path, 'w') as f:
        json.dump(dataset, f, indent=4)
    print("Process complete. Results saved.")

# run main function async
asyncio.run(main())
