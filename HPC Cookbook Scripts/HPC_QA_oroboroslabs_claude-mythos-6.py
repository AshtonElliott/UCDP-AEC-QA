import json
import os
import ollama
import re
import sys
import asyncio
from tqdm.asyncio import tqdm

# Unset proxies
os.environ.pop("http_proxy", None)
os.environ.pop("https_proxy", None)

script_dir = os.path.dirname(os.path.abspath(__file__))
input_path = os.path.join(script_dir, 'train2.json')
output_path = os.path.join(script_dir, 'OroborosLabs_claude_mythos_6_results2_cb.json')

# Load dataset
with open(input_path, 'r') as f:
    dataset = json.load(f)

if isinstance(dataset, dict):
    dataset = [dataset]

sem = asyncio.Semaphore(5) # max concurrency
client = ollama.AsyncClient()

# Creating a function for one single async action
async def process_entry(idx, entry):
    async with sem:
        try:
            print(f"Processing entry {idx+1}/{len(dataset)}...")
            context = entry.get('source_article', '')
            question = entry.get('question2', '')
            chunks = [context[i:i+1500] for i in range(0, len(context), 1500)]
            retrieved_context = "\n".join(chunks[:5])
            response = await client.chat(
                model='oroboroslabs/claude-mythos-6:latest',
                format = 'json',
                messages=[
                    {'role': 'system', 'content': (
                    'You are a Political Scientist trying to find what weapons/arms were used.'
                    'Your task is to extract words from the provided text.'
                    
                    # Example
                    '<example>\n'
                    'Input: "The local port was hit by Drone Strikes from the military. Many firearms and bombs were destroyed." \n'
                    'Response: \n'
                    '{\n'
                        "\"thinking\": \"The text mentions 'Drone Strikes' which were used. The text mentions firearms and bombs that were destroyed; since they were not used, they are not included.\",\n"
                        "\"extractions\": [\n"
                          "{\"word\": \"Drone Strikes\" \n"
                        "]\n"
                    '}\n'
                    '<example>\n\n'
                    
                    'Please analyze the provided context. First, use a <thinking> tag to reason through your extractions,'
                    'then provide the final JSON output containing the "extractions" list with the "word" key.'
                    )}
                ]
                
            )
            prediction = response['message']['content']
            
            # Read in JSON Produced by OroborosLabs's Claude
            data = json.loads(prediction)
            
            if isinstance(data, dict):
                extractions = data.get('extractions', [])
            elif isinstance(data, list):
                extractions = data
            else:
                extractions = []
            
            spans = []
            for label in extractions:
                text = label.get('word', '')
                
                # Apply Text & Label
                for match in re.finditer(re.escape(text), context, re.IGNORECASE):
                    spans.append({
                        "end": match.end(),
                        "text": context[match.start():match.end()],
                        "start": match.start(),
                        "labels": ["Answer"]
                    })
            if len(spans) == 0 or data.get('extractions', []) == "None found":
                entry['no_answer'] = "No Damage Detected"
            else:
                entry['answer_labels'] = spans
            entry.pop('extractions', None)
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