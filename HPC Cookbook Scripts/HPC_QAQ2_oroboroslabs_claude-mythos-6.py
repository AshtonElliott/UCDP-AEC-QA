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
input_path = os.path.join(script_dir, 'train2sample.json')
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
                messages=[
                    {'role': 'system', 'content': (
                    'You are a Political Scientist trying to find what infrastructure was damaged.'
                    'Your task is to extract specific infrastructure-related words from the provided text and classify them.'
                    '\n\n'
                    '<categories>'
                    '1. Energy: infrastructure related to energy exploration, production, and distribution.'
                    '2. Water: infrastructure related to drinking water, purification, irrigation, wastewatertreatment, and sanitation'
                    '3. Transportation/Marketing: infrastructure related to the transportation marketing, andexchange of commodities'
                    '4. Energy/Water: infrastructure related to both producing energy and water (e.g. a dam where hydropower and irrigation functions cannot be separated)'
                    '5. Health: infrastructure related to public health, including but not limited to hospitals, clinics, ambulances'
                    '6. Agriculture/Fishing: infrastructure  related to crop cultivation and harvesting, and infrastructure related to fisheries'
                    '7. Government/Rebel: infrastructure related to Government or Public Based Buildings such as Schools, Admin Buildings, and Military Bases'
                    '8. Other: infrastructure not related to any of the previous categories.'
                    '<categories>'
                    '\n\n'
                    
                    "<rules>\n"
                    "1. Use a <thinking> tag to reason through findings first.\n"
                    "2. Return answers only as a comma-separated list in the format: 'Word | Category'.\n"
                    "</rules>\n\n"
                    
                    '<example>\n'
                    'Input: "The local clinic was damaged while the hydroelectric dam and nearby school were bombed. The airport was used to launch the attacks." \n'
                    'Response: \n'
                    '{\n'
                        "\"thinking\": \"The text mentions a 'clinic' (health), a 'hydroelectric dam' (dual purpose energy/water), and a 'school' (government/public) that were damaged. It also mentions an airport, but it was not damaged so it is not included.\",\n"
                        "\"extractions\": [\n"
                          "{\"word\": \"clinic\", \"category\": \"Health\"},\n"
                          "{\"word\": \"hydroelectric dam\", \"category\": \"Energy/Water\"},\n"
                          "{\"word\": \"school\", \"category\": \"Government/Rebel\"}\n"
                        "]\n"
                    '}\n'
                    '<example>\n\n'
                    )},
                    
                    {'role': 'user', 'content': (
                        f"<article>\n{retrieved_context}\n</article>\n\n"
                        f"<question>\n{question}\n</question>"
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
                text = ""
                QALabel = ""
                if isinstance(label, dict):
                    text = label.get('word', '')
                    QALabel = label.get('category', 'Other')
                
                # Apply Text & Label
                for match in re.finditer(re.escape(text), context, re.IGNORECASE):
                    spans.append({
                        "end": match.end(),
                        "text": context[match.start():match.end()],
                        "start": match.start(),
                        "labels": [QALabel]
                    })
            if len(spans) == 0 or QALabel == "":
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
