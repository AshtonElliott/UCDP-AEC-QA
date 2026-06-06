import json
import ollama
import re


# Change json source file here
with open('C:/Users/atown/OneDrive/Documents/ConfliBERT/AEC Dataset/JSON_Files/train.json', 'r') as f:
    dataset = json.load(f)

# Wrap in a list if the file contains only one object
if isinstance(dataset, dict): dataset = [dataset]

for entry in dataset:
    context = entry.get('source_article', '')
    question = entry.get('question', '')
    
    chunks = [context[i:i+1500] for i in range(0, len(context), 1500)]
    retrieved_context = "\n".join(chunks[:5])
    
    # Change Model Here
    response = ollama.chat(model='qwen3:8b', messages=[
        {'role': 'system', 'content': (
            'Identify a word or string of words that answer the question.'
            'Do NOT include generic terms like "clashes", "attack", "violence", or "military equipment". '
            'Return only a comma-separated list of specific, concrete answers found in the article.'
            )},
            
        # Positive Example From train.json, ID# 215280
        {'role': 'user', 'content': 'Article: Shelling targets the countryside of Deir Ezzor, and airstrikes carried out on al- Qaryatain, while IS targets YPG in al- Raqqa\nHoms Province:\nThe warplanes carried out 3 raids at least on the IS-held city of al- Qaryatian in the southeast of Homs, no information about casualties.\nAl- Raqqa Province:\nIS targeted a YPG vehicle in the east of the town of Sluk in the eastern countryside of al- Raqqa, information reported casualties.\nDeir Ezzor Province:\nThe regime forces shelled places in the village of al- Husayniyya in the west of Deir Ezzor.'},
        {'role': 'assistant', 'content': 'Shelling, airstrikes, shelled'},
        
        # Negative Example from train.json, ID# 269371
        {'role': 'user', 'content': 'Article: Taliban Militants Killed in Faryab Conflict\nTuesday, October 30, 2018\nMaimana (BNA) Three armed militants were killed by security forces in Faryab province the other day.\nHead of Faryab security commandment told BNA, the clash occurred in Qoriash village, Doulatabad District, Faryab province, in which three armed Taliban including a local commander of them were killed and five others were injured.\nThe source added, no harm and casualties sustained to security forces in the conflict.\nT. Yarzada'},
        {'role': 'assistant', 'content': ''},
        
        # Example to Reinenforce not to count "Clashes" from train.json, ID# 213506
        {'role': 'user', 'content': 'Article: Al-Hasakah Province:\nThe warplanes carried out some raids on places in IS-held area of al- Shaddadi and its vicinity in the south of al- Hasakah, no information about victims.\nAleppo Province: The rebel factions launched some shells on places in the regime-held neighborhood of al- A\u2019zamiyyah this morning, no information about casualties.\nIS shelled by mortar shells places in the city of Marea in the north of Aleppo leading to wound some people.\nClashes took place after midnight between IS against the regime forces and allied militiamen around the two villages of Tal Riman and al- Salhiyyah in the eastern countryside of Aleppo, information reported casualties on both sides.'},
        {'role': 'assistant', 'content': 'shells, shelled by mortar shells'},
        
        {'role': 'user', 'content': f"Context: {retrieved_context}\n\nQuestion: {question}"}
    ])

    prediction = response['message']['content']
    labels = prediction.split(',')
    
    spans = []
    for label in labels:
        clean_label = re.sub(r'[^\w\s]', '', label.strip())
        if clean_label:
            for match in re.finditer(re.escape(clean_label), context, re.IGNORECASE):
                spans.append({
                    "end": match.end(),
                    "text":  context[match.start():match.end()],
                    "start":  match.start(),
                    "labels": ["Answer"]
                })
                
    if len(spans) == 0:
        entry['no_answer'] = "No arms or methods mentioned"
    else:
        entry['answer_labels'] = spans
    
    # Print Results to JSON; change JSON File Name and Filepath here
    with open('C:/Users/atown/OneDrive/Documents/ConfliBERT/Local Results/qwen3.8b_Results.json', 'w') as f:
        json.dump(dataset, f, indent=4)

    print("Process complete. Results saved to newly created JSON File.")

    