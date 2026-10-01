import json
import csv
import os

# JSON field name -> CSV column header. The JSON files are left unchanged.
column_names = {'source_article': 'context'}
# Change/Add Extra Question Headers here.
column_names_by_file = {
    'train2': {'question2': 'question'},
}

script_dir = os.path.dirname(os.path.abspath(__file__))

# Change FileNames and Number here
names = ['train', 'train2']

for name in names:
    input_path = os.path.join(script_dir, name + '.json')
    output_path = os.path.join(script_dir, name + '.csv')

    with open(input_path, 'r', encoding='utf-8') as json_file:
        data = json.load(json_file)

    fieldnames = list(data[0].keys())
    renames = {**column_names, **column_names_by_file.get(name, {})}
    header = [renames.get(field, field) for field in fieldnames]

    with open(output_path, 'w', newline='', encoding='utf-8-sig') as csv_file:
        csv.writer(csv_file).writerow(header)
        csv.DictWriter(csv_file, fieldnames=fieldnames).writerows(data)
