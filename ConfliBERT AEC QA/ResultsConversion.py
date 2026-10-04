import json
import csv
import os

script_dir = os.path.dirname(os.path.abspath(__file__))
# CSV column header -> JSON field name.
column_names = {'context': 'source_article'}
# Change/Add Extra Question Headers here.
column_names_by_file = {
    'ConfliBERT_results2': {'question': 'question2'},
}

# CSV stores everything as text; these fields are integers in the source JSON.
integer_fields = {'id', 'deaths_side_a', 'deaths_side_b', 'deaths_civilian',
                  'deaths_unknown', 'deaths_low', 'deaths_high'}

# The GUI joins several answers with this separator (see QA_CSV_SEPARATOR).
ANSWER_SEPARATOR = ' | '


def build_answer_labels(record):
    """Replace flat answer/label columns with an answer_labels span list.

    Each span is the answer text, its character offsets in the article, and
    the label. Offsets are missing when the answer text is not in the article.
    """
    if 'answer' not in record and 'label' not in record:
        return record

    answer = record.pop('answer', '') or ''
    label = record.pop('label', '') or ''
    context = record.get('source_article', record.get('context', '')) or ''
    context = str(context)

    texts = [text for text in str(answer).split(ANSWER_SEPARATOR) if text != '']
    if not texts:
        record['answer_labels'] = []
        return record

    label_text = str(label)
    per_span_labels = (
        label_text.split(ANSWER_SEPARATOR)
        if ANSWER_SEPARATOR in label_text else None
    )

    spans = []
    search_from = 0
    for index, text in enumerate(texts):
        start = context.find(text, search_from)
        if start == -1:
            start = context.find(text)
        if start == -1:
            start_value = None
            end = None
        else:
            start_value = start
            end = start + len(text)
            search_from = end

        if per_span_labels is not None and index < len(per_span_labels) and per_span_labels[index] != '':
            labels = [per_span_labels[index]]
        elif label_text != '':
            labels = [label_text]
        else:
            labels = []

        spans.append({
            'end': end,
            'text': text,
            'start': start_value,
            'labels': labels,
        })

    record['answer_labels'] = spans
    return record

# Change FileNames and Number here
names = ['ConfliBERT_results', 'ConfliBERT_results2']

for name in names:
    input_path = os.path.join(script_dir, name + '.csv')
    output_path = os.path.join(script_dir, name + '.json')

    if not os.path.exists(input_path):
        print('skipping ' + name + '.csv (not found)')
        continue

    renames = {**column_names, **column_names_by_file.get(name, {})}

    with open(input_path, 'r', newline='', encoding='utf-8-sig') as csv_file:
        data = []
        for row in csv.DictReader(csv_file):
            record = {}
            for column, value in row.items():
                field = renames.get(column, column)
                if field in integer_fields and value != '':
                    value = int(value)
                record[field] = value
            data.append(build_answer_labels(record))

    with open(output_path, 'w', encoding='utf-8') as json_file:
        json.dump(data, json_file, ensure_ascii=False, indent=2)

    print(name + '.json: ' + str(len(data)) + ' records')
