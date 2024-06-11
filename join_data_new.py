import glob
import os
import json
from tqdm import tqdm

datafolder = 'Dataset_gpt_com/Synte_Answers/mini/Dataset_nuovo_phi_base'

all_files = glob.glob(f'{datafolder}/*/*.json')

def write_jsonl(path, data):
    with open(path, 'w') as f:
        for d in data:
            json.dump(d, f)
            f.write('\n')

def load_jsonl(file: str):
    with open(file, 'r') as f:
        for lne in f:
            yield json.loads(lne)

se_pqa_datafolder = '../multidomain/SE-PQA/'
with open(os.path.join(se_pqa_datafolder, 'answer_collection.json'), 'r') as f:
    collection = json.load(f)

train_queries = load_jsonl(os.path.join(se_pqa_datafolder, 'train/data_pers.jsonl'))
q_id_to_best_answer = {}
for query in train_queries:
    q_id_to_best_answer[query['id']] = query['best_answer']

final_jsonl = []
no_bueno = []
which_comm = []
for file in tqdm(all_files):
    with open(file, 'r') as f:
        data = json.load(f)
    try:
        data_json = {
            'id': data['id'],
            'title': data['Title'],
            'body': data['Body'],
            'best_answer': collection[q_id_to_best_answer[data['id']]],
            # 'answer_OTP': data['answer_only_title_pers'][0]['generated_text'],
            # 'answer_WBP': data['answer_with_body_pers'][0]['generated_text'],
            'answer_WBC': data['answer_with_body_context'][0]['generated_text'],
            'answer_WBB': data['answer_with_body_base'][0]['generated_text'],
            
        }
        final_jsonl.append(data_json)
    except KeyError:
        no_bueno.append(data)
        

write_jsonl('data_phi_mini.jsonl', final_jsonl)