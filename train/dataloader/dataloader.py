import random

from indxr import Indxr
from torch.utils.data import Dataset
from os.path import join

class SyntheticDataset(Dataset):
    def __init__(self, data_dir: str, filename: str, bm25_run: dict, collection: dict):
        super(SyntheticDataset, self).__init__()  
        self.data_dir = data_dir

        self.queries = Indxr(join(self.data_dir, filename), key_id='id')
        self.bm25_run = bm25_run
        self.collection = collection


    def __getitem__(self, idx):
        query = self.queries[idx]
        query_text = query['title'] + ' ' + query['body']
        bm25_docs = list(self.bm25_run[query['id']].keys())

        real_answer = query['best_answer']
        synt_context_answer = query['answer_WBC']
        synt_body_answer = query['answer_WBB']
        
        neg_doc = random.sample(bm25_docs, k=1)[0]
        neg_text = self.collection[neg_doc]
        
        return {
            'query': query_text, 
            'real_answer': real_answer, 
            'synt_context_answer': synt_context_answer,
            'synt_body_answer': synt_body_answer,
            'neg_text': neg_text,
        }

    def __len__(self):
        return len(self.queries)
    

class PersSyntheticDataset(Dataset):
    def __init__(self, data_dir: str, filename: str, bm25_run: dict, collection: dict):
        super(PersSyntheticDataset, self).__init__()  
        self.data_dir = data_dir

        self.queries = Indxr(join(self.data_dir, filename), key_id='id')
        self.bm25_run = bm25_run
        self.collection = collection


    def __getitem__(self, idx):
        query = self.queries[idx]
        query_text = query['title'] + ' ' + query['body']
        bm25_docs = list(self.bm25_run[query['id']].keys())

        real_answer = query['best_answer']
        synt_title_pers = query['answer_OTP']
        synt_body_pers = query['answer_WBP']
        
        neg_doc = random.sample(bm25_docs, k=1)[0]
        neg_text = self.collection[neg_doc]
        
        return {
            'query': query_text, 
            'real_answer': real_answer, 
            'synt_title_pers': synt_title_pers,
            'synt_body_pers': synt_body_pers,
            'neg_text': neg_text,
        }

    def __len__(self):
        return len(self.queries)