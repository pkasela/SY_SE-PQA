cfg = {
    'datafolder': '../',
    'model_name': 'distilbert-base-uncased',
    'tokenizer_name': 'distilbert-base-uncased',
    'max_tokens': 512,
    'normalize': True,
    'pooling_model': 'mean',
    'device': 'cuda',
    'batch_size': 64,
    'seed': 42,
    # 'training_mode': 'synt_context',
    'margin': 1,
    'lr': 1e-6,
    'max_epoch': 5,
    'se_data_folder' :'../../multidomain/SE-PQA/',
    'embedding_size': 768
}


import json
import numpy as np
import logging
import os
import random
from indxr import Indxr
import click
import torch
import tqdm
from model.model import BiEncoder
from ranx import Qrels, Run, compare, fuse, optimize_fusion
import subprocess

logger = logging.getLogger(__name__)

def get_bert_rank(data, model, doc_embedding, bm25_runs, id_to_index, k=100):
    test_qrels = {}
    bert_run = {}
    index_to_id = {ind: _id for _id, ind in id_to_index.items()}
    for d in tqdm.tqdm(data, total=len(data)):
        q_text = d['text']
        with torch.no_grad():
            q_embedding = model.query_encoder(q_text)#.cpu()
        d_qrels = {k: 1 for k in d['rel_ids']}
        test_qrels[d['id']] = d_qrels
        
        bm25_docs = list(bm25_runs[d['id']].keys())
        d_embeddings = doc_embedding[torch.tensor([int(id_to_index[x]) for x in bm25_docs])]
        bert_scores = torch.einsum('xy, ly -> x', d_embeddings, q_embedding)

        bert_run[d['id']] = {doc_id: bert_scores[i].item() for i, doc_id in enumerate(bm25_docs)}
        
    return test_qrels, bert_run

def seed_everything(seed: int):
    logging.info(f'Setting global random seed to {seed}')
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

def read_jsonl(path, verbose=True):
    with open(path) as f:
        data = [json.loads(line) for line in f]
    return data

def file_len(fname):
    p = subprocess.Popen(['wc', '-l', fname], stdout=subprocess.PIPE, 
                                              stderr=subprocess.PIPE)
    result, err = p.communicate()
    if p.returncode != 0:
        raise IOError(err)
    return int(result.strip().split()[0])

@click.command()
@click.option(
    "--training_mode",
    type=str,
    required=True
)
def main(training_mode):
    seed_everything(cfg['seed'])
    print(training_mode)

    model = BiEncoder(
        model_name=cfg['model_name'],
        tokenizer_name=cfg['tokenizer_name'],
        max_tokens=cfg['max_tokens'],
        normalize=cfg['normalize'],
        pooling_mode=cfg['pooling_model'],
        device=cfg['device'],
    )
    model.load(f"{training_mode}_epoch_19.pt")


    embedding_matrix = torch.load(os.path.join(f'collection_embedding_{training_mode}_epoch_19.pt')).to(cfg['device'])

    with open(os.path.join(f'id_to_index_{training_mode}.json'), 'r') as f:
        id_to_index = json.load(f)

    se_pqa_datafolder = cfg['se_data_folder']

    split = 'val'
    val_queries = read_jsonl(os.path.join(se_pqa_datafolder, split, 'data_pers.jsonl'))

    bm25_filename = os.path.join(se_pqa_datafolder, split, 'bm25_run.json')
    with open(bm25_filename, 'r') as f:
        bm25_run = json.load(f)


    test_qrels, bert_run = get_bert_rank(val_queries, model, embedding_matrix, bm25_run, id_to_index) 
    qrels = Qrels(test_qrels)

    bm25_run = {d: bm25_run[d] for d in test_qrels.keys()}

    ranx_bert_run = Run(bert_run, name='BERT')
    ranx_bm25_run = Run(bm25_run, name='BM25')


    bm25_bert_best_params = optimize_fusion(
        qrels=qrels,
        runs=[ranx_bm25_run, ranx_bert_run],
        norm="min-max",
        method="wsum",
        metric="ndcg@10",  # The metric to maximize during optimization
        return_optimization_report=True
    )
    print(bm25_bert_best_params[0])

    split = 'test'
    test_queries = read_jsonl(os.path.join(se_pqa_datafolder, split, 'data_pers.jsonl'))

    bm25_filename = os.path.join(se_pqa_datafolder, split, 'bm25_run.json')
    with open(bm25_filename, 'r') as f:
        test_bm25_run = json.load(f)


    test_qrels, bert_run = get_bert_rank(test_queries, model, embedding_matrix, test_bm25_run, id_to_index) 
    test_qrels = Qrels(test_qrels)

    test_bm25_run = {d: test_bm25_run[d] for d in test_qrels.keys()}

    ranx_bert_run = Run(bert_run, name='BERT')
    ranx_bm25_run = Run(test_bm25_run, name='BM25')

    ranx_bm25_bert_run = fuse(
        [ranx_bm25_run, ranx_bert_run],
        norm="min-max",
        method="wsum",
        params=bm25_bert_best_params[0],
    )
    ranx_bm25_bert_run.name = 'BM25 + BERT'

    models = [ranx_bert_run, ranx_bm25_run, ranx_bm25_bert_run]
    report = compare(
            qrels=test_qrels,
            runs=models,
            metrics=['precision@1', 'ndcg@3', 'ndcg@10', 'recall@100', 'map@100', 'map@100'],
            max_p=0.01  # P-value threshold
        )

    print(report)

if __name__ == '__main__':
    main()