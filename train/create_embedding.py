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
    'margin': 0.5,
    'lr': 1e-5,
    'max_epoch': 5,
    'se_data_folder' :'../../multidomain/SE-PQA/',
    'embedding_size': 768
}


import logging
import os
import random
import click
import numpy as np
import torch
import json


from indxr import Indxr
from dataloader.dataloader import SyntheticDataset
from model.model import BiEncoder
from model.loss import MarginLoss
from torch.utils.data import DataLoader, random_split
from torch.optim import AdamW
from tqdm import tqdm
logger = logging.getLogger(__name__)


def seed_everything(seed: int):
    logging.info(f'Setting global random seed to {seed}')
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


@click.command()
@click.option(
    "--training_mode",
    type=str,
    required=True
)
def main(training_mode):
    seed_everything(cfg['seed'])
    print(training_mode)

    se_pqa_datafolder = cfg['se_data_folder']
    with open(os.path.join(se_pqa_datafolder, 'answer_collection.json'), 'r') as f:
        corpus = json.load(f)


    model = BiEncoder(
        model_name=cfg['model_name'],
        tokenizer_name=cfg['tokenizer_name'],
        max_tokens=cfg['max_tokens'],
        normalize=cfg['normalize'],
        pooling_mode=cfg['pooling_model'],
        device=cfg['device'],
    )
    model.load(f"{training_mode}_epoch_9.pt")
    embedding_matrix = torch.zeros(len(corpus), cfg['embedding_size']).float()


    index = 0
    batch_val = 0
    texts = []
    id_to_index = {}
    for id_, val in tqdm(corpus.items()):
        id_to_index[id_] = index
        batch_val += 1
        index += 1
        if type(val) != 'str':
            val = str(val)
        texts.append(val)
        if batch_val == cfg['batch_size']:
            with torch.no_grad():
                embedding_matrix[index - batch_val : index] = model.doc_encoder(texts).cpu()
            batch_val = 0
            texts = []

    if texts:
        embedding_matrix[index - batch_val : index, :] = model.doc_encoder(texts).cpu()

    torch.save(embedding_matrix, os.path.join(f'collection_embedding_{training_mode}_epoch_9.pt'))

    with open(os.path.join(f'id_to_index_{training_mode}.json'), 'w') as f:
        json.dump(id_to_index, f)

if __name__ == '__main__':
    main()