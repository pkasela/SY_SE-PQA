cfg = {
    'datafolder': '../',
    'data_file': 'data_phi.jsonl',
    'model_name': 'distilbert-base-uncased',
    'tokenizer_name': 'distilbert-base-uncased',
    'max_tokens': 512,
    'normalize': True,
    'pooling_model': 'mean',
    'device': 'cuda',
    'batch_size': 128,
    'seed': 42,
    # 'training_mode': 'synt_context',
    'margin': .5,
    'lr': 5e-6,
    'max_epoch': 20,
    'se_data_folder' :'../../multidomain/SE-PQA/',
    'embedding_size': 768
}

import json
import logging
import os
import random
import click
import numpy as np
import torch


from dataloader.dataloader import SyntheticDataset, PersSyntheticDataset
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

# train_mode = 'best_answer' # or 'synt_title' or 'synt_body'

def train(train_dataloader, loss_fn, optimizer, model, epoch, train_mode):
    
    losses = []
    accuracies = []

    pbar = tqdm(train_dataloader)
    optimizer.zero_grad()
    for data in pbar:
        if train_mode == 'best_answer':
            pos_doc = data['real_answer']
        
        if train_mode == 'synt_context':
            pos_doc = data['synt_context_answer']
        if train_mode == 'synt_body':
            pos_doc = data['synt_body_answer']

        if train_mode == 'synt_title_pers':
            pos_doc = data['synt_title_pers']
        if train_mode == 'synt_body_pers':
            pos_doc = data['synt_body_pers']
        

        neg_doc = data['neg_text']

        batch = {
            'query': data['query'],
            'pos_doc': pos_doc,
            'neg_doc': neg_doc
        }
        output = model(batch)
        loss_val, accuracy = loss_fn(output)
        loss_val.backward()
        
        optimizer.step()
        optimizer.zero_grad()

        accuracies.extend(accuracy.view(-1).tolist())
                
        losses.append(loss_val.cpu().detach().item())
        average_loss = np.mean(losses)
        average_sim_accuracy = np.mean(accuracies)
        average_sim_accuracy = round(average_sim_accuracy*100,2)
        pbar.set_description("TRAIN EPOCH {:3d} Current loss {:.2e}, Average {:.2e} Sim Accuracy {}".format(epoch, loss_val, average_loss, average_sim_accuracy))

    return average_loss

def validate(val_dataloader, loss_fn, optimizer, model, epoch, train_mode):
    
    losses = []
    accuracies = []

    pbar = tqdm(val_dataloader)
    optimizer.zero_grad()
    for data in pbar:
        # if train_mode == 'best_answer':
        pos_doc = data['real_answer']
        # if train_mode == 'synt_title':
        #     pos_doc = data['synt_title_answer']
        # if train_mode == 'synt_body':
        #     pos_doc = data['synt_title_answer']
        neg_doc = data['neg_text']

        batch = {
            'query': data['query'],
            'pos_doc': pos_doc,
            'neg_doc': neg_doc
        }
        with torch.no_grad():
            output = model(batch)
            loss_val, accuracy = loss_fn(output)
            
        accuracies.extend(accuracy.view(-1).tolist())
        
        losses.append(loss_val.cpu().detach().item())
        average_loss = np.mean(losses)
        average_sim_accuracy = np.mean(accuracies)
        average_sim_accuracy = round(average_sim_accuracy*100,2)
        pbar.set_description("VAL EPOCH {:5d} Current loss {:.2e}, Average {:.2e} Sim Accuracy {}".format(epoch, loss_val, average_loss, average_sim_accuracy))

    return average_loss

@click.command()
@click.option(
    "--training_mode",
    type=str,
    required=True
)
def main(training_mode):
    seed_everything(cfg['seed'])
    print(training_mode)

    with open(f'{cfg["se_data_folder"]}/train/bm25_run.json', 'r') as f:
        bm25_run = json.load(f)
        

    with open(f'{cfg["se_data_folder"]}/answer_collection.json', 'r') as f:
        collection = json.load(f)

    if training_mode in ['synt_context', 'synt_body', 'best_answer']:
        data = SyntheticDataset(cfg['datafolder'], cfg['data_file'], bm25_run, collection)
    if training_mode in ['synt_title_pers', 'synt_body_pers']:
        data = PersSyntheticDataset(cfg['datafolder'], cfg['data_file'], bm25_run, collection)

    # val_size = int(len(data) * .01)
    # train_size = len(data) - val_size


    # train_dataset, val_dataset = random_split(data, [train_size, val_size])

    train_dataloader = DataLoader(
            data, #train_dataset,
            batch_size=cfg['batch_size'],
            shuffle=True
        )

    # val_dataloader = DataLoader(
    #         val_dataset,
    #         batch_size=cfg['batch_size'],
    #         shuffle=True
    #     )


    model = BiEncoder(
        model_name=cfg['model_name'],
        tokenizer_name=cfg['tokenizer_name'],
        max_tokens=cfg['max_tokens'],
        normalize=cfg['normalize'],
        pooling_mode=cfg['pooling_model'],
        device=cfg['device'],
    )
    loss_fn = MarginLoss(cfg['margin']).to(cfg['device'])

    optimizer = AdamW(model.parameters(), lr=cfg['lr'])
    # epoch = 0
    # validate(val_dataloader, loss_fn, optimizer, model, epoch, cfg['training_mode'])

    # best_loss = 999
    for epoch in tqdm(range(cfg['max_epoch'])):
        train_loss = train(train_dataloader, loss_fn, optimizer, model, epoch, training_mode)

        # val_loss = validate(val_dataloader, loss_fn, optimizer, model, epoch, cfg['training_mode'])
        # if val_loss < best_loss:
        #     best_loss = val_loss
        model.save(f"{training_mode}.pt")
        if ((epoch + 1) % 5) == 0:
            model.save(f"{training_mode}_epoch_{epoch}.pt")


if __name__ == '__main__':
    main()