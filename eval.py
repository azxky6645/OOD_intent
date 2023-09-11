import random
import os
import argparse
from datetime import datetime

import pandas as pd
import numpy as np
import wandb
import transformers
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data.distributed import DistributedSampler
from transformers import BertConfig, BertModel, BertTokenizer
from transformers import AutoTokenizer

from model import BertProxy, BertCross
import dataset
import train, train
from dataset import LABEL_DICT, DATASET
from losses import Proxy_Anchor


def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def define_args():
    p = argparse.ArgumentParser()
    p.add_argument('--wan', type=int, default=1)
    p.add_argument('--name', type=str, default=datetime.now().strftime('%Y-%m-%d %H-%M-%S'))

    p.add_argument('--data_dir', type=str, default='./data')
    p.add_argument('--loss', type=str, default='proxy_anchor')
    p.add_argument('--data', type=str, default='MELD')
    p.add_argument('--model_dir', type=str, default='D:/ICAAI2/Emory', help="load model path")
    p.add_argument('--model_file', type=str, default='2022-08-15 12-40-11.pt', help="load model path")
    p.add_argument('--gpu_id', type=int, default=0 if torch.cuda.is_available() else -1)
    p.add_argument('--eval_method', type=str, default='pred_proxy_100nn')
    # p.add_argument('--proxy_random', type=bool, default=False)
    p.add_argument('--proxy_random', action='store_true')

    p.add_argument('--lr', type=float, default=2e-5)
    p.add_argument('--clips', type=float, default=0.8, help="clip grad norm")
    p.add_argument('--batch_size', type=int, default=64)
    p.add_argument('--hidden_size', type=int, default=768)
    p.add_argument('--n_epochs', type=int, default=20)
    p.add_argument('--n_classes', type=int, default=7)
    p.add_argument('--dropout', type=float, default=.3)
    p.add_argument('--max_seq_len', type=int, default=100)
    p.add_argument('--num_workers', type=int, default=0)
    p.add_argument('--warm', type=int, default=1)
    p.add_argument('--proxy_lr_scale', type=int, default=1000)
    p.add_argument('--alpha', type=int, default=32)
    p.add_argument('--weight_decay', type=float, default=1e-3)

    p.add_argument('--valid', type=bool, default=True)
    p.add_argument('--parallel', type=bool, default=False)
    p.add_argument('--seed', type=int, default=-1, help="set seed num")

    c = p.parse_args()
    return c


def main(config):
    if config.wan >= 1:
        wandb.init(project="bert_proxy_Final_emory", entity="azxky6645", name=config.model_file, config=config.__dict__)
        wandb.config = {
            "learning_rate": config.lr,
            "epochs": config.n_epochs,
            "batch_size": config.batch_size,
            "drop_out": config.dropout
        }

    device = torch.device('cpu') if config.gpu_id < 0 else torch.device(f'cuda:{config.gpu_id}')
    # logging.info(f"Working GPU: {device}")
    print(f"Working GPU: {device}")

    if config.loss == 'proxy_anchor':
        model = BertProxy(config=config).to(device)

    elif config.loss == 'cross_entropy':
        model = BertCross(config=config).to(device)

    tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
    data_train = dataset.DATASET[config.data](data_path=os.path.join(config.data_dir, config.data + '_train.tsv'),
                                              tokenizer=tokenizer,
                                              max_seq_len=config.max_seq_len)
    data_test = dataset.DATASET[config.data](data_path=os.path.join(config.data_dir, config.data + '_dev.tsv'),
                                             tokenizer=tokenizer,
                                             max_seq_len=config.max_seq_len)

    # data_train = dataset.IterableDataset(data=)
    # data_test = dataset.IterableDataset(data=)

    # 모델 병렬 처리
    if config.parallel:
        model = DistributedDataParallel(model, device_ids=config.gpu_list)
        data_train_sampler = DistributedSampler(data_train)
        train_dataloader = DataLoader(data_train,
                                      batch_size=config.batch_size,
                                      num_workers=config.num_workers,
                                      pin_memory=True,
                                      sampler=data_train_sampler)

        # train_dataloader = DataLoader(data_train, batch_size=config.batch_size, num_workers=2, worker_init_fn=dataset.worker_init_fn, pin_memory=True, sampler=data_test_sampler)
        # test_dataloader = DataLoader(data_test, batch_size=config.batch_size, num_workers=2, worker_init_fn=dataset.worker_init_fn, pin_memory=True, sampler=data_test_sampler)

    else:
        train_dataloader = DataLoader(data_train, batch_size=config.batch_size,
                                      num_workers=config.num_workers, shuffle=False)

        # train_dataloader = DataLoader(data_train, batch_size=config.batch_size, num_workers=2, worker_init_fn=dataset.worker_init_fn)
        # test_dataloader = DataLoader(data_test, batch_size=config.batch_size, num_workers=2, worker_init_fn=dataset.worker_init_fn)

    # scheduler setting
    # 한 에포크 마다 learning rate가 변함
    total = config.n_epochs * len(train_dataloader)
    warmup_rate = 0.1

    if config.loss == 'proxy_anchor':
        if config.proxy_random == False:
            label_list = tokenizer(list(LABEL_DICT[config.data].keys()), padding=True)
            label_list = label_list.convert_to_tensors(prepend_batch_axis=False, tensor_type='pt').to(device)
            label_output = model(label_list)
        else:
            label_output = None
        loss_function = Proxy_Anchor(label_output, config.n_classes, config.hidden_size, alpha=config.alpha).cuda()
        param_groups = [{'params': model.parameters(), 'lr': float(config.lr) * 1},
                        {'params': loss_function.parameters(), 'lr': float(config.lr) * config.proxy_lr_scale}]

    elif config.loss == 'cross_entropy':
        loss_function = nn.CrossEntropyLoss().cuda()
        param_groups = model.parameters()

    optimizer = optim.AdamW(param_groups, lr=float(config.lr), weight_decay=config.weight_decay)
    scheduler = transformers.get_linear_schedule_with_warmup(optimizer, int(total * warmup_rate), total)

    if config.loss == 'proxy_anchor':
        trainer = train1.Trainer(model=model,
                                 optimizer=optimizer,
                                 loss_function=loss_function,
                                 scheduler=scheduler,
                                 config=config)
    elif config.loss == 'cross_entropy':
        trainer = train.Trainer(model=model,
                                optimizer=optimizer,
                                loss_function=loss_function,
                                scheduler=scheduler,
                                config=config)

    data_test = dataset.DATASET[config.data](data_path=os.path.join(config.data_dir, config.data + '_test.tsv'),
                                             tokenizer=tokenizer,
                                             max_seq_len=config.max_seq_len)

    test_dataloader = DataLoader(data_test, batch_size=config.batch_size,
                                 num_workers=config.num_workers, shuffle=False)

    trainer.model.load_state_dict(torch.load(os.path.join(config.model_dir, config.model_file))['model'])
    model.eval()
    loss_function.eval()
    if config.loss == 'proxy_anchor':
        test_acc, test_f1, test_loss, test_re, test_pr = trainer.validation_step(train_dataloader, test_dataloader, 1, device, True)
    elif config.loss == 'cross_entropy':
        test_acc, test_f1, test_loss, test_re, test_pr = trainer.validation_step(test_dataloader, 1, device, True)

    if config.wan >= 1:
        wandb.log({'Best Test accuracy': test_acc,
                   'Best Test f1-score': test_f1,
                   'Best Test loss': test_loss,
                   'Best Test recall': test_re,
                   'Best Test precision': test_pr})


if __name__ == "__main__":
    config = define_args()
    if config.seed >= 0:
        seed_everything(config.seed)

    main(config=config)


