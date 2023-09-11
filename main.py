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

from model import BertProxy
import dataset
from dataset import read_file
import train
from losses import Proxy_Anchor
from losses_origin import Proxy_Anchor_org



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
    p.add_argument('--wan', type=int, default=0)
    p.add_argument('--name', type=str, default=datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))

    p.add_argument('--data_dir', type=str, default='./data')
    p.add_argument('--data', type=str, default='BANKING')
    p.add_argument('--model_dir', type=str, default='D:/intent_proxy', help="save model path")

    p.add_argument('--known_cls_ratio', type=float, default=0.5)
    p.add_argument('--lr', type=float, default=2e-5)
    p.add_argument('--clips', type=float, default=0.6, help="clip grad norm")
    p.add_argument('--batch_size', type=int, default=150)
    p.add_argument('--hidden_size', type=int, default=768)
    p.add_argument('--n_epochs', type=int, default=50)
    p.add_argument('--dropout', type=float, default=.3)
    p.add_argument('--max_seq_len', type=int, default=50)
    p.add_argument('--num_workers', type=int, default=0)
    p.add_argument('--warm', type=int, default=1)
    p.add_argument('--proxy_lr_scale', type=int, default=100)
    p.add_argument('--margin_lr_scale', type=int, default=10)
    p.add_argument('--alpha', type=int, default=64)
    p.add_argument('--weight_decay', type=float, default=1e-3)

    p.add_argument('--valid', type=bool, default=True)
    p.add_argument('--parallel', type=bool, default=False)
    p.add_argument('--seed', type=int, default=-1, help="set seed num")
    p.add_argument('--gpu_id', type=int, default=0 if torch.cuda.is_available() else -1)
    p.add_argument('--flag', action='store_true')
    p.add_argument('--old', action='store_true')
    p.add_argument('--select', action='store_true')

    c = p.parse_args()
    return c


def main(config):
    if config.wan >= 1:
        wandb.init(project=config.data,
                   entity="azxky6645",
                   name=config.name + ', ' + str(config.known_cls_ratio),
                   config=config.__dict__)
        wandb.config = {
            "learning_rate": config.lr,
            "epochs": config.n_epochs,
            "batch_size": config.batch_size,
            "drop_out": config.dropout
        }

    device = torch.device('cpu') if config.gpu_id < 0 else torch.device(f'cuda:{config.gpu_id}')
    # logging.info(f"Working GPU: {device}")
    print(f"Working GPU: {device}")

    model = BertProxy(config=config).to(device)
    tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')

    all_label_list = read_file(os.path.join(config.data_dir, config.data, 'intent_label.txt'))
    n_known_cls = round(len(all_label_list) * config.known_cls_ratio)

    if config.select == True:
        known_label_list = list(np.array(list(all_label_list))[:n_known_cls])
    else:
        known_label_list = list(np.random.choice(np.array(list(all_label_list)),
                                                 n_known_cls,
                                                 replace=False))

    data_train = dataset.CustomDataset(data_path=os.path.join(config.data_dir, config.data),
                                       tokenizer=tokenizer,
                                       known_label_list=known_label_list,
                                       max_seq_len=config.max_seq_len,
                                       mode='train')

    data_valid = dataset.CustomDataset(data_path=os.path.join(config.data_dir, config.data),
                                       tokenizer=tokenizer,
                                       known_label_list=known_label_list,
                                       max_seq_len=config.max_seq_len,
                                       mode='valid')

    data_test = dataset.CustomDataset(data_path=os.path.join(config.data_dir, config.data),
                                      tokenizer=tokenizer,
                                      known_label_list=known_label_list,
                                      max_seq_len=config.max_seq_len,
                                      mode='test')

    # 모델 병렬 처리
    if config.parallel:
        model = DistributedDataParallel(model, device_ids=config.gpu_list)
        data_train_sampler = DistributedSampler(data_train)
        data_test_sampler = DistributedSampler(data_test)
        train_dataloader = DataLoader(data_train,
                                      batch_size=config.batch_size,
                                      num_workers=config.num_workers,
                                      pin_memory=True,
                                      sampler=data_train_sampler)
        valid_dataloader = DataLoader(data_valid,
                                      batch_size=config.batch_size,
                                      num_workers=config.num_workers,
                                      pin_memory=True,
                                      sampler=data_test_sampler)
        test_dataloader = DataLoader(data_test,
                                     batch_size=config.batch_size,
                                     num_workers=config.num_workers,
                                     pin_memory=True,
                                     sampler=data_test_sampler)

    else:
        train_dataloader = DataLoader(data_train, batch_size=config.batch_size,
                                      num_workers=config.num_workers, shuffle=True)
        valid_dataloader = DataLoader(data_valid, batch_size=config.batch_size,
                                      num_workers=config.num_workers, shuffle=False)
        test_dataloader = DataLoader(data_test, batch_size=config.batch_size,
                                     num_workers=config.num_workers, shuffle=False)

    # scheduler setting
    # 한 에포크 마다 learning rate가 변함
    total = config.n_epochs * len(train_dataloader)
    warmup_rate = 0.1
    label_output = None
    if config.old == True:
        loss_function = Proxy_Anchor_org(label_output, len(data_train.label_list), config.hidden_size, alpha=config.alpha).cuda()
    else:
        loss_function = Proxy_Anchor(label_output, len(data_train.label_list), config.hidden_size, alpha=config.alpha).cuda()

    param_groups = [{'params': model.parameters(), 'lr': float(config.lr) * 1},
                    {'params': loss_function.proxies, 'lr': float(config.lr) * config.proxy_lr_scale},
                    {'params': loss_function.mrg, 'lr': float(config.lr) * config.margin_lr_scale}]

    optimizer = optim.AdamW(param_groups, lr=float(config.lr), weight_decay=config.weight_decay)
    scheduler = transformers.get_linear_schedule_with_warmup(optimizer, int(total * warmup_rate), total)

    trainer = train.Trainer(model=model,
                            optimizer=optimizer,
                            loss_function=loss_function,
                            scheduler=scheduler,
                            config=config)
    trainer.fit(train_dataloader=train_dataloader,
                valid_dataloader=valid_dataloader,
                test_dataloader=test_dataloader,
                device=device)


if __name__ == "__main__":
    config = define_args()
    if config.seed >= 0:
        seed_everything(config.seed)

    main(config=config)


