import os
from copy import deepcopy

from tqdm.auto import tqdm
import torch
import torch.nn.functional as F
import torchmetrics.functional as FNT
import wandb
import csv

from sklearn.metrics import classification_report, confusion_matrix, f1_score, recall_score, precision_score, accuracy_score

from utils import l2_norm, proxy_distance_with_margin, pred_nearest_proxy

class Trainer:
    def __init__(self, model, optimizer, loss_function, scheduler, config):
        super(Trainer, self).__init__()
        self.model = model
        self.optimizer = optimizer
        self.loss_function = loss_function
        self.scheduler = scheduler
        self.config = config
        pass

    def train_step(self, data, epoch, device):
        trn_acc = 0.
        trn_loss = 0.
        trn_f1 = 0.
        count = 0
        target_list = []
        pred_list = []

        with tqdm(total=len(data), desc=f"EPOCH - {epoch} ") as pbar:
            if self.config.flag == True:
                x_file = open(os.path.join('embedding_tsv', (self.config.name + '_x_embedding.tsv')), 'w', newline='\n')
                wr = csv.writer(x_file, delimiter='\t')
                y_file = open(os.path.join('embedding_tsv', (self.config.name + '_y_true.tsv')), 'w', newline='\n')
                wr_y = csv.writer(y_file, delimiter='\t')
            for step, batch in enumerate(data):

                self.model.train()
                self.model.zero_grad()

                x, y = batch

                count += 1
                x = x.to(device)
                y = y.to(device)

                x_emb = self.model(x)
                loss, proxy, margin, p_m_distance, m_pos, m_neg = self.loss_function(x_emb, y)

                #print(proxy)
                self.optimizer.zero_grad()
                loss.backward()

                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.clips)
                torch.nn.utils.clip_grad_norm_(self.loss_function.parameters(), self.config.clips)

                self.optimizer.step()

                y_pred = proxy_distance_with_margin(x_emb, proxy, margin, self.config.old)
                # y_pred = pred_nearest_proxy(x_emb, proxy)

                target_list.append(y.tolist())
                pred_list.append(y_pred.tolist())

                trn_acc += accuracy_score(y.cpu(), y_pred.cpu().detach())
                trn_f1 += f1_score(y.cpu(), y_pred.cpu().detach(), average='macro')

                trn_loss += loss.detach().cpu()

                if self.scheduler:
                    self.scheduler.step()

                learning_rate = self.scheduler.get_last_lr()[0]

                pbar.update(1)
                postfix_str = ""
                postfix_str += f" Trn acc: {(trn_acc / count):1.4f}, "
                postfix_str += f" Trn f1: {(trn_f1 / count):1.4f}, "
                postfix_str += f" Trn loss: {(trn_loss.item() / count):.4f}"

                pbar.set_postfix_str(postfix_str)
                if self.config.flag == True:
                    for x in x_emb:
                        wr.writerow(x.tolist())

                    for y_true in y:
                        wr_y.writerow([y_true.item()])

            pred_list = sum(pred_list, [])
            target_list = sum(target_list, [])

            cm = confusion_matrix(target_list, pred_list, labels=[i for i in range(len(proxy))])
            print('\n' + str(cm))
            print('ACCURACY: ', accuracy_score(pred_list, target_list))
            print('F1-score: ', f1_score(pred_list, target_list, average='macro'))

            if self.config.flag == True:
                x_file.close()
                y_file.close()
        return accuracy_score(pred_list, target_list), f1_score(pred_list, target_list, average='macro'), trn_loss/count, learning_rate, p_m_distance, m_pos, m_neg

    def validation_step(self, data, epoch, device):
        valid_acc = 0.
        valid_f1 = 0.
        valid_loss = 0.
        valid_count = 0
        target_list = []
        pred_list = []
        valid_re = 0.
        valid_pr = 0.

        self.model.eval()
        self.loss_function.eval()

        with torch.no_grad():
            with tqdm(total=len(data), desc=f"EPOCH - {epoch} ") as pbar:
                if self.config.flag == True:
                    x_file = open(os.path.join('embedding_tsv', (self.config.name + '_test_x_embedding.tsv')), 'w', newline='\n')
                    wr = csv.writer(x_file, delimiter='\t')
                    y_file = open(os.path.join('embedding_tsv', (self.config.name + '_test_y_true.tsv')), 'w', newline='\n')
                    wr_y = csv.writer(y_file, delimiter='\t')
                for step, (x, y) in enumerate(data):

                    valid_count += 1

                    x = x.to(device)
                    y = y.to(device)

                    x_emb = self.model(x)
                    loss, proxy, margin, p_m_distance, m_pos, m_neg = self.loss_function(x_emb, y)

                    y_pred = proxy_distance_with_margin(x_emb, proxy, margin, self.config.old)
                    # y_pred = pred_nearest_proxy(x_emb, proxy)

                    target_list.append(y.tolist())
                    pred_list.append(y_pred.tolist())

                    valid_loss += loss.detach().cpu()
                    valid_acc += accuracy_score(y.cpu(), y_pred.cpu().detach())
                    valid_f1 += f1_score(y.cpu(), y_pred.cpu().detach(), average='macro')

                    pbar.update(1)
                    postfix_str = ""
                    postfix_str += f" valid acc: {valid_acc / valid_count:1.4f}, "
                    postfix_str += f" valid f1: {valid_f1 / valid_count:1.4f}, "
                    postfix_str += f" valid loss: {valid_loss.item() / valid_count:.4f}"
                    pbar.set_postfix_str(postfix_str)
                    if self.config.flag == True:
                        for x in x_emb:
                            wr.writerow(x.tolist())

                        for y_true in y:
                            wr_y.writerow([y_true.item()])

                pred_list = sum(pred_list, [])
                target_list = sum(target_list, [])

                cm = confusion_matrix(target_list, pred_list, labels=[i for i in range(len(proxy))])
                print('\n' + str(cm))
                print('ACCURACY: ', accuracy_score(pred_list, target_list))
                print('F1-score: ', f1_score(pred_list, target_list, average='macro'))
                report = classification_report(
                    target_list, pred_list,
                    labels=list(set(target_list)),
                    target_names=list(set(target_list)),
                    output_dict=True,
                    zero_division=1
                )

                IND_sum = 0
                for i in range(len(list(set(target_list)))-1):
                    IND_sum += report[i]['f1-score']
                IND_f1 = IND_sum / (len(list(set(target_list)))-1)
                OOD_f1 = report[len(list(set(target_list)))-1]['f1-score']
                print('IND-F1: ', IND_f1)
                print('OOD-F1: ', OOD_f1)

                if self.config.flag == True:
                    x_file.close()
                    y_file.close()

        return accuracy_score(pred_list, target_list), \
               f1_score(pred_list, target_list, average='macro'), \
               valid_loss/valid_count, IND_f1, OOD_f1

    def fit(self, train_dataloader, valid_dataloader, test_dataloader, device):
        best_model = None
        best_valid_score = 0.
        best_test_acc_score = 0.
        best_test_f1_score = 0.
        best_test_f1_ind_score = 0.
        best_test_f1_ood_score = 0.

        self.model.zero_grad()
        self.optimizer.zero_grad()
        for epoch in range(self.config.n_epochs):
            if self.config.warm > 0:
                if epoch == 0:
                    for param in list(set(self.model.parameters())):
                        param.requires_grad = False
                if epoch == self.config.warm:
                    for param in list(set(self.model.parameters())):
                        param.requires_grad = True

            train_accuracy, train_f1, train_loss, learning_rate, p_m_distance, m_pos, m_neg  = self.train_step(train_dataloader, epoch, device)
            valid_acc, valid_f1, valid_loss, valid_ind_f1, valid_ood_f1 = self.validation_step(valid_dataloader, epoch, device)
            test_acc, test_f1, test_loss, test_ind_f1, test_ood_f1 = self.validation_step(test_dataloader, epoch, device)

            # save model
            if not os.path.exists(os.path.join(self.config.model_dir, self.config.data, str(self.config.known_cls_ratio))):
                os.makedirs(os.path.join(self.config.model_dir, self.config.data, str(self.config.known_cls_ratio)))

            if valid_f1 >= best_valid_score:
                best_model = deepcopy(self.model.state_dict())
                print(f"SAVE! Epoch: {epoch + 1}/{self.config.n_epochs}")
                best_valid_score = valid_f1
                best_test_acc_score = test_acc
                best_test_f1_score = test_f1
                best_test_f1_ind_score = test_ind_f1
                best_test_f1_ood_score = test_ood_f1

                model_name = f"{self.config.name}.pt"
                model_path = os.path.join(self.config.model_dir, self.config.data, str(self.config.known_cls_ratio), model_name)
                torch.save({
                    'model': self.model.state_dict(),
                    'config': self.config,
                    'proxy': self.loss_function.state_dict(),
                    'known_label_list': train_dataloader.dataset.known_label_list
                }, model_path)

            if self.config.wan >= 1:
                wandb.log({
                    "Train loss": train_loss,
                    "Train accuracy": train_accuracy,
                    "Train F1-score": train_f1,
                    "Learning rate": learning_rate,
                    "Validation loss": valid_loss,
                    "Validation accuracy": valid_acc,
                    "Validation All-F1-score": valid_f1,
                    "Validation IND-F1-score": valid_ind_f1,
                    "Validation OOD-F1-score": valid_ood_f1,
                    "Test loss": test_loss,
                    "Test accuracy": test_acc,
                    "Test F1-score": test_f1,
                    "Test IND-F1-score": test_ind_f1,
                    "Test OOD-F1-score": test_ood_f1
                })

            print("END")

        if self.config.wan >= 1:
            wandb.log({
                "best_valid_score": best_valid_score,
                "best_test_acc_score": best_test_acc_score,
                "best_test_f1_score": best_test_f1_score,
                "best_test_f1_ind_score": best_test_f1_ind_score,
                "best_test_f1_ood_score": best_test_f1_ood_score
            })

        self.model.load_state_dict(best_model)
