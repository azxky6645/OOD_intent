import torch
import torch.nn as nn
from transformers import BertModel, BertConfig, BertForSequenceClassification, AutoTokenizer


class BertProxy(nn.Module):
    def __init__(self, config):
        super(BertProxy, self).__init__()
        self.model = BertModel.from_pretrained('bert-base-uncased')
    def forward(self, x):
        bert_output = self.model(**x).last_hidden_state
        mean_pooled_output = bert_output.mean(dim=1)
        return mean_pooled_output

