from matplotlib.pyplot import axis
import torch
import pandas as pd 
import torch.nn as nn 
from transformers import DistilBertModel
from transformers import DistilBertTokenizer

from src.CONST import NUM_LABELS
from src.config import DISTIL_BERT_MODEL, MAX_TOKENS, EMBEDDING_SIZE

class unimodal_dBERT_Model(nn.Module):

    def __init__(self) -> None:
        super().__init__()
        self.distil_BERT = DistilBertModel.from_pretrained(DISTIL_BERT_MODEL)
        for param in self.distil_BERT.parameters():
            param.requires_grad = False 
        self.l1 = nn.Sequential(
            nn.Linear(EMBEDDING_SIZE,NUM_LABELS)
        )
        self.sig = nn.Sigmoid()

    def l2_norm(self, r_factor):
        weights = torch.cat([x.view(-1) for x in self.parameters() if x.requires_grad])
        return r_factor*torch.norm(weights, 2)

    def forward(self, id, mask):
        pooled_output = self.distil_BERT(input_ids=id, attention_mask=mask)
        x = pooled_output.last_hidden_state
        x = x[:, 0, :]
        x = self.l1(x)
        x = self.sig(x)
        pred = torch.round(x)
        return x, pred

class unimodal_dBERT_Dataset(torch.utils.data.Dataset):

    tokenizer_params = { 'padding':'max_length', 
                         'max_length':MAX_TOKENS, 
                         'truncation':True,
                         'return_tensors':"pt"
                        }
    def __init__(self, data):
        
        tokenizer = DistilBertTokenizer.from_pretrained(DISTIL_BERT_MODEL)
        self.tokenized = [tokenizer( data_pt['text'], **self.tokenizer_params) for data_pt in data]
        self.labels = [data_pt['encoded_labels'] for data_pt in data]
        self.length = len(self.tokenized)

    def __len__(self):        
        return self.length

    def __getitem__(self, idx):
        return self.tokenized[idx],  self.labels[idx]

class unimodal_dBERT_Input_transformer:

    def __init__(self) -> None:
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def transform(self,x,y):
        id = x['input_ids'].squeeze(1).to(self.device)
        mask = x['attention_mask'].to(self.device)
        y = torch.as_tensor(y).float().to(self.device)
        return (id,mask), y