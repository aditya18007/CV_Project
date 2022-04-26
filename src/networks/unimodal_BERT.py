import torch
import pandas as pd 
import torch.nn as nn 
from transformers import DistilBertModel
from transformers import DistilBertTokenizer

DISTIL_BERT_MODEL="distilbert-base-uncased"
MAX_TOKENS = 256
EMBEDDING_SIZE = 768

class unimodal_dBERT_Model(nn.Module):

    def __init__(self) -> None:
        super(unimodal_dBERT_Model).__init__()
        self.distil_BERT = DistilBertModel.from_pretrained(DISTIL_BERT_MODEL)
        self.distil_BERT.eval()
    
    def forward(self, id, mask):
        pooled_output = self.distil_BERT(input_ids=id, attention_mask=mask)
        x = pooled_output.last_hidden_state
        print(x.shape)
        return None 

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