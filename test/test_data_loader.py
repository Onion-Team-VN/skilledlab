import torch 
import pandas as pd 
from torch.utils.data import DataLoader, Dataset
from transformers import T5Tokenizer
import os 

import os, sys
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

BATCH_SIZE = 4
PRETRAINED_MODEL = 't5-base'
SEQ_LENGTH = 512
DIR = "../data/"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class QGDataset(Dataset):
    def __init__(self, csv,tokenizer,seq_length,device):
        self.df = pd.read_csv(csv, engine='python')
        self.tokenizer = tokenizer
        self.seq_length = seq_length
        self.device = device

    def __len__(self):
         return len(self.df)

    def __getitem__(self, idx):   
        if torch.is_tensor(idx):
            idx = idx.tolist()
        row = self.df.iloc[idx, 1:]       

        encoded_text = self.tokenizer(
            row['text'], 
            padding='max_length',  
            max_length=self.seq_length,
            truncation=True,
            return_tensors="pt"
        )
        encoded_text['input_ids'] = torch.squeeze(encoded_text['input_ids'])
        encoded_text['attention_mask'] = torch.squeeze(encoded_text['attention_mask'])

        encoded_question = self.tokenizer(
            row['question'],
            padding='max_length', 
            max_length=self.seq_length,
            truncation=True,
            return_tensors='pt'
        )
        encoded_question['input_ids'] = torch.squeeze(encoded_question['input_ids'])

        return (encoded_text.to(self.device), encoded_question.to(self.device))

if __name__ == '__main__':
    tokenizer = T5Tokenizer.from_pretrained(PRETRAINED_MODEL)
    tokenizer.add_special_tokens(
        {'additional_special_tokens': ['<answer>', '<context>']}
    )
    train_set = QGDataset(os.path.join(DIR, 'qg_train.csv'),tokenizer,SEQ_LENGTH,device)
    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
    for batch_index, batch in enumerate(train_loader):
            data, target = batch
            print(data['input_ids'].size())
            print(target.keys())
            break