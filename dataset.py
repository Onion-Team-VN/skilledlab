import torch 
from torch.utils.data import Dataset
import pandas as pd 


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