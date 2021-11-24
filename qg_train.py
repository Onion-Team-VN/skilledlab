import os
import torch
from torch.utils.data import DataLoader
from transformers import T5Tokenizer, T5ForConditionalGeneration, T5Config
from utils import mask_label_padding, save_model, print_line
from dataset import QGDataset

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# logging.info(('Using device:', device))
print('Using device:', device)


def train_step(model,optimizer, epoch, best_val_loss, tokenizer, data_loader, log_interval, temp_save_path):
    model.train()
    total_loss = 0.
    for batch_index, batch in enumerate(data_loader):
        data, target = batch
        optimizer.zero_grad()
        masked_labels = mask_label_padding(target['input_ids'],tokenizer)
        output = model(
            input_ids=data['input_ids'],
            attention_mask=data['attention_mask'],
            labels=masked_labels
        )
        loss = output[0]
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        optimizer.step()

        total_loss += loss.item()
        if batch_index % log_interval == 0 and batch_index > 0:
            cur_loss = total_loss / log_interval
            print('| epoch {:3d} | ' 
                  '{:5d}/{:5d} batches | '
                  'loss {:5.2f}'.format(
                    epoch, 
                    batch_index, len(data_loader), 
                    cur_loss))
            save_model(
                temp_save_path,
                epoch, 
                model.state_dict(), 
                optimizer.state_dict(), 
                best_val_loss
            )
            total_loss = 0

def evaluate_step(eval_model, data_loader, tokenizer):
    eval_model.eval()
    total_loss = 0.
    with torch.no_grad():
        for batch_index, batch in enumerate(data_loader):
            data, target = batch
            masked_labels = mask_label_padding(target['input_ids'],tokenizer)
            output = eval_model(
                input_ids=data['input_ids'],
                attention_mask=data['attention_mask'],
                labels=masked_labels
            )
            total_loss += output[0].item()
    return total_loss / len(data_loader)

def train(model,tokenizer,train_loader,optimizer,valid_loader,epochs,log_interval,temp_save_path,model_save_path):
    best_val_loss = float("inf")
    val_loss = evaluate_step(model, valid_loader,tokenizer)
    print('| Before training | valid loss {:5.2f}'.format(
        val_loss)
    )
    print_line()    
    for epoch in range(1, epochs + 1):

        train_step(model,optimizer, epoch, best_val_loss, tokenizer, train_loader, log_interval, temp_save_path)
        val_loss = evaluate_step(model, valid_loader,tokenizer)
        print_line()
        print('| end of epoch {:3d} | valid loss {:5.2f}'.format(
            epoch,
            val_loss)
        )
        print_line()

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_model(
                model_save_path,
                epoch, 
                model.state_dict(), 
                optimizer.state_dict(), 
                best_val_loss
            )
            print("| Model saved.")
            print_line()


if __name__ == '__main__':
    PRETRAINED_MODEL = 't5-base'
    DIR = "data/"
    BATCH_SIZE = 4
    SEQ_LENGTH = 512
    LR = 0.001
    EPOCHS = 20
    LOG_INTERVAL = 5000
    SAVED_MODEL_PATH = "trained/qg_pretrained_t5_model_trained.pth"
    TEMP_SAVE_PATH = "tmp_trained/qg_pretrained_t5_model_trained.pth"

    tokenizer = T5Tokenizer.from_pretrained(PRETRAINED_MODEL)
    tokenizer.add_special_tokens(
        {'additional_special_tokens': ['<answer>', '<context>']}
    )
    train_set = QGDataset(os.path.join(DIR, 'qg_train.csv'),tokenizer,SEQ_LENGTH,device)
    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
    valid_set = QGDataset(os.path.join(DIR, 'qg_valid.csv'),tokenizer,SEQ_LENGTH,device) 
    valid_loader = DataLoader(valid_set, batch_size=BATCH_SIZE, shuffle=False)

    config = T5Config(decoder_start_token_id=tokenizer.pad_token_id)
    model = T5ForConditionalGeneration(config).from_pretrained(PRETRAINED_MODEL)
    model.resize_token_embeddings(len(tokenizer)) # to account for new special tokens
    model = model.to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=LR)
    train(model,tokenizer,train_loader,optimizer,valid_loader,EPOCHS,LOG_INTERVAL,TEMP_SAVE_PATH,SAVED_MODEL_PATH)

