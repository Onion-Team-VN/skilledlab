
import torch

def mask_label_padding(labels,tokenizer):
    MASK_ID = -100
    labels[labels==tokenizer.pad_token_id] = MASK_ID
    return labels

def save_model(path, epoch, model_state_dict, optimizer_state_dict, loss):
    torch.save({
            'epoch': epoch,
            'model_state_dict': model_state_dict,
            'optimizer_state_dict': optimizer_state_dict,
            'best_loss': loss,
            }, path)

def load_model(path):
    return torch.load(path)

def print_line():
    LINE_WIDTH = 60
    print('-' * LINE_WIDTH)