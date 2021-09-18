import torch


def read_data(path):
    with open(path, 'r') as f:
        data = f.readlines()

    sent = []
    abbr = []
    long_term = []
    label = []

    for idx, ins in enumerate(data):
        ins = ins.split('\t')
        sent.append(ins[0])
        abbr.append(ins[1])
        long_term.append(ins[2])
        label.append(ins[3])
    
    return sent, abbr, long_term, label

def preprocess(args, tokenizer):
    train_sent, train_abbr, train_long_term, train_label = read_data(args.trainset)
    val_sent, val_abbr, val_long_term, val_label = read_data(args.valset)
    
    print("Dataset loaded.")
    
class CrossInteractionDataset(torch.utils.data.Dataset):
    def __init__(self, tokenized_data, device):
        self.data = tokenized_data 
        self.label = label
        self.device = device

    def __getitem__(self, idx):
        return {'data': self.data[idx].to(self.device), 'label': self.label[idx].to(self.device)}
