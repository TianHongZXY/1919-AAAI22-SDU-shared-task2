import torch


def read_data(path):
    with open(path, 'r') as f:
        data = f.readlines()

    sent = []
    abbr = []
    long_term = []
    label = []

    for idx, ins in enumerate(data):
        ins = ins.strip()
        ins = ins.split('\t')
        sent.append(ins[0])
        abbr.append(ins[1])
        long_term.append(ins[2])
        label.append(int(ins[3]))
    
    return sent, abbr, long_term, label

def preprocess(args, tokenizer):
    train_sent, train_abbr, train_long_term, train_label = read_data(args.trainset)
    val_sent, val_abbr, val_long_term, val_label = read_data(args.valset)
    
    print("Dataset loaded.")
    
class CrossInteractionDataset(torch.utils.data.Dataset):
    def __init__(self, tokenized_data, label):
        self.input_ids = tokenized_data["input_ids"]
        self.token_type_ids = tokenized_data["token_type_ids"]
        self.attention_mask = tokenized_data["attention_mask"]
        self.label = label

    def __getitem__(self, idx):
        return {"data":
                {"input_ids": self.input_ids[idx], 
                "token_type_ids": self.token_type_ids[idx], 
                "attention_mask": self.attention_mask[idx]
                },
                "label": self.label[idx]
                }
    
    def __len__(self):
        return len(self.label)
