import argparse
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import Adam
from transformers import AutoModel, AutoTokenizer, AdamW
from dataloader import read_data


class Pooler(nn.Module):
    """
    Parameter-free poolers to get the sentence embedding
    'cls': [CLS] representation with BERT/RoBERTa's MLP pooler.
    'cls_before_pooler': [CLS] representation without the original MLP pooler.
    'avg': average of the last layers' hidden states at each token.
    'avg_top2': average of the last two layers.
    'avg_first_last': average of the first and the last layers.
    """
    def __init__(self, pooler_type):
        super().__init__()
        self.pooler_type = pooler_type
        assert self.pooler_type in ["cls", "cls_before_pooler", "avg", "avg_top2", "avg_first_last"], "unrecognized pooling type %s" % self.pooler_type

    def forward(self, attention_mask, outputs):
        last_hidden = outputs.last_hidden_state
        pooler_output = outputs.pooler_output
        hidden_states = outputs.hidden_states

        if self.pooler_type in ['cls_before_pooler', 'cls']:
            return last_hidden[:, 0]
        elif self.pooler_type == "avg":
            return ((last_hidden * attention_mask.unsqueeze(-1)).sum(1) / attention_mask.sum(-1).unsqueeze(-1))
        elif self.pooler_type == "avg_first_last":
            first_hidden = hidden_states[0]
            last_hidden = hidden_states[-1]
            pooled_result = ((first_hidden + last_hidden) / 2.0 * attention_mask.unsqueeze(-1)).sum(1) / attention_mask.sum(-1).unsqueeze(-1)
            return pooled_result
        elif self.pooler_type == "avg_top2":
            second_last_hidden = hidden_states[-2]
            last_hidden = hidden_states[-1]
            pooled_result = ((last_hidden + second_last_hidden) / 2.0 * attention_mask.unsqueeze(-1)).sum(1) / attention_mask.sum(-1).unsqueeze(-1)
            return pooled_result
        else:
            raise NotImplementedError

class MLPLayer(nn.Module):
    """
    Head for getting sentence representations over RoBERTa/BERT's CLS representation.
    """

    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = nn.Tanh()

    def forward(self, features, **kwargs):
        x = self.dense(features)
        x = self.activation(x)

        return x

class OutputLayer(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, 1)

    def forward(self, features, **kwargs):
        logits = self.dense(features)

        return logits

def init_model(cls, args):
    """
    init function.
    """
    cls.pooler_type = args.pooler_type
    cls.pooler = Pooler(args.pooler_type)
    if cls.model_args.pooler_type == "cls":
        cls.mlp = MLPLayer(args)
    cls.output = OutputLayer(args)
    cls.init_weights()


def train(model, dataloader):




def main():
    parser = argparse.ArgumentParser("Transformers Classifier")
    parser.add_argument("--do_train", action="store_true")
    #  parser.add_argument("--do_predict", action="store_true")
    #  parser.add_argument("--do_evaluation", action="store_true")
    parser.add_argument(
        "--model",
        type=str,
        default="bert-base-cased")
    parser.add_argument("--trainset", type=str, default="processed_data/english/legal/train_single.tsv")
    parser.add_argument("--validset", type=str, default="processed_data/english/legal/dev_single.tsv")
    parser.add_argument("--load_checkpoint", action="store_true")
    parser.add_argument("--checkpoint", type=str, default="./checkpoints/bertad_1")
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--total_epochs", type=int, default=20)
    parser.add_argument("--eval_checkpoint", type=int, default=7)
    parser.add_argument("--print_frequency", type=int, default=-1)
    parser.add_argument("--warm_up_steps", type=int, default=6000)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--learning_rate", type=float, default=3e-5)
    parser.add_argument("--warm_up_learning_rate", type=float, default=3e-5)
    parser.add_argument("--device", type=int, default=-1)

    parser.add_argument("--save_model_path",
                        type=str,
                        default="checkpoints/bertad")
    parser.add_argument("--save_result_path",
                        type=str,
                        default="test_result.tsv")
    parser.add_argument("--dataset_type",
                        type=str,
                        default='english')

    args = parser.parse_args()
    device = torch.device(args.device) if torch.cuda.is_available() and args.device != -1 else torch.device('cpu')

    model = AutoModel.from_pretrained(args.model)
    tokenizer = AutoTokenizer.from_pretrained(args.model)

    train_sent, train_abbr, train_long_term, train_label = read_data(args.trainset)
    valid_sent, valid_abbr, valid_long_term, valid_label = read_data(args.validset)

    tokenizer.add_tokens(train_abbr)
    model.resize_token_embeddings(len(tokenizer))

    init_model(model, args)
    model = model.to(device)

    train_encoded_inputs = tokenizer(train_sent, train_long_term, padding=True, truncation=True, return_tensors="pt")
    valid_encoded_inputs = tokenizer(valid_sent, valid_long_term, padding=True, truncation=True, return_tensors="pt")

    train_dataset = CrossInteractionDataset(train_encoded_inputs, train_label, device)
    valid_dataset = CrossInteractionDataset(valid_encoded_inputs, valid_label, device)

    train_loader = DataLoader(train_dataset,
                              batch_size=args.batch_size,
                              shuffle=True,
                              pin_memory=True,
                              num_workers=4)
    valid_loader = DataLoader(val_dataset,
                            batch_size=args.batch_size,
                            shuffle=True,
                            pin_memory=True,
                            num_workers=4)

    #  optimizer_warmup = AdamW(model.parameters(), lr=args.warm_up_learning_rate)
    optimizer = AdamW(model.parameters(), lr=args.learning_rate)
    criterion = nn.BCEWithLogitsLoss()

    step = 0
    start_epoch = int(args.checkpoint.split("_")[-1]) if args.load_checkpoint else 0

    for epoch in range(start_epoch, args.total_epochs):
        print('\nTRAINING EPOCH %d' % epoch)

        for idx, batch in enumerate(train_loader):
            step += 1
            #  optim_warmup.zero_grad()
            optim.zero_grad()

            outputs = model(**batch['data'])
            pooler_output = model.pooler(batch['data']['attention_mask'], outputs)
            # If using "cls", we add an extra MLP layer
            # (same as BERT's original implementation) over the representation.
            if cls.pooler_type == "cls":
                pooler_output = model.mlp(pooler_output)

            logits = model.output(pooler_output)
            loss = criterion(logits, batch['label'])


            loss_prt = loss.cpu().detach().numpy() if CUDA_AVAILABLE else loss.detach().numpy()
            loss_prt = round(float(loss_prt), 3)

            loss.backward()
            optimizer.step()
            print(f"Train step {step}\tLoss: {loss_prt}")

            if step % args.print_frequency == 0 and not step <= args.warm_up_steps and not args.print_frequency == -1:
                print('Evaluating...')
                model.eval()
                for idx, batch in enumerate(valid_loader):
                    outputs = model(**batch['data'])
                    pooler_output = model.pooler(batch['data']['attention_mask'], outputs)
                    if cls.pooler_type == "cls":
                        pooler_output = model.mlp(pooler_output)

                    logits = model.output(pooler_output)
                    loss = criterion(logits, batch['label'])
                    loss_prt = loss.cpu().detach().numpy() if CUDA_AVAILABLE else loss.detach().numpy()
                    loss_prt = round(float(loss_prt), 3)
                    print(f"Valid Loss: {loss_prt}")

            print(f'Saving model at epoch {epoch} step {step}')
            model.save_pretrained(f"{args.save_model_path}_%d" % epoch)



            #  if step <= args.warm_up_steps:
            #      if step % 500 == 0:
            #          print(f"warm up step {step}\tLoss: {loss_prt}")
            #      loss.backward()
            #      optim_warmup.step()
            #  else:
            #      if step % 500 == 0:
            #          print(f"train step {step}\tL_nll_d1: {loss_prt}, L_nll_d2: {loss_2_prt} and L_ul: {ul_loss_prt}")
            #      (loss + 0.01 * loss_2 + 0.01 * ul_loss).backward()
            #      optim.step()


if __name__ == "__main__":
    CUDA_AVAILABLE = False
    if torch.cuda.is_available():
        CUDA_AVAILABLE = True
        print("CUDA IS AVAILABLE")
    else:
        print("CUDA NOT AVAILABLE")
