import json
import torch
import argparse
from tqdm import tqdm
from scipy.spatial.distance import cosine
from transformers import AutoModel, AutoTokenizer

def predict(model, tokenizer, data, diction):
    predictions = []
    for d in tqdm(data):
        pred = {
                'ID': d['ID'],
                'label': ''
                }
        sent = d['sentence']
        candids = diction[d['acronym']]
        highest_score = -1
        best = ''
        inputs = tokenizer(sent, truncation=True, return_tensors="pt").to(args.device)
        with torch.no_grad():
            embeddings = model(**inputs, output_hidden_states=True, return_dict=True).pooler_output
            for candid in candids:
                score = 0
                cand_inputs = tokenizer(candid, truncation=True, return_tensors="pt").to(args.device)
                cand_emb = model(**cand_inputs, output_hidden_states=True, return_dict=True).pooler_output
                score = 1 - cosine(embeddings.cpu(), cand_emb.cpu())
                if score > highest_score:
                    highest_score = score
                    best = candid
            if best == '':
                best = candids[0]
            pred['label'] = best
            predictions.append(pred)
    return predictions

def train(model, )

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-input', type=str,
            help='Path to the input file (e.g., dev.json)')
    parser.add_argument('-diction', type=str,
            help='Path to the dictionary')
    parser.add_argument('-output', type=str,
            help='Prediction file path')
    parser.add_argument('-cuda', type=int, default=-1,
            help='which gpu to use')
    args = parser.parse_args()
    if args.cuda == -1:
        args.device = torch.device("cpu")
    else:
        args.device = torch.device(args.cuda)

    tokenizer = AutoTokenizer.from_pretrained("princeton-nlp/sup-simcse-bert-base-uncased")
    model = AutoModel.from_pretrained("princeton-nlp/sup-simcse-bert-base-uncased")
    model.to(args.device)

    with open(args.input) as file:
        data = json.load(file)
    with open(args.diction) as file:
        diction = json.load(file)

    # Import our models. The package will take care of downloading the models automatically
    predictions = predict(model, tokenizer, data, diction)

    ## Save
    with open(args.output, 'w') as file:
        json.dump(predictions, file)


