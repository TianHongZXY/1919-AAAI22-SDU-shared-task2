import json
import argparse
from tqdm import tqdm

def count_duplicate(file_path):
    with open(file_path) as file:
        f = json.load(file)
    uniq_sent = set()
    cnt = 0
    for idx, example in enumerate(tqdm(f)):
        sent = example['sentence'].lower()
        if sent in uniq_sent:
            cnt += 1
            print(example)
        else:
            uniq_sent.add(sent.lower())
    print("duplicate count: ", cnt)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    #  parser.add_argument("--diction", type=str)
    parser.add_argument("--file", type=str)
    args = parser.parse_args()
    #  with open(args.diction) as file:
    #      diction = json.load(file)
    #
    #  distinct_abbr = []
    #  distinct_long_term = []
    #
    #  for key, val in diction.items():
    #      distinct_abbr.append(key)
    #      distinct_long_term.extend(val)
    #
    #  print(f"Total {len(distinct_abbr)} unique abbr words.")
    #  print(f"Total {len(distinct_long_term)} long term words.")
    count_duplicate(args.file)
