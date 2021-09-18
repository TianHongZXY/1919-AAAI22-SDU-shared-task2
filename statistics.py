import json
import argparse


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--diction", type=str)
    args = parser.parse_args()
    with open(args.diction) as file:
        diction = json.load(file)

    distinct_abbr = []
    distinct_long_term = []

    for key, val in diction.items():
        distinct_abbr.append(key)
        distinct_long_term.extend(val)

    print(f"Total {len(distinct_abbr)} unique abbr words.")
    print(f"Total {len(distinct_long_term)} long term words.")

