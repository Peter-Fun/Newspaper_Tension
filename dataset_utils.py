import os
from dataset_bert import GDELT_BERT
from typing import List
import tqdm
import random
from collections import defaultdict

SPLIT_OPTIONS = ['train', 'test']


def split_dataset(data_dir: str, train_ratio: float=0.8):
    """
    Given a dataset, assign split.
    """
    # Open and Process Data.
    tsvs = recursive_scan_tsv_files(data_dir)
    # Sort TSVs by Pair Folder:
    pair2tsv = defaultdict(list)
    for tsv in tsvs:
        pair = os.path.dirname(os.path.dirname(tsv))
        pair2tsv[pair].append(tsv)

    for pair, pair_tsvs in tqdm.tqdm(pair2tsv.items(), leave=False, desc="Pairs"):
        split_data = []
        for tsv in tqdm.tqdm(pair_tsvs, leave=False, desc="tsv_files"):
            with open(tsv, "r") as buffer:
                tsv_data = buffer.readlines()
            for row in tsv_data:
                split = "train" if random.random() <= train_ratio else "test"
                split_data.append(row[:-1] + "\t" + split + "\n")
        split_file = os.path.join(pair, "split.tsv")
        with open(split_file, "w") as buffer:
                buffer.writelines(split_data)


def recursive_scan_tsv_files(folder: str) -> List[str]:
    """
    Recursively scan a folder for tsv files. Returns absolute path to each tsv file.
    """
    tsvs = []
    for root, _, files in os.walk(folder):
        if os.path.basename(root) == "extracted" or os.path.basename(root) == "tsv_files":
            for filename in files:
                if os.path.splitext(filename)[1] == ".tsv":
                    tsvs.append(os.path.abspath(os.path.join(root, filename)))
    return tsvs


if __name__=="__main__":
    """
    Main function used for generating splits.
    """
    import argparse
    parser = argparse.ArgumentParser(description="split generator")
    parser.add_argument("data_dir", type=str, help="path to the GDELT data directory")
    parser.add_argument("-r", "--ratio", type=float, default=0.8, help="The ratio used for training")
    args = parser.parse_args()

    split_dataset(args.data_dir, train_ratio=args.ratio)