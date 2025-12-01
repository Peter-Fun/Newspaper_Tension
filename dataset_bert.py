import torch
import tqdm
import random
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import torchtext.data.utils as tt_ut
import torchtext.vocab as tt_vb

from Newspaper_Tension.constants import DEFAULT_COLUMN_LABELS, DEFAULT_DOWNLOAD_DIR, DEFAULT_GDELT_BASE_URL
from Newspaper_Tension.utils import load_GDELT_tsv
from common.logging import setup_logging, getlogger

SPLIT_OPTIONS = ['train', 'test']
SPLIT_WEIGHTS = [0.8, 0.2]

logger = getlogger("ISEF_Training")

class GDELT_BERT(Dataset):
    """
    Create a dataset to load the GDELT csv file.
    """
    def __init__(
            self,
            tokenizer,
            root_dir: str = None, 
            selected_factor = "GoldsteinScale", 
            split: str = None,
            max_length: int = 512,
            show_progress: bool = True
        ):
        """
        Construct the GDELT class. Prepare to load the article text and label requested.

        Args:
            root_dir: path to the database root. (containing pairs)
            csv: a path to the file containing the csv file to load
            selected_factor: the requested column of the csv file to predict from the article, default of goldstein
            split: either "train" or "test", default "train"
        """
        self.progress = show_progress
        if split:
            assert split in SPLIT_OPTIONS
        self._root_dir = root_dir or os.path.abspath("/home/peter/data/GDELT_large")
        self.selected_factor = selected_factor # chosen factor to predict based on given article
        self.split = split or "train"
        logger.info("Finding TSVs...")
        self._csv_paths = self._find_split_files()
        logger.info(f"Found {len(self._csv_paths)} TSVs")
        logger.info(f"Making the dataset...")
        self.data = self.make_dataset(selected_factor)
        logger.info(f"Made dataset with length {self.__len__()}")
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.data = self.keep_split(self.data)

       
    def _find_split_files(self,):
        """
        Look for csv files within the root directory.
        """

        csv_paths = []
        for root, _, files in os.walk(self._root_dir):
            for filename in files:
                if os.path.basename(filename) == "split.tsv":
                    csv_paths.append(os.path.abspath(os.path.join(root, filename)))
        return csv_paths


    def make_dataset(self, selected_factor: str = None):
        """
        Read in all of the texts and label information.
        """
        selected_factor = selected_factor or "GoldsteinScale"
        self.labels = []
        self.texts = []
        self.summaries = []
        self.splits = []
        for csv_path in tqdm.tqdm(self._csv_paths, disable=(not self.progress)):
            pair_folder = os.path.dirname(csv_path)
            article_folder = os.path.join(pair_folder, "articles")
            summary_folder = os.path.join(pair_folder, "summaries")
            try: # in case the tsv file is empty
                csv_data = load_GDELT_tsv(csv_path)
                events = csv_data["GLOBALEVENTID"].tolist()  # Getting GLOBALEVENTID.
                labels = csv_data[selected_factor].tolist()  # Getting GoldsteinScale scores.
                splits = csv_data["SPLIT"].tolist()

                texts = []
                summaries = []
                delete_inds = []
                for ind, (event, split) in enumerate(zip(events, splits)):
                    article_path = os.path.join(article_folder, str(event) + ".txt")
                    summary_path = os.path.join(summary_folder, str(event) + ".txt")
                    if os.path.isfile(article_path) and os.path.isfile(summary_path) and split==self.split:
                        with open(article_path, "r") as buffer:
                            texts.append(buffer.read())
                        with open(summary_path, "r") as buffer:
                            summaries.append(buffer.read())
                    else:
                        delete_inds.append(ind)
                for ind in sorted(delete_inds, reverse=True):
                    del labels[ind]
                    del splits[ind]
                
                self.texts.extend(texts)
                self.summaries.extend(summaries)
                self.labels.extend(labels)
                self.splits.extend(splits)
            except:
                pass

        data = np.array([self.texts, self.summaries, self.labels, self.splits ], dtype=object).T
        return data


    def __len__(self):
        return self.data.shape[0]
    
    def __getitem__(self, idx: int):
        
        text, summary, label, s = self.data[idx, :]

        inputs = self.tokenizer.encode_plus(text, None,
            padding='max_length',
            pad_to_max_length=True,
            add_special_tokens=True,
            return_attention_mask=True,
            max_length=self.max_length,
            truncation=True,
        )
        ids = inputs["input_ids"]
        token_type_ids = inputs["token_type_ids"]
        mask = inputs["attention_mask"]

        return {
            "ids": torch.tensor(ids, dtype=torch.long),
            "mask": torch.tensor(mask, dtype=torch.long),
            "token_type_ids": torch.tensor(token_type_ids, dtype=torch.long),
            "target": int(label)-1,
            "text": text
        }
    
    def keep_split(self, data):
        return data[data[:, 3] == self.split, :] # return all rows where split column is equal to requested split
    
    def collate_fn(self, batch):
        """
        batch = [(text_i, summary_i, float(label) s), ....]

        Return
            batch_text, batch_summary, batch_label, batch_s
        """
        batch_size = len(batch)
        # Find the text input (tensor of ints) that is the longest.
        longest_length = 0
        for text, summary, label, s in batch:
            if text.shape[0] > longest_length:
                longest_length = text.shape[0]

        # Use a tensor to contain the batch's text, padding the shorter texts
        batch_texts = torch.zeros((batch_size, longest_length), dtype=int)
        batch_summaries = torch.zeros((batch_size, longest_length), dtype=int)
        batch_labels = torch.zeros((batch_size,), dtype=int)
        batch_lengths = torch.zeros((batch_size,), dtype=int)
        batch_summary_lengths = torch.zeros((batch_size,), dtype=int)
        batch_splits = torch.zeros((batch_size,), dtype=int)
        for i, (sample_text, sample_summary, sample_label, sample_s) in enumerate(batch):
            sample_size = sample_text.shape[0]
            batch_texts[i,:sample_size] = sample_text

            sample_summary_size = sample_summary.shape[0]
            batch_summaries[i,:sample_summary_size] = sample_summary

            batch_lengths[i] = sample_size
            batch_summary_lengths[i] = sample_summary_size

            batch_labels[i] = sample_label
            batch_splits[i] = 0 if sample_s == "train" else 1

        return batch_texts, batch_summaries, batch_labels, batch_splits, batch_lengths, batch_summary_lengths