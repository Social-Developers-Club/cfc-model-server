"""

IDE: PyCharm
Project: semantic-match-classifier
Author: Robin
Filename: data.yp
Date: 21.03.2020
TODO: custom data loader, tokenizer, data cleaning
"""
import pandas as pd
import torch
from torch.utils.data import Dataset
from transformers import BertTokenizer


class BertPreprocessor:
    def __init__(self):
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-cased')

    def process_text(self, texta, textb, max_length=None):
        """
        Tokenizes the text
        :param text:
        :return:
        """
        output = self.tokenizer.encode_plus(texta, textb, pad_to_max_length=True, return_tensors="pt",
                                            max_length=max_length)
        # token_ids, type_ids, attn_mask
        return output["input_ids"], output["token_type_ids"], output["attention_mask"]


class NLIDataset(Dataset):
    def __init__(self, dataset_file, max_length=100, limit=0, device=None, labels=[]):
        super(NLIDataset, self).__init__()

        self.csv_file = dataset_file
        if limit <= 0:
            self.data_frame = pd.read_csv(self.csv_file, header=0, sep="\t", encoding='utf8')
        else:
            self.data_frame = pd.read_csv(self.csv_file, nrows=limit, header=0, encoding='utf8')

        self.preprocessor = BertPreprocessor()
        self.device = device
        self.max_length = max_length
        self.labels = labels

    def _get_label_id(self, name):
        return torch.LongTensor([self.labels.index(name)])

    def __len__(self):
        return len(self.data_frame)

    def __getitem__(self, idx):
        row = self.data_frame.iloc[idx]

        sentence_a = row["sentenceA"]
        sentence_b = row["sentenceB"]
        label = self._get_label_id(row["label"])
        token_id_tensor, type_id_tensor, attn_mask_tensor = self.preprocessor.process_text(sentence_a, sentence_b,
                                                                                           self.max_length)

        return token_id_tensor, type_id_tensor, attn_mask_tensor, label
