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


def get_dataset_for_model(model_name, labels, config):
    if model_name == "bert_cls_basic":
        return ClassificationDataset(labels=labels,**config)
    elif model_name == "bert_matcher_basic":
        return NLIDataset(labels=labels,**config)
    raise Exception("Unknown model %s" % model_name)

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


class ClassificationDataset(Dataset):
    def __init__(self, data_file, max_length=60, limit=0, device=None, labels=[]):
        super(ClassificationDataset, self).__init__()

        self.csv_file = data_file
        csv_props = {"header": 0, "sep": "\t", "encoding": 'utf8'}
        if limit > 0:
            csv_props["nrows"] = limit
        self.data_frame = pd.read_csv(self.csv_file, **csv_props)

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
        text = row["text"]
        label = self._get_label_id(row["label"])
        token_id_tensor, type_id_tensor, attn_mask_tensor = self.preprocessor.process_text(text, None,
                                                                                           self.max_length)

        return token_id_tensor, type_id_tensor, attn_mask_tensor, label

    def get_columns(self):
        return ["token_id_tensor", "type_id_tensor", "attn_mask_tensor", "label"]

class NLIDataset(Dataset):
    def __init__(self, dataset_file, max_length=60, limit=0, device=None, labels=[]):
        super(NLIDataset, self).__init__()

        self.csv_file = dataset_file
        csv_props = {"header": 0, "sep": "\t", "encoding": 'utf8'}
        if limit > 0:
            csv_props["nrows"] = limit
        self.data_frame = pd.read_csv(self.csv_file, **csv_props)

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

    def get_columns(self):
        return ["token_id_tensor", "type_id_tensor", "attn_mask_tensor","label"]
