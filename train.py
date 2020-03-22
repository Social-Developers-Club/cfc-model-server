"""

IDE: PyCharm
Project: semantic-match-classifier
Author: Robin
Filename: train.py
Date: 21.03.2020
TODO: training loop + validation?
"""
import json
import os

import torch
import torch.nn as nn
from tqdm import tqdm
from transformers import AdamW, get_linear_schedule_with_warmup

from data import NLIDataset
from model import BertClassifierModel

with open("data/config/bert_cls_config.json", "r", encoding="utf8") as config_file:
    hyperparams = json.load(config_file)

# init device
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
print("Device: " + str(device))

data_path = "data/preprocessed/"
train_file = os.path.join(data_path, "mnli_train_translated.tsv")
val_file = ""
epochs = hyperparams["n_epochs"]

# load and batch data
train_dataset = NLIDataset(train_file, labels=["entailment", "contradiction", "neutral"])
train_loader = torch.utils.data.DataLoader(
    train_dataset,
    batch_size=hyperparams["batch_size"], shuffle=True, num_workers=0, pin_memory=True)

# init model
model = BertClassifierModel(n_outputs=3).to(device)
model.train()

# init optim
optimizer = AdamW(model.parameters(), lr=hyperparams["learning_rate"],
                  correct_bias=False)  # To reproduce BertAdam specific behavior set correct_bias=False
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=100,
                                            num_training_steps=1000)  # PyTorch scheduler

criterion = nn.CrossEntropyLoss().to(device)

for epoch in range(1, epochs + 1):
    for index, (token_id_tensor, type_id_tensor, attn_mask_tensor, label,) in tqdm(enumerate(train_loader), total=int(
            len(train_dataset) / hyperparams["batch_size"])):
        token_id_tensor = token_id_tensor.to(device)
        type_id_tensor = type_id_tensor.to(device)
        attn_mask_tensor = attn_mask_tensor.to(device)
        label = label.squeeze(dim=1).to(device)

        # zero gradients
        model.zero_grad()

        # calculate loss
        logits = model(token_id_tensor, type_id_tensor, attn_mask_tensor)[0]
        probs = torch.softmax(logits, dim=1).tolist()[0]

        loss = criterion(logits, label)
        loss.backward()

        # Gradient clipping is not in AdamW anymore (so you can use amp without issue)
        torch.nn.utils.clip_grad_norm_(model.parameters(), hyperparams["max_grad_norm"])

        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()

        print(loss.detach().item())
