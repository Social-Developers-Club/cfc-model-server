"""

IDE: PyCharm
Project: semantic-match-classifier
Author: Robin
Filename: train.py
Date: 21.03.2020
"""
import json
import os

import torch
import torch.nn as nn
from torch.utils.data import Dataset
from tqdm import tqdm
from transformers import AdamW, get_linear_schedule_with_warmup

from data import get_dataset_for_model
from model import copy_to_device, get_model

with open("data/config/bert_cls_config.json", "r", encoding="utf8") as config_file:
    hyperparams = json.load(config_file)

# init device
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
print("Device: " + str(device))

data_path = "data/preprocessed/"
model_path = "data/models/"
model_name = hyperparams["model"]
epochs = hyperparams["n_epochs"]
labels = hyperparams["labels"]

# load and batch data
train_dataset: Dataset = get_dataset_for_model(hyperparams["model"], hyperparams["labels"], hyperparams["dataset"]["train"])
train_loader = torch.utils.data.DataLoader(
    train_dataset,
    batch_size=hyperparams["batch_size"], shuffle=True, num_workers=0, pin_memory=True)

val_dataset: Dataset  = get_dataset_for_model(hyperparams["model"], hyperparams["labels"], hyperparams["dataset"]["validation"])
val_loader = torch.utils.data.DataLoader(
    val_dataset,
    batch_size=hyperparams["batch_size"], shuffle=True, num_workers=0, pin_memory=True)

# init model
model = get_model(hyperparams["model"], hyperparams["model_config"], device)
model.train()

# init optim
optimizer = AdamW(model.parameters(), lr=hyperparams["learning_rate"],
                  correct_bias=False)  # To reproduce BertAdam specific behavior set correct_bias=False
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=100,
                                            num_training_steps=1000)  # PyTorch scheduler

criterion = nn.CrossEntropyLoss().to(device)

best_val_loss = float('inf')
best_accuracy = .0

for epoch in range(1, epochs + 1):
    epoch_loss = .0
    for index, batch_data in tqdm(enumerate(train_loader), total=int(
            len(train_dataset) / hyperparams["batch_size"])):
        batch_data = copy_to_device(batch_data, device=device, columns=train_dataset.get_columns())

        # zero gradients
        model.zero_grad()

        # calculate loss
        logits = model(**batch_data)
        loss = criterion(logits, batch_data["label"])
        loss.backward()

        epoch_loss += loss.detach().item()

        # Gradient clipping is not in AdamW anymore (so you can use amp without issue)
        torch.nn.utils.clip_grad_norm_(model.parameters(), hyperparams["max_grad_norm"])

        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()

    print("Epoch loss %.04f" % epoch_loss)

    model.eval()
    val_loss = .0
    for index, batch_data in tqdm(enumerate(val_loader), total=int(
            len(train_dataset) / hyperparams["batch_size"])):
        batch_data = copy_to_device(batch_data, device=device, columns=val_dataset.get_columns())

        logits = model(**batch_data)
        probs = torch.softmax(logits, dim=1)

        loss = criterion(logits, batch_data["label"])
        val_loss += loss.item()

        accuracy = (probs.argmax(dim=1) == batch_data["label"]).sum().float() / float(batch_data["label"].size(0))

    if val_loss < best_val_loss:
        best_accuracy = accuracy
        best_val_loss = val_loss
        torch.save(model.state_dict(), os.path.join(model_path, "nli_model_basic.pt"))
    print("Val loss {0:.04f}, Accuracy: {1:.04f}".format(val_loss, accuracy))
    model.train()
