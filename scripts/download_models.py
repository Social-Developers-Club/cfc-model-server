"""

IDE: PyCharm
Project: semantic-match-classifier
Author: Robin
Filename: download_models.py
Date: 22.03.2020

"""
import sys

sys.path.append("../")

from model import BertClassifierModel, BertMatcherModel

models = [
    BertClassifierModel,
    BertMatcherModel
]

for model in models:
    model = model()
