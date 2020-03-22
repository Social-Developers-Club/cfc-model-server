"""

IDE: PyCharm
Project: semantic-match-classifier
Author: Robin
Filename: fix_csv.py
Date: 22.03.2020

"""
import re

csv_path = "../data/preprocessed/mnli_train_translated.tsv"
output_path = "../data/preprocessed/mnli_train_translated_fixed.tsv"

with open(csv_path, "r", encoding="utf8") as csv_file:
    with open(output_path, "w+", encoding="utf8", newline='') as output_file:
        for line in csv_file:
            line = re.sub("[ ]+[\t]", "\t", line)
            output_file.write(line)