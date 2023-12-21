#!/bin/bash

ENV=$1

case $ENV in
    pretrained)
pretrained_vars(
    [INPUT_FILE_PATH]="bank_transactions.csv"
    [PRETRAINED_MODEL_NAME]="mgrella/autonlp-bank-transaction-classification-5521155"
    [PRETRAINED_TOKENIZER_NAME]="mgrella/autonlp-bank-transaction-classification-5521155"
    [PRETRAINED_CLASS_LABELS]="./pretrained/classes.json"
    [PRETRAINED_OUTPUT_FILE_PATH]="categorized_pretrained.csv"
)


