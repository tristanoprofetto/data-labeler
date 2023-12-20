import json
import re
import numpy as np
import pandas as pd
import nltk
import torch
import torch.nn.functional as F
from transformers import AutoModelForSequenceClassification, AutoTokenizer


def clean_descriptions(description):
    """
    Applies regex transformations to clean transaction descriptions
    """
    # set all characters to lower case
    description = description.lower()
    # remove special characters
    description = re.sub(r'[^\w\s]|https?://\S+|www\.\S+|https?:/\S+|[^\x00-\x7F]+|\d+', '', str(description).strip())
    
    return description


def get_predictions_from_pretrained(model, tokenizer, classes, transactions, outputs = []):
    """
    Applies transactions categorization to input descriptions

    Args:
        model (transformers.AutoModelForSequenceClassification): pretrained model object
        tokenizer (trasformers.AutoTokenizer): pretrained tokenizer 
        classes (dict): output label classes
        transactions (pd.DataFrame): input bank transactions dataframe
        outputs (list): list of output transaction labels 

    Returns:
        outputs (list): list of output transaction labels 
    """
    for i in range(len(transactions)):
        # current description to be processed
        input_description = transactions['Description'][i]
        # tokenize transaction description
        tokenized_description = tokenizer(input_description, return_tensors='pt')
        # get model pretrained model outputs
        prediction = model(**tokenized_description)
        logits = F.softmax(prediction.logits, dim=1)
        # map probabilities to labels
        label_id = torch.argmax(logits, dim=1).item()
        label_class = classes[label_id]
        # strip unecessary characters from label
        outputs.append(label_class[9:])
    
    return outputs


def pred_2_category(df_transactions, outputs, class_map):
    """
    Maps pretrained model labels to proper Zum defined categories

    Args:
        df_transactions (pd.DataFrame): transactions dataset
        outputs (list): pretrained model outputs
        class_map (dict): mapping of pretrained labels to proper income/expense labels
    
    Returns: 
        df_transactions (pd.DataFrame): updated transactions dataset with 'Category' column
    """
    # Loop through pretrained labels and map to actual income/expense labels
    for i in range(len(outputs)):
        categories = []
        if outputs[i] in class_map:
            categories.append(class_map[outputs[i]])
    # Add category column
    df_transactions['Category'] = categories
    return df_transactions


if __name__ == "__main__":
    # Read transactions dataset
    df_transactions = pd.read_csv('./bank_transactions.csv')
    # clean transaction descriptions
    df_transactions['Description'] = df_transactions['Description'].apply(lambda x: clean_descriptions(x))
    # Load pretrained model, tokenizer, and class labels
    model = AutoModelForSequenceClassification.from_pretrained('mgrella/autonlp-bank-transaction-classification-5521155')
    tokenizer = AutoTokenizer.from_pretrained('mgrella/autonlp-bank-transaction-classification-5521155')
    classes = model.config.id2label
    # obtain predictions from pretrained model
    predictions = get_predictions_from_pretrained(model, tokenizer, classes, df_transactions)
    # load class mapping from json file
    class_map = json.loads('classes.json')
    # get final output dataset
    categorized_transactions = pred_2_category(df_transactions, predictions, class_map)
    # write csv to local path
    categorized_transactions.to_csv('categorized_pretrained.csv')


