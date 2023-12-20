import json
import ast
import openai
import numpy as np
import pandas as pd


def split_dataset(df: pd.DataFrame):
  """
  Splits the input dataset into two dataframes representing income and expense transactions

  Args:
    df (pd.DataFrame): pandas dataframe of financial transactions
  
  Returns:
    expenses (pd.DataFrame): transactions where Debit is NaN
    income (pd.DataFrame): transactions where Credit is NaN
  """
  expenses = df[~df['Debit'].isna()][['Description', 'Debit']]
  income = df[~df['Credit'].isna()][['Description', 'Credit']]
  return expenses, income


def get_chat(model, system_prompt, descriptions):
  """
  Use ChatGPT API to label transactions

  Args:
    model (str): version of chatGPT
    system_prompt (str): instructions for ChatGPT to follow
    descriptions (str): transaction descriptions

  Returns:
    Labels (str): string list of labels
  """

  response = openai.ChatCompletion.create(
      model=model,
      messages=[
          {"role": "system", "content": system_prompt},
          {"role": "user", "content": descriptions}
      ],
      temperature=0
  )
  return response['choices'][0]['message']['content']


if __name__ == "__main__":
    # Read transactions dataset
    df_transactions = pd.read_csv('./bank_transactions.csv')
    # split dataset into income and expenses
    expenses, income = split_dataset(df_transactions)
    # Load system prompts for income and expense
    system_prompts = json.loads('system.json')
    # get labels from ChatGPT
    labels_expense = get_chat('gpt-3.5-turbo', system_prompts['system_prompt_expense'], str(expenses['Description'].tolist()))
    labels_income = get_chat('gpt-3.5-turbo', system_prompts['system_prompt_income'], str(income['Description'].tolist()))
    # Transform string representations of list from ChatGPT to actual Python list
    labels_expense = ast.literal_eval(labels_expense)
    labels_income = ast.literal_eval(labels_income)
    # Add Category column to expense and income
    expenses['Category'] = labels_expense
    income['Category'] = labels_income
    # Merge for final output dataframe
    categorized_transactions = pd.concat([expenses, income], axis=0)
    # Write output dataframe to local file path
    categorized_transactions.to_csv('categorized_gpt.csv')


