# Data Labeler for Financial Transactions
The purpose of this repository is to build a solution that can correctly categorize a list of transaction desccriptions into the following primary-categories:
* Income
* Expenses

### About the Dataset
Given 189 transactions:
| **Column Name** | **Type** | 
| Description | string | Raw transaction description |
| Debit | float | Amount of money ($) flowing out of the given account (expense) |
| Credit | float | Amount of money ($) flowing into the account (income) |

* There are currently no rows where Credit AND Debit are 'nan'
* We assume that debited transactions are expense related since money is flowing out of the account, and credit transactions are income related since money is flowing into the account.

### Categories of Transaction Descriptions
Income Categories:
* Employment Income
* Government Income
* Other Income

Expense Categories:
* Bills and Utilities
* Entertainment
* Fees and Charges
* Shopping
* Food and Dining
* Health and Fitness
* Loans
* Investments
* Pets
* Travel
* Taxes
* Homes
* Gifts and Donations
* Other

### Solutions
Two solutions are proposed:
1. Pre-Trained Transformer Classifier (from Huggingface)
2. ChatGPT 

Before testing either solution be sure to install the required packages:
```pip install -r requirements.txt```


### Pretrained Transformer
The proposed solution is to leverage the capabilities of pretrained Transformers. Because of a lack of labeled data pretrained models are a good starting point to build up a proper dataset that would
allow for applying more classical supervised, unsupervised, or perhaps even semi-supervised techniques in the future.

Categorization can be done with the pretrained solution by running:
```python ./pretrained/model.py``` 

### ChatGPT
The proposed solution is to leverage the generalized capabilities of large language models (LLM's), specifically, OpenAI's ChatGPT, to assign labels to input descriptions.
After experimenting with this solution I have verified that the outputs correctly match the corresponding labels.
Ofcourse, ChatGPT is not free, but if we could use it to build a glossary of transaction descriptions and their primary and sub categories in the form of {key: value} pairs,
over time would save alot of money in the form of human labor and time for developing lower level solutions.

Categorization can be done with the ChatGPT solution by running:
```python ./gpt/chat.py``` 

**NOTE** For security reasons I did not provide my own API key for the ChatGPT API