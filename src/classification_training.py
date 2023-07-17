# %% [markdown]
# Load dependencies
# used to make train/dev/test partitions
from sklearn.model_selection import train_test_split
from typing import Dict
from sklearn.metrics import precision_recall_fscore_support
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import Dataset
from transformers import Trainer, TrainingArguments
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import json
import pandas as pd
import numpy as np
from tqdm.auto import tqdm
from IPython.display import clear_output, display
from sklearn.model_selection import train_test_split
import torch
from sklearn.metrics import classification_report, f1_score, accuracy_score, precision_score, recall_score, log_loss
import random
import os
# import csv
from pgfuncs import tokenize_and_vectorize, pad_trunc, collect_expected, tokenize_and_vectorize_1dlist, collect_expected_1dlist, df_classification_report

from datetime import datetime
# timestamp for file naming
now = datetime.now()
time_str = now.strftime("%Y-%m-%d_%H-%M-%S")
date_str = now.strftime("%Y-%m-%d")


def set_seeds(seed_value=42):
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    torch.cuda.manual_seed_all(seed_value)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# %% Load data


# load incelsis_5203 dataset
df_incelsis_5203 = pd.read_csv(
    '/home/pgajo/working/data/datasets/English/Incels.is/IFD-EN-5203_splits.csv')

df_train_incelsis_5203 = df_incelsis_5203[df_incelsis_5203['data_type']
                                          == 'train_incelsis']
df_dev_incelsis_5203 = df_incelsis_5203[df_incelsis_5203['data_type']
                                        == 'dev_incelsis']
df_test_incelsis_5203 = df_incelsis_5203[df_incelsis_5203['data_type']
                                         == 'test_incelsis']

# Print the size of each split
print('Incels.is train set size:', len(df_train_incelsis_5203))
print('Incels.is dev set size:', len(df_dev_incelsis_5203))
print('Incels.is test set size:', len(df_test_incelsis_5203))

# load fdb_500 dataset
df_fdb_500 = pd.read_csv(
    '/home/pgajo/working/data/datasets/Italian/Il_forum_dei_brutti/IFD-IT-500.csv')
df_fdb_500 = df_fdb_500[['hs', 'misogynous', 'racist', 'text']]
df_fdb_500
df_fdb_500['data_type'] = 'test_fdb_500'

print('Forum dei brutti test set size:', len(df_fdb_500))

# load the davidson set
file_path_csv_davidson = '/home/pgajo/working/data/datasets/English/hate-speech-and-offensive-language (davidson)/davidson_labeled_data.csv'
df_davidson = pd.read_csv(file_path_csv_davidson, index_col=None)
df_davidson = df_davidson[['hs', 'text']]
df_davidson['data_type'] = 'davidson'
df_davidson = df_davidson.sample(
    frac=1).reset_index(drop=True)  # shuffle the set
mask = df_davidson['hs'] >= 1

# Set those values to 1
df_davidson.loc[mask, 'hs'] = 1

# Split the data into training and test sets (70% for training, 30% for test)
df_train_davidson, df_test_davidson = train_test_split(
    df_davidson, test_size=0.3, random_state=42)

# Split the test data into validation and test sets (50% for validation, 50% for test)
df_dev_davidson, df_test_davidson = train_test_split(
    df_test_davidson, test_size=0.5, random_state=42)

# sample and get the same proportions of binary classes as the incels dataset

# Set the desired proportions of 1's and 0's in the sample
prop_1 = df_train_incelsis_5203['hs'].value_counts(normalize=True)[1]
df_len = len(df_train_incelsis_5203)
# Calculate the number of rows with 1s and 0s in the sample
num_hs_1 = int(df_len * prop_1)
num_hs_0 = df_len - num_hs_1

# Select rows with 1s and 0s separately, and concatenate the results
df_hs_1 = df_train_davidson[df_train_davidson['hs']
                            == 1].sample(n=num_hs_1, replace=True)
df_hs_0 = df_train_davidson[df_train_davidson['hs']
                            == 0].sample(n=num_hs_0, replace=True)
df_train_davidson_sample = pd.concat([df_hs_1, df_hs_0])

# Print the sample
# print(df_sample_davidson['hs'].value_counts(normalize=True))
# print(df_sample_davidson['hs'].value_counts(normalize=False))

# Print the sample
print('df_train_davidson_sample value_counts:')
print(df_train_davidson_sample['hs'].value_counts(normalize=False))
print()

# Print the size of each split
df_train_davidson['data_type'] = 'train_davidson'
df_dev_davidson['data_type'] = 'dev_davidson'
df_test_davidson['data_type'] = 'test_davidson'
print('Davidson full train set size:', len(df_train_davidson))
print('Davidson full dev set size:', len(df_dev_davidson))
print('Davidson full test set size:', len(df_test_davidson))

# load the hateval_2019_english set
file_path_csv_hateval_2019_english_train = '/home/pgajo/working/data/datasets/English/hateval2019_en/hateval2019_en_train_miso.csv'
file_path_csv_hateval_2019_english_dev = '/home/pgajo/working/data/datasets/English/hateval2019_en/hateval2019_en_dev_miso.csv'
file_path_csv_hateval_2019_english_test = '/home/pgajo/working/data/datasets/English/hateval2019_en/hateval2019_en_test_miso.csv'

df_train_hateval_2019_english = pd.read_csv(
    file_path_csv_hateval_2019_english_train, index_col=None)
df_dev_hateval_2019_english = pd.read_csv(
    file_path_csv_hateval_2019_english_dev, index_col=None)
df_test_hateval_2019_english = pd.read_csv(
    file_path_csv_hateval_2019_english_test, index_col=None)

# sample and get the same proportions of binary classes as the incels dataset

# Set the desired proportions of 1's and 0's in the sample
prop_1 = df_train_incelsis_5203['hs'].value_counts(normalize=True)[1]
df_len = len(df_train_incelsis_5203)
# Calculate the number of rows with 1s and 0s in the sample
num_hs_1 = int(df_len * prop_1)
num_hs_0 = df_len - num_hs_1

# Select rows with 1s and 0s separately, and concatenate the results
df_hs_1 = df_train_hateval_2019_english[df_train_hateval_2019_english['hs'] == 1].sample(
    n=num_hs_1, replace=True)
df_hs_0 = df_train_hateval_2019_english[df_train_hateval_2019_english['hs'] == 0].sample(
    n=num_hs_0, replace=True)
df_train_hateval_2019_english_sample = pd.concat([df_hs_1, df_hs_0])

# Print the sample
print('HatEval english sample value_counts:')
print(df_train_hateval_2019_english_sample['hs'].value_counts(normalize=False))
print()
df_train_hateval_2019_english['data_type'] = 'train_hateval_2019_english'
df_dev_hateval_2019_english['data_type'] = 'dev_hateval_2019_english'
df_test_hateval_2019_english['data_type'] = 'test_hateval_2019_english'
print('HatEval english full train set size:',
      len(df_train_hateval_2019_english))
print('HatEval english full dev set size:', len(df_dev_hateval_2019_english))
print('HatEval english full test set size:', len(df_test_hateval_2019_english))

# load the hateval_2019_spanish set
file_path_csv_hateval_2019_spanish_train = '/home/pgajo/working/data/datasets/Spanish/hateval2019_es/hateval2019_es_train.csv'
file_path_csv_hateval_2019_spanish_dev = '/home/pgajo/working/data/datasets/Spanish/hateval2019_es/hateval2019_es_dev.csv'
file_path_csv_hateval_2019_spanish_test = '/home/pgajo/working/data/datasets/Spanish/hateval2019_es/hateval2019_es_test.csv'

df_train_hateval_2019_spanish = pd.read_csv(
    file_path_csv_hateval_2019_spanish_train, index_col=None)
df_train_hateval_2019_spanish = df_train_hateval_2019_spanish.rename(columns={
                                                                     'HS': 'hs'})

df_dev_hateval_2019_spanish = pd.read_csv(
    file_path_csv_hateval_2019_spanish_dev, index_col=None)
df_dev_hateval_2019_spanish = df_dev_hateval_2019_spanish.rename(columns={
                                                                 'HS': 'hs'})

df_test_hateval_2019_spanish = pd.read_csv(
    file_path_csv_hateval_2019_spanish_test, index_col=None)
df_test_hateval_2019_spanish = df_test_hateval_2019_spanish.rename(columns={
                                                                   'HS': 'hs'})

# sample and get the same proportions of binary classes as the incels dataset

# Set the desired proportions of 1's and 0's in the sample
prop_1 = df_train_incelsis_5203['hs'].value_counts(normalize=True)[1]
df_len = len(df_train_incelsis_5203)
# Calculate the number of rows with 1s and 0s in the sample
num_hs_1 = int(df_len * prop_1)
num_hs_0 = df_len - num_hs_1

# Select rows with 1s and 0s separately, and concatenate the results
df_hs_1 = df_train_hateval_2019_spanish[df_train_hateval_2019_spanish['hs'] == 1].sample(
    n=num_hs_1, replace=True)
df_hs_0 = df_train_hateval_2019_spanish[df_train_hateval_2019_spanish['hs'] == 0].sample(
    n=num_hs_0, replace=True)
df_train_hateval_2019_spanish_sample = pd.concat([df_hs_1, df_hs_0])

# Print the sample
print('HatEval spanish sample value_counts:')
print(df_train_hateval_2019_spanish_sample['hs'].value_counts(normalize=False))
print()
df_train_hateval_2019_spanish['data_type'] = 'train_hateval_2019_spanish'
df_dev_hateval_2019_spanish['data_type'] = 'dev_hateval_2019_spanish'
df_test_hateval_2019_spanish['data_type'] = 'test_hateval_2019_spanish'
print('HatEval spanish full train set size:',
      len(df_train_hateval_2019_spanish))
print('HatEval spanish full dev set size:', len(df_dev_hateval_2019_spanish))
print('HatEval spanish full test set size:', len(df_test_hateval_2019_spanish))

# load the HateXplain dataset
filename_json = '/home/pgajo/working/data/datasets/English/HateXplain/Data/dataset.json'

# Open the JSON file
with open(filename_json, 'r') as f:
    # Load the JSON data into a Python dictionary
    dataset_json = json.load(f)


def post_majority_vote_choice(label_list):
    '''
    Returns the majority vote for a post in the HateXplain json dataset.
    '''
    label_dict = {}
    for i, post_label in enumerate(label_list):
        # print(i,post_label)
        if post_label not in label_dict:
            label_dict[post_label] = 1
        else:
            label_dict[post_label] += 1
    max_key = max(label_dict, key=label_dict.get)
    if label_dict[max_key] > 1:
        return max_key  # return the label key with the highest value if > 1


df_hatexplain_list = []
for key_post in dataset_json.keys():
    post = []
    labels_post = [key_annotators['label']
                   for key_annotators in dataset_json[key_post]['annotators']]  # get the list of labels
    label_majority = post_majority_vote_choice(
        labels_post)  # return the majority label
    if label_majority != None:  # the post_majority_vote_choice returns None if there is no majority label, i.e., they all have the same occurrences
        post.append(label_majority)  # append the label of the post
        # append the text tokens of the post
        post.append(' '.join(dataset_json[key_post]['post_tokens']))
        df_hatexplain_list.append(post)  # append the label-text pair
df_hatexplain = pd.DataFrame(df_hatexplain_list, columns=['hs', 'text'])
df_hatexplain_binary = df_hatexplain.loc[df_hatexplain['hs'] != 'offensive']
df_hatexplain_binary['hs'] = df_hatexplain_binary['hs'].replace(
    {'normal': 0, 'hatespeech': 1})
# df_hatexplain_binary
# Split the data into training and test sets (70% for training, 20% for test)
hatexplain_binary_devtest_size = 0.3
df_train_hatexplain_binary, df_test_hatexplain_binary = train_test_split(
    df_hatexplain_binary, test_size=hatexplain_binary_devtest_size, random_state=42)

# Split the test data into validation and test sets (50% for validation, 50% for test)
df_dev_hatexplain_binary, df_test_hatexplain_binary = train_test_split(
    df_test_hatexplain_binary, test_size=0.5, random_state=42)

df_train_hatexplain_binary['data_type'] = 'hatexplain_binary_train'
df_dev_hatexplain_binary['data_type'] = 'hatexplain_binary_dev'
df_test_hatexplain_binary['data_type'] = 'hatexplain_binary_test'
print('HateXplain binary dev+test split ratio:', hatexplain_binary_devtest_size)
print('HateXplain binary full train set size:', len(df_train_hatexplain_binary))
print('HateXplain binary full dev set size:', len(df_dev_hatexplain_binary))
print('HateXplain binary full test set size:', len(df_test_hatexplain_binary))

# load the stormfront dataset from "Hate speech dataset from a white supremacist forum"

df_stormfront_raw = pd.read_csv(
    '/home/pgajo/working/data/datasets/English/hate-speech-dataset-stormfront/annotations_metadata.csv')
df_stormfront_raw['label'] = df_stormfront_raw['label'].replace(
    {'noHate': 0, 'hate': 1})
df_stormfront_raw = df_stormfront_raw.rename(columns={'label': 'hs'})

post_dir = '/home/pgajo/working/data/datasets/English/hate-speech-dataset-stormfront/all_files'
dict_ids_labels = {}
dict_post_pairs_ws = []

for row in df_stormfront_raw.values.tolist():
    dict_ids_labels[row[0]] = row[4]
len(dict_ids_labels)
for filename in os.listdir(post_dir):
    with open(os.path.join(post_dir, filename), 'r') as file:
        # Read the contents of the file into a string variable
        file_contents = file.read()
        filename = filename[:-4]
    dict_post_pairs_ws.append(
        [dict_ids_labels[filename], file_contents, filename])
df_stormfront = pd.DataFrame(dict_post_pairs_ws, columns=[
                             'hs', 'text', 'filename'])
df_stormfront = df_stormfront[(
    df_stormfront['hs'] == 0) | (df_stormfront['hs'] == 1)]
df_stormfront['hs'] = df_stormfront['hs'].astype(int)

# Split the data into training and test sets (80% for training, 30% for test)
df_stormfront_devtest_size = 0.3
df_train_stormfront, df_test_stormfront = train_test_split(
    df_stormfront, test_size=df_stormfront_devtest_size, random_state=42)

# Split the test data into validation and test sets (50% for validation, 50% for test)
df_dev_stormfront, df_test_stormfront = train_test_split(
    df_test_stormfront, test_size=0.5, random_state=42)

df_train_stormfront['data_type'] = 'df_stormfront_train'
df_dev_stormfront['data_type'] = 'df_stormfront_dev'
df_test_stormfront['data_type'] = 'df_stormfront_test'
print('Stormfront dataset dev+test split size:', df_stormfront_devtest_size)
print('Stormfront dataset train set size:', len(df_train_stormfront))
print('Stormfront dataset dev set size:', len(df_dev_stormfront))
print('Stormfront dataset test set size:', len(df_test_stormfront))

# load the evalita18twitter set
file_path_csv_evalita18twitter_train = '/home/pgajo/working/data/datasets/Italian/haspeede_evalita/2018/TW-folder-20230313T173228Z-001/TW-folder/TW-train/haspeede_TW-train.tsv'

df_train_evalita18twitter = pd.read_csv(
    file_path_csv_evalita18twitter_train, sep='\t', names=['id', 'text', 'hs'])
df_train_evalita18twitter.columns = ['id', 'text', 'hs']
# display(df_train_evalita18twitter)
df_train_evalita18twitter['data_type'] = 'train_evalita18twitter'
print('evalita18twitter full train set size:', len(df_train_evalita18twitter))

# load the evalita18facebook set
file_path_csv_evalita18facebook_train = '/home/pgajo/working/data/datasets/Italian/haspeede_evalita/2018/FB-folder-20230313T173818Z-001/FB-folder/FB-train/haspeede_FB-train.tsv'

df_train_evalita18facebook = pd.read_csv(
    file_path_csv_evalita18facebook_train, sep='\t', names=['id', 'text', 'hs'])
df_train_evalita18facebook['data_type'] = 'train_evalita18facebook'
# display(df_train_evalita18facebook)
print('evalita18facebook full train set size:', len(df_train_evalita18facebook))

# load the evalita20 set
file_path_csv_evalita20_train = '/home/pgajo/working/data/datasets/Italian/haspeede_evalita/2020/haspeede2_dev/haspeede2_dev_taskAB.csv'

df_train_evalita20 = pd.read_csv(
    file_path_csv_evalita20_train, index_col=None)
# display(df_train_evalita20)
df_train_evalita20 = df_train_evalita20.fillna('')
df_train_evalita20['data_type'] = 'train_evalita20'
display(df_train_evalita20[:5])
print('evalita20 full train set size:', len(df_train_evalita20))
# display(df_train_evalita20)
# # load the offenseval_2020 dataset
# from datasets import load_dataset

# configs = ['ar', 'da', 'en', 'gr', 'tr']
# datasets = {}

# for config in configs:
#     datasets[config] = load_dataset("strombergnlp/offenseval_2020", config)

# %% Dataset combination choice

metrics_id = 23

# %% Dataset combinations

metrics_list_names = [
    # monolingual
    ['train_incelsis_5203', 'dev_incelsis_5203', 'test_incelsis_5203'],  # 0
    ['train_incelsis_5203+train_davidson_sample', 'dev_incelsis_5203', 'test_incelsis_5203'],  # 1
    ['train_incelsis_5203+train_hateval_2019_english_sample', 'dev_incelsis_5203', 'test_incelsis_5203'],  # 2
    ['train_incelsis_5203+train_davidson_sample+train_hateval_2019_english_sample', 'dev_incelsis_5203', 'test_incelsis_5203'],  # 3
    ['train_hateval_2019_english', 'dev_incelsis_5203', 'test_incelsis_5203'],  # 4 - no incelsis
    ['train_hateval_2019_english+train_davidson', 'dev_incelsis_5203', 'test_incelsis_5203'],  # 5 - no incelsis
    ['train_incelsis_5203', 'dev_hateval_2019_english', 'test_hateval_2019_english'],  # 6 - no incelsis
    ['train_davidson', 'dev_incelsis_5203', 'test_incelsis_5203'],  # 7 - no incelsis
    ['train_incelsis_5203', 'dev_davidson', 'test_davidson'],  # 8 - no incelsis
    ['train_incelsis_5203+train_davidson+train_hateval_2019_english', 'dev_davidson', 'test_davidson'],  # 9 - no incelsis
    ['train_incelsis_5203+train_hateval_2019_english', 'dev_incelsis_5203', 'test_incelsis_5203'],  # 10
    ['train_hatexplain_binary', 'dev_hatexplain_binary', 'test_hatexplain_binary'],  # 11 - no incelsis
    ['train_hatexplain_binary', 'dev_incelsis_5203', 'test_incelsis_5203'],  # 12 - no incelsis
    ['train_incelsis_5203+train_hatexplain_binary', 'dev_incelsis_5203', 'test_incelsis_5203'],  # 13
    ['train_incelsis_5203+train_hatexplain_binary+train_hateval_2019_english', 'dev_incelsis_5203', 'test_incelsis_5203'],  # 14
    ['train_incelsis_5203+train_stormfront', 'dev_incelsis_5203', 'test_incelsis_5203'],  # 15
    ['train_incelsis_5203+train_stormfront+train_hateval_2019_english', 'dev_incelsis_5203', 'test_incelsis_5203'],  # 16

    # multilingual
    ['train_incelsis_5203', 'dev_incelsis_5203', 'test_fdb_500'],  # 17
    ['train_incelsis_5203+train_hateval_2019_english', 'dev_incelsis_5203', 'test_fdb_500'],  # 18
    ['train_incelsis_5203+train_hateval_2019_spanish', 'dev_incelsis_5203', 'test_fdb_500'],  # 19
    ['train_incelsis_5203+train_hateval_2019_english+train_hateval_2019_spanish', 'dev_incelsis_5203', 'test_fdb_500'],  # 20
    ['train_incelsis_5203+train_evalita18facebook', 'dev_incelsis_5203', 'test_fdb_500'],  # 21
    ['train_incelsis_5203+train_evalita18twitter', 'dev_incelsis_5203', 'test_fdb_500'],  # 22
    ['train_incelsis_5203+train_evalita18facebook+train_evalita18twitter', 'dev_incelsis_5203', 'test_fdb_500'],  # 23
    ['train_incelsis_5203+train_evalita20', 'dev_incelsis_5203', 'test_fdb_500'],  # 24
    ['train_incelsis_5203+train_evalita18facebook+train_evalita18twitter+train_evalita20', 'dev_incelsis_5203', 'test_fdb_500'],  # 25
]

# set train datasets
df_train = pd.DataFrame()
print(metrics_list_names[metrics_id][0])
if 'incelsis' in metrics_list_names[metrics_id][0]:
    df_train = pd.concat([df_train, df_train_incelsis_5203])

if 'davidson' in metrics_list_names[metrics_id][0]:
    if 'incelsis' in metrics_list_names[metrics_id][0] and 'sample' in metrics_list_names[metrics_id][0]:
        df_train = pd.concat([df_train, df_train_davidson_sample])
    else:
        df_train = pd.concat([df_train, df_train_davidson])

if 'hateval' in metrics_list_names[metrics_id][0]:
    if 'english' in metrics_list_names[metrics_id][0]:
        if 'incelsis' in metrics_list_names[metrics_id][0] and 'sample' in metrics_list_names[metrics_id][0]:
            df_train = pd.concat(
                [df_train, df_train_hateval_2019_english_sample])
        else:
            df_train = pd.concat([df_train, df_train_hateval_2019_english])
    if 'spanish' in metrics_list_names[metrics_id][0]:
        if 'incelsis' in metrics_list_names[metrics_id][0] and 'sample' in metrics_list_names[metrics_id][0]:
            df_train = pd.concat(
                [df_train, df_train_hateval_2019_english_sample])
        else:
            df_train = pd.concat([df_train, df_train_hateval_2019_spanish])

if 'train_hatexplain_binary' in metrics_list_names[metrics_id][0]:
    df_train = pd.concat([df_train, df_train_hatexplain_binary])

if 'train_stormfront' in metrics_list_names[metrics_id][0]:
    df_train = pd.concat([df_train, df_train_stormfront])

if 'train_evalita18facebook' in metrics_list_names[metrics_id][0]:
    df_train = pd.concat([df_train, df_train_evalita18facebook])

if 'train_evalita18twitter' in metrics_list_names[metrics_id][0]:
    df_train = pd.concat([df_train, df_train_evalita18twitter])

if 'train_evalita20' in metrics_list_names[metrics_id][0]:
    df_train = pd.concat([df_train, df_train_evalita20])

# set dev datasets
df_dev = pd.DataFrame()

if 'dev_incelsis_5203' in metrics_list_names[metrics_id][1]:
    df_dev = pd.concat([df_dev, df_dev_incelsis_5203])

if 'dev_davidson' in metrics_list_names[metrics_id][1]:
    df_dev = pd.concat([df_dev, df_dev_davidson])

if 'dev_hateval_2019' in metrics_list_names[metrics_id][1]:
    if 'english' in metrics_list_names[metrics_id][1]:
        df_dev = pd.concat([df_dev, df_dev_hateval_2019_english])
    if 'spanish' in metrics_list_names[metrics_id][1]:
        df_dev = pd.concat([df_dev, df_dev_hateval_2019_spanish])

if 'dev_hatexplain_binary' in metrics_list_names[metrics_id][1]:
    df_dev = pd.concat([df_dev, df_dev_hatexplain_binary])

if 'dev_stormfront' in metrics_list_names[metrics_id][1]:
    df_dev = pd.concat([df_dev, df_dev_stormfront])


# set test datasets
if 'test_incelsis_5203' in metrics_list_names[metrics_id][2]:
    df_test = df_test_incelsis_5203

if 'test_fdb_500' in metrics_list_names[metrics_id][2]:
    df_test = df_fdb_500

if 'test_davidson' in metrics_list_names[metrics_id][2]:
    df_test = df_test_davidson

if 'test_hateval_2019' in metrics_list_names[metrics_id][2]:
    if 'english' in metrics_list_names[metrics_id][2]:
        df_test = df_test_hateval_2019_english
    if 'spanish' in metrics_list_names[metrics_id][1]:
        df_test = df_test_hateval_2019_spanish

if 'test_hatexplain_binary' in metrics_list_names[metrics_id][2]:
    df_test = df_test_hatexplain_binary


df_train = df_train.sample(frac=1)[:]
df_dev = df_dev.sample(frac=1)[:]
df_test = df_test.sample(frac=1)[:]

print('Run ID:', metrics_id)
print('Train sets:')
print(df_train['data_type'].value_counts(normalize=False))
print('Train set length:', len(df_train), '\n')
print('Dev sets:')
print(df_dev['data_type'].value_counts(normalize=False))
print('Dev set length:', len(df_dev), '\n')
print('Test sets:')
print(df_test['data_type'].value_counts(normalize=False))
print('Test set length:', len(df_test), '\n')

# %% Model choice

model_name_list = [
    # monolingual models
    'bert-base-uncased',  # 0
    'roberta-base',  # 1
    '/home/pgajo/working/pt_models/HateBERT',  # 2
    'Hate-speech-CNERG/bert-base-uncased-hatexplain', # 3
    '/home/pgajo/working/pt_models/incel-bert-base-uncased-10k_english',  # 4
    '/home/pgajo/working/pt_models/incel-bert-base-uncased-100k_english',  # 5
    '/home/pgajo/working/pt_models/incel-bert-base-uncased-1000k_english',  # 6
    '/home/pgajo/working/pt_models/incel-roberta-base-10k_english',  # 7
    '/home/pgajo/working/pt_models/incel-roberta-base-100k_english',  # 8
    '/home/pgajo/working/pt_models/incel-roberta-base-1000k_english',  # 9

    # multilingual models
    'bert-base-multilingual-cased',  # 10
    '/home/pgajo/working/pt_models/incel-bert-base-multilingual-cased-10k_multi',  # 11
    '/home/pgajo/working/pt_models/incel-bert-base-multilingual-cased-100k_multi',  # 12
    '/home/pgajo/working/pt_models/incel-bert-base-multilingual-cased-1000k_multi',  # 13
]

model_name = model_name_list[13]

# Filename bits
metrics_path_category = '/home/pgajo/working/data/metrics/1_hate_speech'
# metrics_path_category = '/home/pgajo/working/data/metrics/2_1_misogyny'
# metrics_path_category = '/home/pgajo/working/data/metrics/2_2_racism'
# metrics_path_category = '/home/pgajo/working/data/metrics/3_hate_forecasting'

index = model_name_list.index(model_name)
if index > 16:
    multilingual = 1
    metrics_save_path = f'{metrics_path_category}/metrics_multilingual/'
    if not os.path.exists(metrics_save_path):
        os.mkdir(metrics_save_path)
else:
    multilingual = 0
    metrics_save_path = f'{metrics_path_category}/metrics_monolingual/'
    if not os.path.exists(metrics_save_path):
        os.mkdir(metrics_save_path)

model_name_simple = model_name.split('/')[-1]

metrics_save_path_model = os.path.join(metrics_save_path, model_name_simple)
print(metrics_save_path_model)
# metrics_save_path_model = metrics_save_path + model_name_simple

if not os.path.exists(metrics_save_path_model):
    os.mkdir(metrics_save_path_model)

print('\n#####################################################\n',
    metrics_save_path_model,
    '\n#####################################################\n')

# reset time
now = datetime.now()
time_str = now.strftime("%Y-%m-%d_%H-%M-%S")
date_str = now.strftime("%Y-%m-%d")

# make unique filepath
metrics_filename = str(metrics_id)+'_' + \
    model_name_simple+'_'+time_str+'_metrics.csv'
metrics_csv_filepath = os.path.join(
    metrics_save_path_model, metrics_filename)
print(metrics_csv_filepath)

# get tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)

# hatexplain needs modified output layer
if model_name == 'Hate-speech-CNERG/bert-base-uncased-hatexplain':
    model.classifier = nn.Linear(model.config.hidden_size, 2)
    model.num_labels = 2
    print(model.eval())
    print(model.config)

# Data pre-processing
display(df_test)
# Encode the training data using the tokenizer
encoded_data_train = tokenizer.batch_encode_plus(
    [el for el in tqdm(df_train.text.values)],
    add_special_tokens=True,
    return_attention_mask=True,
    padding='max_length',  # change pad_to_max_length to padding
    max_length=256,
    truncation=True,  # add truncation
    return_tensors='pt'
)

encoded_data_val = tokenizer.batch_encode_plus(
    [el for el in tqdm(df_dev.text.values)],
    add_special_tokens=True,
    return_attention_mask=True,
    padding='max_length',  # change pad_to_max_length to padding
    max_length=256,
    truncation=True,  # add truncation
    return_tensors='pt'
)

encoded_data_test = tokenizer.batch_encode_plus(
    [el for el in tqdm(df_test.text.values)],
    add_special_tokens=True,
    return_attention_mask=True,
    padding='max_length',  # change pad_to_max_length to padding
    max_length=256,
    truncation=True,  # add truncation
    return_tensors='pt'
)

# Extract IDs, attention masks and labels from training dataset
input_ids_train = encoded_data_train['input_ids']
attention_masks_train = encoded_data_train['attention_mask']
labels_train = torch.tensor(df_train.hs.values)
# Extract IDs, attention masks and labels from validation dataset
input_ids_val = encoded_data_val['input_ids']
attention_masks_val = encoded_data_val['attention_mask']
labels_val = torch.tensor(df_dev.hs.values)
# Extract IDs, attention masks and labels from test dataset
input_ids_test = encoded_data_test['input_ids']
attention_masks_test = encoded_data_test['attention_mask']
labels_test = torch.tensor(df_test.hs.values)

# # Model setup
epochs = 4  # number of epochs
# Define the size of each batch
batch_size = 8  # number of examples to include in each batch

# convert my train/dev/test pandas dataframes to huggingface-compatible datasets
class CustomDataset(Dataset):
    def __init__(self, input_ids, attention_masks, labels):
        self.input_ids = input_ids
        self.attention_masks = attention_masks
        self.labels = labels

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return {'input_ids': self.input_ids[idx], 'attention_mask': self.attention_masks[idx], 'labels': self.labels[idx]}

# make initial empty metrics dataframe
df_metrics = pd.DataFrame(columns=['epoch', 'loss_train', 'eval_loss', 'eval_f1',
                        'eval_prec', 'eval_rec', 'test_loss', 'test_f1', 'test_prec', 'test_rec'])

# custom compute metrics function
def compute_metrics(eval_pred, metric_key_prefix="eval"):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, predictions, average='binary')

    return {
        f'{metric_key_prefix}_prec': precision,
        f'{metric_key_prefix}_rec': recall,
        f'{metric_key_prefix}_f1': f1
    }

# Create the custom dataset instances
train_dataset = CustomDataset(
    input_ids_train, attention_masks_train, labels_train)
val_dataset = CustomDataset(
    input_ids_val, attention_masks_val, labels_val)
test_dataset = CustomDataset(
    input_ids_test, attention_masks_test, labels_test)

# write set identifiers for the pandas metrics dataframe
df_metrics_train_set_string = ''
for i, index in enumerate(df_train['data_type'].value_counts(normalize=False).index.to_list()):
    set_len = df_train['data_type'].value_counts(
        normalize=False).values[i]
    df_metrics_train_set_string += index+'('+str(set_len)+')'+'\n'

df_metrics_dev_set_string = ''
for i, index in enumerate(df_dev['data_type'].value_counts(normalize=False).index.to_list()):
    set_len = df_dev['data_type'].value_counts(
        normalize=False).values[i]
    df_metrics_dev_set_string += index+'('+str(set_len)+')'+'\n'

df_metrics_test_set_string = ''
for i, index in enumerate(df_test['data_type'].value_counts(normalize=False).index.to_list()):
    set_len = df_test['data_type'].value_counts(
        normalize=False).values[i]
    df_metrics_test_set_string += index+'('+str(set_len)+')'+'\n'

# extend the huggingface Trainer class to make custom methods
class CustomTrainer(Trainer):
    def evaluate(self, eval_dataset=None, ignore_keys=None):
        val_output = self.predict(val_dataset)
        test_output = self.predict(test_dataset)
        val_metrics = compute_metrics(
            (val_output.predictions, val_output.label_ids), metric_key_prefix="val")
        test_metrics = compute_metrics(
            (test_output.predictions, test_output.label_ids), metric_key_prefix="test")
        df_metrics = pd.DataFrame(columns=[
                                'epoch', 'val_f1', 'val_prec', 'val_rec', 'test_f1', 'test_prec', 'test_rec'])
        if self.state.epoch == None:
            current_epoch = -1
        else:
            current_epoch = self.state.epoch
        df_metrics = df_metrics.append({
            'epoch': current_epoch, # self.state.epoch,
            'val_f1': val_metrics['val_f1'],
            'val_prec': val_metrics['val_prec'],
            'val_rec': val_metrics['val_rec'],
            'test_f1': test_metrics['test_f1'],
            'test_prec': test_metrics['test_prec'],
            'test_rec': test_metrics['test_rec'],
        }, ignore_index=True)

        df_metrics['model'] = model_name_simple
        df_metrics['train_len'] = str(len(df_train))
        df_metrics['train_set(s)'] = df_metrics_train_set_string[:-1]
        df_metrics['dev_set(s)'] = df_metrics_dev_set_string[:-1]
        df_metrics['test_set(s)'] = df_metrics_test_set_string[:-1]
        df_metrics['run_id'] = metrics_id

        # make unique filepath
        metrics_filename = str(metrics_id)+'_' + \
            model_name_simple+'_'+time_str+'_metrics.csv'
        metrics_csv_filepath = os.path.join(
            metrics_save_path_model, metrics_filename)
        print(metrics_csv_filepath)

        # Save test metrics to CSV
        if not os.path.exists(metrics_csv_filepath):
            df_metrics.to_csv(metrics_csv_filepath, index=False)
        else:
            df_metrics.to_csv(metrics_csv_filepath,
                            mode='a', header=False, index=False)

        return val_metrics

    def log(self, logs: Dict[str, float]):
        # Call the original `log` method to preserve its functionality
        super().log(logs)

        # Calculate total steps
        total_steps = len(
            self.train_dataset) * self.args.num_train_epochs // self.args.per_device_train_batch_size
        if self.args.world_size > 1:
            total_steps = total_steps // self.args.world_size

        # Calculate the percentage of completed steps
        progress_percentage = 100 * self.state.global_step / total_steps

        # Print the custom message
        print("Global step:", self.state.global_step)
        print(
            f"Progress: {progress_percentage:.2f}% steps completed ({self.state.global_step}/{total_steps})")
        print(f"Current model: {model_name_simple}")
        print(f"Current run id: {metrics_id}")

# Define training arguments
training_args = TrainingArguments(
    output_dir='./results',           # Output directory for model and predictions
    num_train_epochs=epochs,          # Number of epochs
    # Batch size per device during training
    per_device_train_batch_size=batch_size,
    # Batch size per device during evaluation
    per_device_eval_batch_size=batch_size,
    warmup_steps=0,                  # Linear warmup over warmup_steps
    weight_decay=0.01,               # Weight decay
    logging_dir='./logs',            # Directory for storing logs
    logging_steps=100,               # Log every X updates steps
    evaluation_strategy='epoch',     # Evaluate every epoch
    save_strategy='no',              # Do not save checkpoint after each epoch
    # load_best_model_at_end=True,     # Load the best model when finished training (best on dev set)
    metric_for_best_model='f1',      # Use f1 score to determine the best model
    greater_is_better=True,           # The higher the f1 score, the better
)

# define optimizer
optimizer = AdamW(
    model.parameters(),
    lr=2e-5,
    eps=1e-8,
)

# instantiate trainer
trainer = CustomTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    compute_metrics=compute_metrics,
    # pass the new optimizer to the trainer
    optimizers=(optimizer, None),
)

model_path = '/home/pgajo/working/pt_models'
model_name_ft = model_name_simple + '_' + 'finetuned' + metrics_path_category.split('/')[-1] + '_' + 'metrics_id_' + str(metrics_id)
model_save_path = os.path.join(model_path, model_name_ft)

if not os.path.exists(model_save_path):
    os.makedirs(model_save_path)

print('###################################')
print('Saving model to: ', model_save_path)
print('###################################')

# Train the model
trainer.train()

model.save_pretrained(model_save_path)
tokenizer.save_pretrained(model_save_path)
