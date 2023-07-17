# Load dependencies
# used to make train/dev/test partitions
from sklearn.model_selection import train_test_split
from typing import Dict
from sklearn.metrics import precision_recall_fscore_support
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import Dataset
from transformers import Trainer, TrainingArguments
from transformers import BertForSequenceClassification, BertConfig, BertTokenizer
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
from pgfuncs import tokenize_and_vectorize, pad_trunc, collect_expected, tokenize_and_vectorize_1dlist, collect_expected_1dlist, df_classification_report
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

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
# load IFC-22-EN_updated_hs_scores_223k dataset
df = pd.read_csv('/home/pgajo/working/data/datasets/Italian/Il_forum_dei_brutti/IFC-22-IT_updated.csv_hs_scores_30k.csv')
df = df.fillna('') # [:100]
# df['hs_score'] = df['hs_score'].astype(float)
df['hs_score'] = df['hs_score'] * 100
# Split the dataset into train/dev/test
# Split the data into training and test sets (70% for training, 30% for test)
df_train, df_test = train_test_split(df, test_size=0.3, random_state=42)

# Split the test data into validation and test sets (50% for validation, 50% for test)
df_dev, df_test = train_test_split(df_test, test_size=0.5, random_state=42)

# Print the size of each split
print('Train set size:', len(df_train))
print('Dev set size:', len(df_dev))
print('Test set size:', len(df_test))

# %% Model choice
model_name = '/home/pgajo/working/pt_models/incel-bert-base-multilingual-cased-1000k_multi'

# Filename bits
# metrics_path_category = '/home/pgajo/working/data/metrics/1_hate_speech'
# metrics_path_category = '/home/pgajo/working/data/metrics/2_1_misogyny'
# metrics_path_category = '/home/pgajo/working/data/metrics/2_2_racism'
metrics_path_category = '/home/pgajo/working/data/metrics/3_hate_forecasting'

metrics_save_path = f'{metrics_path_category}/metrics_multilingual/'
# metrics_save_path = f'{metrics_path_category}/metrics_monolingual/'

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
metrics_filename = model_name_simple+'_'+time_str+'_metrics.csv'
metrics_csv_filepath = os.path.join(metrics_save_path_model, metrics_filename)
print(metrics_csv_filepath)

# get tokenizer and model
tokenizer = BertTokenizer.from_pretrained(model_name)

# Load the pre-trained model with a specific configuration
config = BertConfig.from_pretrained(model_name)

# Change the number of labels to 1 for regression
config.num_labels = 1

# Create the model with the modified configuration
model = BertForSequenceClassification.from_pretrained(model_name, config=config)

# Remove the activation function from the final layer
model.classifier = nn.Linear(model.classifier.in_features, 1)

# Data pre-processing
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
labels_train = torch.tensor(df_train.hs_score.values)
# Extract IDs, attention masks and labels from validation dataset
input_ids_val = encoded_data_val['input_ids']
attention_masks_val = encoded_data_val['attention_mask']
labels_val = torch.tensor(df_dev.hs_score.values)
# Extract IDs, attention masks and labels from test dataset
input_ids_test = encoded_data_test['input_ids']
attention_masks_test = encoded_data_test['attention_mask']
labels_test = torch.tensor(df_test.hs_score.values)

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

mean_baseline_val = df_dev.hs_score.mean()
mean_baseline_test = df_test.hs_score.mean()

baseline_array_val = np.zeros((len(labels_val), 1))
baseline_array_val.fill(mean_baseline_val)
baseline_array_test = np.zeros((len(labels_test), 1))
baseline_array_test.fill(mean_baseline_test)

mse_val_mean_baseline = mean_squared_error(labels_val, baseline_array_val)
print('mse_val', mse_val_mean_baseline)
mse_test_mean_baseline = mean_squared_error(labels_test, baseline_array_test)
print('mse_test', mse_test_mean_baseline)

mae_val_mean_baseline = mean_absolute_error(labels_val, baseline_array_val)
print('mae_val', mae_val_mean_baseline)
mae_test_mean_baseline = mean_absolute_error(labels_test, baseline_array_test)
print('mae_test', mae_test_mean_baseline)


# Adjust the custom compute metrics function for regression
def compute_metrics(eval_pred, metric_key_prefix="eval"):
    predictions, labels = eval_pred
    mse = mean_squared_error(labels, predictions)
    mae = mean_absolute_error(labels, predictions)
    r2 = r2_score(labels, predictions)
    return {
        f'{metric_key_prefix}_mse': mse,
        f'{metric_key_prefix}_mae': mae,
        f'{metric_key_prefix}_r2': r2,
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
            (val_output.predictions.squeeze(), val_output.label_ids), metric_key_prefix="val")
        test_metrics = compute_metrics(
            (test_output.predictions.squeeze(), test_output.label_ids), metric_key_prefix="test")
        df_metrics = pd.DataFrame(columns=[
                                'epoch', 'val_mse', 'val_mae', 'val_r2', 'test_mse', 'test_mae', 'test_r2'])
        if self.state.epoch == None:
            current_epoch = -1
        else:
            current_epoch = self.state.epoch
        df_metrics = df_metrics.append({
            'epoch': current_epoch, # self.state.epoch,
            'val_mse': val_metrics['val_mse'],
            'val_mae': val_metrics['val_mae'],
            'val_r2': val_metrics['val_r2'],
            'test_mse': test_metrics['test_mse'],
            'test_mae': test_metrics['test_mae'],
            'test_r2': test_metrics['test_r2'],
            'mse_val_mean_baseline': mse_val_mean_baseline,
            'mse_test_mean_baseline': mse_test_mean_baseline,
            'val_mse_baseline_diff_perc': val_metrics['val_mse']/mse_val_mean_baseline,
            'test_mse_baseline_diff_perc': test_metrics['test_mse']/mse_test_mean_baseline,
            'mae_val_mean_baseline': mae_val_mean_baseline,
            'mae_test_mean_baseline': mae_test_mean_baseline,
            'val_mae_baseline_diff_perc': val_metrics['val_mae']/mae_val_mean_baseline,
            'test_mae_baseline_diff_perc': test_metrics['test_mae']/mae_test_mean_baseline,
        }, ignore_index=True)

        df_metrics['model'] = model_name_simple
        df_metrics['train_len'] = str(len(df_train))
        df_metrics['train_set(s)'] = df_metrics_train_set_string[:-1]
        df_metrics['dev_set(s)'] = df_metrics_dev_set_string[:-1]
        df_metrics['test_set(s)'] = df_metrics_test_set_string[:-1]
        df_metrics['run_id'] = 'scores'

        # make unique filepath
        metrics_filename = model_name_simple+'_'+time_str+'_metrics.csv'
        metrics_csv_filepath = os.path.join(metrics_save_path_model, metrics_filename)
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
    
    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.get("labels").float()
        outputs = model(**inputs)
        logits = outputs.get('logits')
        loss_fct = nn.MSELoss()
        loss = loss_fct(logits.squeeze(), labels.squeeze())
        return (loss, outputs) if return_outputs else loss

# Create a custom loss function for the Trainer
# mse_loss = nn.MSELoss()

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

# Instantiate trainer with the custom loss function
trainer = CustomTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    compute_metrics=compute_metrics,
    optimizers=(optimizer, None),
)

model_path = '/home/pgajo/working/pt_models'
model_name_ft = model_name_simple + '_' + 'finetuned_hs_score'
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