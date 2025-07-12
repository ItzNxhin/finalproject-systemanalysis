import pandas as pd
import numpy as np
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification
from transformers import Trainer, TrainingArguments
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
import torch

# Load data
train_labels = pd.read_csv('train_labels.csv')
train_tracking = pd.read_csv('train_player_tracking.csv')
test_tracking = pd.read_csv('test_player_tracking.csv')
sample_submission = pd.read_csv('sample_submission.csv')

# Set index for fast lookup
train_tracking.set_index(['game_key', 'play_id', 'step', 'nfl_player_id'], inplace=True)
test_tracking.set_index(['game_key', 'play_id', 'step', 'nfl_player_id'], inplace=True)

# Function to parse contact_id
def parse_contact_id(contact_id):
    parts = contact_id.split('_')
    game_key = parts[0]
    play_id = parts[1]
    step = int(parts[2])
    player1 = parts[3]
    player2 = parts[4]
    return game_key, play_id, step, player1, player2

# Function to generate textual description
def generate_text(row, tracking_df):
    game_key, play_id, step, player1, player2 = parse_contact_id(row['contact_id'])
    if player2 == 'G':
        try:
            p1_data = tracking_df.loc[(game_key, play_id, step, player1)]
            text = f"Player {player1} at ({p1_data['x_position']},{p1_data['y_position']}) with speed {p1_data['speed']}, contacting ground."
        except KeyError:
            text = "Missing data for player"
    else:
        try:
            p1_data = tracking_df.loc[(game_key, play_id, step, player1)]
            p2_data = tracking_df.loc[(game_key, play_id, step, player2)]
            distance = np.sqrt((p1_data['x_position'] - p2_data['x_position'])*2 + (p1_data['y_position'] - p2_data['y_position'])*2)
            text = f"Player {player1} at ({p1_data['x_position']},{p1_data['y_position']}) with speed {p1_data['speed']}, Player {player2} at ({p2_data['x_position']},{p2_data['y_position']}) with speed {p2_data['speed']}, distance {distance}."
        except KeyError:
            text = "Missing data for players"
    return text

# Generate textual descriptions for training set
train_labels['text'] = train_labels.apply(lambda row: generate_text(row, train_tracking), axis=1)

# Split into training and validation sets
train_df, val_df = train_test_split(train_labels[['text', 'contact']], test_size=0.2, random_state=42)

# Load DistilBERT tokenizer
tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')

# Define dataset class
class ContactDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length):
        self.encodings = tokenizer(texts, truncation=True, padding=True, max_length=max_length)
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

# Create datasets
max_length = 128
train_dataset = ContactDataset(train_df['text'].tolist(), train_df['contact'].tolist(), tokenizer, max_length)
val_dataset = ContactDataset(val_df['text'].tolist(), val_df['contact'].tolist(), tokenizer, max_length)

# Load DistilBERT model for classification
model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels=2)

# Set training arguments
training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=3,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir='./logs',
    logging_steps=10,
    evaluation_strategy="epoch",
)

# Create trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
)

# Train model
trainer.train()

# Generate textual descriptions for test set

sample_submission['text'] = sample_submission.apply(lambda row: generate_text(row, test_tracking), axis=1)

# Create test dataset
test_dataset = ContactDataset(sample_submission['text'].tolist(), [0]*len(sample_submission), tokenizer, max_length)

#Predict
predictions = trainer.predict(test_dataset)
logits = predictions.predictions
probs = torch.nn.functional.softmax(torch.tensor(logits), dim=1)
contact_probs = probs[:, 1].numpy()

# Save submission
sample_submission['contact'] = contact_probs
sample_submission[['contact_id', 'contact']].to_csv('submission.csv', index=False)
