import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score

# Load data files
train_labels = pd.read_csv("train_labels.csv")  # training labels (contact events, 'G' denotes ground):contentReference[oaicite:0]{index=0}
train_tracking = pd.read_csv("train_player_tracking.csv")  # player tracking (positions, speed, acceleration, etc.):contentReference[oaicite:1]{index=1}

# Prepare training data: merge tracking features for player1 and player2
# Replace ground indicator 'G' with dummy ID (0) to allow merging
train_labels['nfl_player_id_2'] = train_labels['nfl_player_id_2'].replace('G', '0').astype(int)
train_labels['nfl_player_id_1'] = train_labels['nfl_player_id_1'].astype(int)
train_labels['step'] = train_labels['step'].astype(int)

# Merge player 1 tracking data
player1_track = train_tracking.rename(columns={
    'nfl_player_id': 'nfl_player_id_1',
    'x_position': 'x_position_1', 'y_position': 'y_position_1',
    'speed': 'speed_1', 'acceleration': 'acceleration_1'
})
train_df = pd.merge(train_labels,
                    player1_track[['game_play','step','nfl_player_id_1','x_position_1','y_position_1','speed_1','acceleration_1']],
                    on=['game_play','step','nfl_player_id_1'],
                    how='left')

# Merge player 2 tracking data
player2_track = train_tracking.rename(columns={
    'nfl_player_id': 'nfl_player_id_2',
    'x_position': 'x_position_2', 'y_position': 'y_position_2',
    'speed': 'speed_2', 'acceleration': 'acceleration_2'
})
train_df = pd.merge(train_df,
                    player2_track[['game_play','step','nfl_player_id_2','x_position_2','y_position_2','speed_2','acceleration_2']],
                    on=['game_play','step','nfl_player_id_2'],
                    how='left')

# Fill missing values (for ground or missing tracking) with 0
train_df[['x_position_1','y_position_1','speed_1','acceleration_1',
          'x_position_2','y_position_2','speed_2','acceleration_2']] = \
    train_df[['x_position_1','y_position_1','speed_1','acceleration_1',
              'x_position_2','y_position_2','speed_2','acceleration_2']].fillna(0)

# Feature engineering: relative distances and kinematic differences:contentReference[oaicite:2]{index=2}:contentReference[oaicite:3]{index=3}
train_df['dx'] = train_df['x_position_1'] - train_df['x_position_2']
train_df['dy'] = train_df['y_position_1'] - train_df['y_position_2']
train_df['distance'] = np.sqrt(train_df['dx']**2 + train_df['dy']**2)
train_df['speed_diff'] = train_df['speed_1'] - train_df['speed_2']
train_df['acc_diff'] = train_df['acceleration_1'] - train_df['acceleration_2']

# Select features and target
feature_cols = ['dx','dy','distance','speed_1','speed_2','speed_diff','acceleration_1','acceleration_2','acc_diff']
X = train_df[feature_cols]
y = train_df['contact']

# Split into 80% train, 20% validation
X_train, X_val, y_train, y_val = train_test_split(X, y,
                                                  test_size=0.2,
                                                  random_state=42,
                                                  stratify=y)

# Train RandomForest with class weights to handle imbalance
clf = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42)
clf.fit(X_train, y_train)

# Evaluate on validation set
y_val_pred = clf.predict(X_val)
acc = accuracy_score(y_val, y_val_pred)
prec = precision_score(y_val, y_val_pred)
rec = recall_score(y_val, y_val_pred)
print(f"Validation Accuracy: {acc:.4f}")
print(f"Validation Precision: {prec:.4f}")
print(f"Validation Recall: {rec:.4f}")

# Prepare test data features
sample_sub = pd.read_csv("sample_submission.csv")
test_tracking = pd.read_csv("test_player_tracking.csv")

# Parse contact_id into game_play, step, and player IDs
sample_sub[['game_key','play_id','step','nfl_player_id_1','nfl_player_id_2']] = \
    sample_sub['contact_id'].str.split('_', expand=True)
sample_sub['game_play'] = sample_sub['game_key'] + "_" + sample_sub['play_id']
sample_sub['step'] = sample_sub['step'].astype(int)
sample_sub['nfl_player_id_1'] = sample_sub['nfl_player_id_1'].astype(int)
sample_sub['nfl_player_id_2'] = sample_sub['nfl_player_id_2'].replace('G', '0').astype(int)

# Merge test tracking for player 1
test_df = pd.merge(sample_sub,
                   test_tracking.rename(columns={'nfl_player_id':'nfl_player_id_1'})[
                       ['game_play','step','nfl_player_id_1','x_position','y_position','speed','acceleration']],
                   on=['game_play','step','nfl_player_id_1'],
                   how='left')
test_df.rename(columns={'x_position':'x_position_1','y_position':'y_position_1',
                        'speed':'speed_1','acceleration':'acceleration_1'},
               inplace=True)

# Merge test tracking for player 2
test_df = pd.merge(test_df,
                   test_tracking.rename(columns={'nfl_player_id':'nfl_player_id_2'})[
                       ['game_play','step','nfl_player_id_2','x_position','y_position','speed','acceleration']],
                   on=['game_play','step','nfl_player_id_2'],
                   how='left')
test_df.rename(columns={'x_position':'x_position_2','y_position':'y_position_2',
                        'speed':'speed_2','acceleration':'acceleration_2'},
               inplace=True)

# Fill missing values for test (ground or missing tracking) with 0
test_df[['x_position_1','y_position_1','speed_1','acceleration_1',
         'x_position_2','y_position_2','speed_2','acceleration_2']] = \
    test_df[['x_position_1','y_position_1','speed_1','acceleration_1',
             'x_position_2','y_position_2','speed_2','acceleration_2']].fillna(0)

# Compute test features
test_df['dx'] = test_df['x_position_1'] - test_df['x_position_2']
test_df['dy'] = test_df['y_position_1'] - test_df['y_position_2']
test_df['distance'] = np.sqrt(test_df['dx']**2 + test_df['dy']**2)
test_df['speed_diff'] = test_df['speed_1'] - test_df['speed_2']
test_df['acc_diff'] = test_df['acceleration_1'] - test_df['acceleration_2']

X_test = test_df[feature_cols]

# Predict probabilities and prepare submission
test_probs = clf.predict_proba(X_test)[:, 1]
sample_sub['contact'] = test_probs
sample_sub[['contact_id','contact']].to_csv('submission.csv', index=False)
