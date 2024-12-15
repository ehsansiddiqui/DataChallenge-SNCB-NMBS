#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import optuna
import pandas as pd
from optuna.samplers import TPESampler
from sklearn.metrics import classification_report, f1_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier


class FeatureSelectionOptuna:
    """
    This class implements feature selection using Optuna optimization framework.

    Parameters:

    - model (object): The predictive model to evaluate; this should be any object that implements fit() and predict() methods.
    - loss_fn (function): The loss function to use for evaluating the model performance. This function should take the true labels and the
                          predictions as inputs and return a loss value.
    - features (list of str): A list containing the names of all possible features that can be selected for the model.
    - X (DataFrame): The complete set of feature data (pandas DataFrame) from which subsets will be selected for training the model.
    - y (Series): The target variable associated with the X data (pandas Series).
    - splits (list of tuples): A list of tuples where each tuple contains two elements, the train indices and the validation indices.
    - penalty (float, optional): A factor used to penalize the objective function based on the number of features used.
    """

    def __init__(self, model, loss_fn, features, X, y, splits, penalty=0):
        self.model = model
        self.loss_fn = loss_fn
        self.features = features
        self.X = X
        self.y = y
        self.splits = splits
        self.penalty = penalty

    def __call__(self, trial: optuna.trial.Trial):
        # Select True / False for each feature
        selected_features = [
            trial.suggest_categorical(name, [True, False]) for name in self.features
        ]

        # List with names of selected features
        selected_feature_names = [
            name for name, selected in zip(self.features, selected_features) if selected
        ]

        # Optional: adds a penalty for the amount of features used
        n_used = len(selected_feature_names)
        total_penalty = n_used * self.penalty

        loss = 0

        for split in self.splits:
            train_idx = split[0]
            valid_idx = split[1]

            X_train = self.X.iloc[train_idx].copy()
            y_train = self.y.iloc[train_idx].copy()
            X_valid = self.X.iloc[valid_idx].copy()
            y_valid = self.y.iloc[valid_idx].copy()

            X_train_selected = X_train[selected_feature_names].copy()
            X_valid_selected = X_valid[selected_feature_names].copy()

            # Train model, get predictions and accumulate loss
            self.model.fit(X_train_selected, y_train)
            pred = self.model.predict(X_valid_selected)

            loss += self.loss_fn(y_valid, pred)

        # Take the average loss across all splits
        loss /= len(self.splits)

        # Add the penalty to the loss
        loss += total_penalty

        return loss


SEED = 32


# Data preprocessing function
def preprocess_data(df):
    """
    Preprocesses a DataFrame by applying transformations to specific columns.
    - Converts stringified sequences into Python lists of integers or floats.
    - Converts approximate latitude and longitude values into integers.
    - Converts boolean sequences into binary values (0 or 1).
    - Drops unnecessary columns after processing.

    Parameters:
    df (pd.DataFrame): The input DataFrame to be processed.

    Returns:
    pd.DataFrame: The processed DataFrame with transformed columns.
    """
    # Convert stringified sequences to lists of integers or floats using eval
    df["vehicles_sequence"] = df["vehicles_sequence"].apply(eval)
    df["events_sequence"] = df["events_sequence"].apply(eval)
    df["seconds_to_incident_sequence"] = df["seconds_to_incident_sequence"].apply(eval)
    df["train_kph_sequence"] = df["train_kph_sequence"].apply(eval)

    # Convert approx_lat and approx_lon to integers (handling potential NaN values)
    df["approx_lat"] = (
        pd.to_numeric(df["approx_lat"], errors="coerce").fillna(np.nan).astype(int)
    )
    df["approx_lon"] = (
        pd.to_numeric(df["approx_lon"], errors="coerce").fillna(np.nan).astype(int)
    )

    # Convert boolean sequences (represented as strings) into lists of 0s and 1s
    df["dj_ac_state_sequence"] = df["dj_ac_state_sequence"].apply(
        lambda x: [1 if val else 0 for val in eval(x)]
    )
    df["dj_dc_state_sequence"] = df["dj_dc_state_sequence"].apply(
        lambda x: [1 if val else 0 for val in eval(x)]
    )

    # Drop the approx_lat and approx_lon columns after processing
    df.drop(columns=["approx_lat", "approx_lon"])

    return df


# Function to process the 'seconds_to_incident_sequence' column
def process_seconds_sequence(data):
    """
    Processes the 'seconds_to_incident_sequence' column in a DataFrame by extracting numerical
    sequences, calculating summary statistics, and performing time gap analysis.

    Parameters:
    data (pd.DataFrame): The input DataFrame containing the 'seconds_to_incident_sequence' column.

    Returns:
    pd.DataFrame: The updated DataFrame with additional columns for summary statistics
                  and time gap analysis.
    """
    # Convert the 'seconds_to_incident_sequence' column from string format to lists of floats
    seconds_sequence = (
        data["seconds_to_incident_sequence"]
        .astype(str)
        .apply(
            lambda x: [float(i) for i in x.replace("[", "").replace("]", "").split(",")]
        )
    )

    # Calculate summary statistics for the sequences
    data["seconds_mean"] = seconds_sequence.apply(np.mean)
    data["seconds_std"] = seconds_sequence.apply(np.std)
    data["seconds_max"] = seconds_sequence.apply(np.max)
    data["seconds_min"] = seconds_sequence.apply(np.min)
    data["seconds_sum"] = seconds_sequence.apply(np.sum)

    # Perform time gap analysis by calculating differences between consecutive elements
    data["seconds_gap_mean"] = seconds_sequence.apply(
        lambda x: np.mean(np.diff(x)) if len(x) > 1 else 0
    )
    data["seconds_gap_std"] = seconds_sequence.apply(
        lambda x: np.std(np.diff(x)) if len(x) > 1 else 0
    )

    return data


# Feature engineering function
def feature_engineering(df):
    """
    Performs feature engineering on the input DataFrame by generating new features
    from existing columns related to vehicles, events, speed, and incident duration.
    Additional processing includes integrating statistical features and TF-IDF transformation.

    Parameters:
    df (pd.DataFrame): The input DataFrame containing sequences and incident-related data.

    Returns:
    pd.DataFrame: The updated DataFrame with new engineered features and merged TF-IDF columns.
    """

    # Compute the total number of vehicles involved in each incident
    df["vehicle_count"] = df["vehicles_sequence"].apply(len)

    # Compute the number of unique vehicles involved in each incident
    df["vehicles_unique_count"] = df["vehicles_sequence"].apply(lambda x: len(set(x)))

    # Compute the total number of events associated with each incident
    df["event_count"] = df["events_sequence"].apply(len)

    # Compute the number of unique events associated with each incident
    df["events_unique_count"] = df["events_sequence"].apply(lambda x: len(set(x)))

    # Calculate the mean speed from the train's speed sequence
    df["mean_speed"] = df["train_kph_sequence"].apply(np.mean)

    # Calculate the total duration of the incident from the sequence of timestamps
    df["incident_duration"] = df["seconds_to_incident_sequence"].apply(
        lambda x: max(x) - min(x) if len(x) > 0 else 0
    )

    # Process the 'seconds_to_incident_sequence' column to extract statistical features
    df = process_seconds_sequence(df)

    # Calculate additional statistical features for the train's speed sequence
    df["max_speed"] = df["train_kph_sequence"].apply(np.max)
    df["min_speed"] = df["train_kph_sequence"].apply(np.min)
    df["std_speed"] = df["train_kph_sequence"].apply(np.std)

    # Calculate the average number of events per vehicle involved in the incident
    df["event_count_per_vehicle"] = df["event_count"] / df["vehicle_count"]

    # Calculate the average incident duration per vehicle involved
    df["incident_duration_per_vehicle"] = df["incident_duration"] / df["vehicle_count"]

    # Compute the difference between the maximum and minimum speed for incidents with multiple vehicles
    df["max_speed_diff"] = df.apply(
        lambda row: (
            np.max(row["train_kph_sequence"]) - np.min(row["train_kph_sequence"])
        )
        if len(row["vehicles_sequence"]) > 1
        else 0,
        axis=1,
    )

    return df


# Load data
# Main script execution
file_path = "sncb_data_challenge.csv"  # Adjust this path as needed
df = pd.read_csv(file_path, sep=";", index_col=0)

# Preprocess data
df = preprocess_data(df)

# Perform feature engineering
df = feature_engineering(df)

# Train - test split
y = df.loc[df["incident_type"].isin([2, 4, 9, 13, 14, 99])]["incident_type"]
X = df.loc[df["incident_type"].isin([2, 4, 9, 13, 14, 99]), "vehicle_count":]

# Encode target labels
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42
)

y_train = pd.Series(y_train)
y_test = pd.Series(y_test)

# Stratified K-Fold
SEED = 32
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
splits = list(skf.split(X_train, y_train))

# Define the model with specific hyperparameters for XGBClassifier
model = XGBClassifier(
    n_estimators=660,  # number of trees
    max_depth=15,  # maximum depth of each tree
    learning_rate=0.04698734918417903,  # learning rate (shrinkage)
    subsample=0.8502600673011051,  # fraction of samples to use for fitting each tree
    colsample_bytree=0.6898905947681309,  # fraction of features to use for each tree
    gamma=0.1,  # regularization parameter to control overfitting
    reg_alpha=1.3390663337605688,
    reg_lambda=1.3121669381733938,
    min_child_weight=2,
    random_state=SEED,
)


model.fit(X_train, y_train)
preds = model.predict(X_test)

print(classification_report(y_test, preds))
print(f"Global F1: {f1_score(y_test, preds, average='weighted')}")


def loss_fn(y_true, y_pred):
    """
    Returns the negative F1 score, to be treated as a loss function.
    """
    res = -f1_score(y_true, y_pred, average="weighted")

    return res


features = list(X_train.columns)

sampler = TPESampler(seed=SEED)
study = optuna.create_study(direction="minimize", sampler=sampler)

# We first try the model using all features
default_features = {ft: True for ft in features}
study.enqueue_trial(default_features)

study.optimize(
    FeatureSelectionOptuna(
        model=model,
        loss_fn=loss_fn,
        features=features,
        X=X_train,
        y=y_train,
        splits=splits,
        penalty=1e-4,
    ),
    n_trials=100,
)

selected_features = study.best_params
selected_features = [ft for ft in selected_features.keys() if selected_features[ft]]
selected_features

# We train the RSF using only the selected features

X_train_selected = X_train[selected_features].copy()
X_test_selected = X_test[selected_features].copy()

model.fit(X_train_selected, y_train)
preds = model.predict(X_test_selected)

print(classification_report(y_test, preds))
print(f"Global F1: {f1_score(y_test, preds, average='weighted')}")
