# Import necessary libraries
import json
import ast
from collections import Counter

import matplotlib.pyplot as plt
import numpy as np
import optuna
import pandas as pd
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report, confusion_matrix,  accuracy_score
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier

from sklearn.ensemble import (
    RandomForestClassifier, 
    GradientBoostingClassifier
)
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import StackingClassifier

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


# Optuna-integrated function to process event sequences into TF-IDF features and merge with the original DataFrame
def process_event_sequences_to_tfidf_and_merge(
    df, column_name, max_df=0.9841170198383609, min_df=9
):
    """
    Transforms a specified column of event sequences into TF-IDF features and merges these features
    back into the original DataFrame. This function is intended for use with Optuna for hyperparameter optimization.

    Parameters:
    df (pd.DataFrame): The input DataFrame containing event sequences in the specified column.
    column_name (str): The name of the column to be processed (event sequences).
    max_df (float, optional): The maximum document frequency for the TF-IDF Vectorizer.
                              Terms appearing in more than this proportion of documents will be ignored.
                              Default is 0.9841170198383609.
    min_df (int, optional): The minimum document frequency for the TF-IDF Vectorizer.
                            Terms appearing in fewer than this number of documents will be ignored.
                            Default is 9.

    Returns:
    pd.DataFrame: The updated DataFrame with TF-IDF features merged into it.
    """
    # Convert the event sequences in the column to strings, where each sequence is joined by spaces
    docs = df[column_name].apply(lambda x: " ".join(str(w) for w in x))

    # Initialize the TF-IDF Vectorizer with specified max_df and min_df parameters
    vectorizer = TfidfVectorizer(max_df=max_df, min_df=min_df)

    # Fit the vectorizer to the document strings and transform them into a sparse matrix
    tfidf = vectorizer.fit_transform(docs)

    # Convert the resulting sparse matrix into a DataFrame, preserving column names and index
    tfidf_df = pd.DataFrame.sparse.from_spmatrix(
        tfidf, columns=vectorizer.get_feature_names_out(), index=df.index
    )

    # Merge the original DataFrame with the new TF-IDF DataFrame along the columns
    merged_df = pd.concat([df, tfidf_df], axis=1)

    return merged_df

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

    # Apply TF-IDF processing to the 'events_sequence' column and merge the results with the DataFrame
    df = process_event_sequences_to_tfidf_and_merge(df, column_name="events_sequence")

    return df


# Optuna objective function for hyperparameter tuning
# Improved Optuna objective function with cross-validation and early stopping
def objective(trial, X_train, y_train):
    """
    Defines the objective function for Optuna to optimize hyperparameters of an XGBoost classifier.

    Parameters:
    trial (optuna.trial.Trial): A trial object from Optuna, used to suggest hyperparameters.
    X_train (pd.DataFrame or np.array): The training feature set.
    y_train (pd.Series or np.array): The training target labels.

    Returns:
    float: The mean F1-weighted score from cross-validation, which is the metric to optimize.
    """

    # Hyperparameters to optimize
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 50, 300),
        'learning_rate': trial.suggest_loguniform('learning_rate', 0.01, 0.5),
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
        'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
        'subsample': trial.suggest_uniform('subsample', 0.5, 1.0),
        'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2', None])
    }

    # Initialize the Gradient Boosting classifier with the suggested parameters
    model = GradientBoostingClassifier(
        random_state=42,
        **params
    )

    # Perform stratified 5-fold cross-validation to evaluate the model
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_val_score(
        model, X_train, y_train, scoring="f1_weighted", cv=skf, n_jobs=-1
    )

    return np.mean(scores)


def classification_with_gradient_boosting(X, y):
    # Encode labels
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.2, random_state=42
    )
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Calculate class weights
    class_counts = np.bincount(y_train)
    total_samples = len(y_train)
    class_weights = {i: total_samples / (len(class_counts) * count) for i, count in enumerate(class_counts)}

     # Optimize hyperparameters using Optuna
    study = optuna.create_study(direction="maximize")
    study.optimize(lambda trial: objective(trial, X_train, y_train), n_trials=30)
    best_params = study.best_params

    # Print the best parameters
    print(f"Best hyperparameters: {best_params}")

    
    # Classifier with class weights
    clf = GradientBoostingClassifier( 
        random_state=42,
        **best_params,
    )
    
    # Fit with class weights
    sample_weights = np.array([class_weights[y] for y in y_train])
    clf.fit(X_train_scaled, y_train, sample_weight=sample_weights)
    
    # Predictions
    predictions = clf.predict(X_test_scaled)

     # Plot the confusion matrix
    cm = confusion_matrix(y_test, predictions)
    plt.figure(figsize=(12, 9))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="magma",
        xticklabels=le.classes_,
        yticklabels=le.classes_,
    )
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.show()
    
    return {
        'model': clf,
        'predictions': predictions,
        'class_weights': class_weights,
        'confusion_matrix': confusion_matrix(y_test, predictions),
        'classification_report': classification_report(y_test, predictions)
    }



def main():
    # Main script execution
    file_path = "sncb_data_challenge.csv"  # Adjust this path as needed
    df = pd.read_csv(file_path, sep=";", index_col=0)

    # Preprocess data
    df = preprocess_data(df)

    # Perform feature engineering
    df = feature_engineering(df)

    # Filter for events with more than 30 obs
    # df = df.loc[df["incident_type"].isin([2, 4, 9, 13, 14, 99])]

    # Train and evaluate models
    y = df["incident_type"]
    X = df.loc[:, "vehicle_count":]

        
        # Run classification
    results = classification_with_gradient_boosting(X, y)
        
    print("Class Weights:", results['class_weights'])
    print("\nConfusion Matrix:\n", results['confusion_matrix'])
    print("\nClassification Report:\n", results['classification_report'])

if __name__ == "__main__":
    main()
