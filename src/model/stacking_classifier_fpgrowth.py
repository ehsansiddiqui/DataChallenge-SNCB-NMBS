# -*- coding: utf-8 -*-

# Import necessary libraries
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from mlxtend.frequent_patterns import fpgrowth
from mlxtend.preprocessing import TransactionEncoder
from sklearn.base import clone
from sklearn.ensemble import (
    GradientBoostingClassifier,
    RandomForestClassifier,
    StackingClassifier,
)
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    log_loss,
)
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

# Set-up plot style
sns.set_theme(style="whitegrid")  # Options: "darkgrid", "white", "ticks"
custom_palette = sns.color_palette(
    "Blues_r"
)  # Reverse Blues palette for a unified look
sns.set_palette(custom_palette)


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
    # data["seconds_min"] = seconds_sequence.apply(np.min)
    # data["seconds_sum"] = seconds_sequence.apply(np.sum)

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
    df["vehicle_unique_perc"] = df["vehicles_sequence"].apply(
        lambda x: len(set(x))
    ) / df["vehicles_sequence"].apply(len)

    # Compute the number of unique vehicles involved in each incident
    df["vehicles_unique_count"] = df["vehicles_sequence"].apply(lambda x: len(set(x)))

    # Compute the total number of events associated with each incident
    df["event_unique_perc"] = df["events_sequence"].apply(lambda x: len(set(x))) / df[
        "events_sequence"
    ].apply(len)

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
    # df["min_speed"] = df["train_kph_sequence"].apply(np.min)
    df["std_speed"] = df["train_kph_sequence"].apply(np.std)

    # Calculate the average incident duration per vehicle involved
    df["incident_duration_per_vehicle"] = df["incident_duration"] / df[
        "vehicles_sequence"
    ].apply(len)

    # Compute the difference between the maximum and minimum speed for incidents with multiple vehicles
    df["max_speed_diff"] = df.apply(
        lambda row: (
            np.max(row["train_kph_sequence"]) - np.min(row["train_kph_sequence"])
        )
        if len(row["vehicles_sequence"]) > 1
        else 0,
        axis=1,
    )

    # Power State Features from 'dj_ac_state_sequence' and 'dj_dc_state_sequence'
    df["ac_active_ratio"] = df["dj_ac_state_sequence"].apply(
        lambda x: sum(x) / len(x) if x else 0
    )
    df["dc_active_ratio"] = df["dj_dc_state_sequence"].apply(
        lambda x: sum(x) / len(x) if x else 0
    )

    # Calculate transitions (True to False or vice versa) for AC and DC power states
    df["ac_state_transitions"] = df["dj_ac_state_sequence"].apply(
        lambda x: sum([x[i] != x[i + 1] for i in range(len(x) - 1)])
    )
    df["dc_state_transitions"] = df["dj_dc_state_sequence"].apply(
        lambda x: sum([x[i] != x[i + 1] for i in range(len(x) - 1)])
    )

    return df


# Function to process event sequences into TF-IDF features and merge with the original DataFrame
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


# Discretize speed values into bins
def discretize_speed(speed):
    if speed < 30:
        return "low_speed"
    elif 30 <= speed < 60:
        return "medium_speed"
    else:
        return "high_speed"


# Preprocess the sequences to split them into before, during, and after time periods
# and combine events and discretized speed
def split_sequences_by_time_and_incident_type(data):
    split_data = defaultdict(lambda: defaultdict(list))

    for _, row in data.iterrows():
        # Split the sequences into before, during, and after periods
        before = [
            (event, discretize_speed(speed))
            for event, speed, time in zip(
                row["events_sequence"],
                row["train_kph_sequence"],
                row["seconds_to_incident_sequence"],
            )
            if time < -1000
        ]
        during = [
            (event, discretize_speed(speed))
            for event, speed, time in zip(
                row["events_sequence"],
                row["train_kph_sequence"],
                row["seconds_to_incident_sequence"],
            )
            if -1000 <= time <= 1000
        ]
        after = [
            (event, discretize_speed(speed))
            for event, speed, time in zip(
                row["events_sequence"],
                row["train_kph_sequence"],
                row["seconds_to_incident_sequence"],
            )
            if time > 1000
        ]

        # Combine event and speed into a single representation
        # Flatten and convert to sets (Apriori ignores order)
        split_data[row["incident_type"]]["before"].append(
            [f"{event}_{speed}" for event, speed in before]
        )
        split_data[row["incident_type"]]["during"].append(
            [f"{event}_{speed}" for event, speed in during]
        )
        split_data[row["incident_type"]]["after"].append(
            [f"{event}_{speed}" for event, speed in after]
        )

    return split_data


# Apply fpgrowth using TransactionEncoder
def apply_fpgrowth_to_sequences(sequences, min_support=0.1):
    # Use TransactionEncoder to convert the list of transactions into a one-hot-encoded DataFrame
    te = TransactionEncoder()
    te_array = te.fit(sequences).transform(sequences)
    one_hot_df = pd.DataFrame(te_array, columns=te.columns_)

    # Apply fpgrowth to find frequent patterns
    frequent_itemsets = fpgrowth(one_hot_df, min_support=min_support, use_colnames=True)
    return frequent_itemsets


# Extract patterns for each incident type and time period
def extract_patterns_by_incident_type(data, min_support=0.5):
    split_data = split_sequences_by_time_and_incident_type(data)
    patterns_by_incident_type = defaultdict(dict)

    for incident_type, sequences_by_time in split_data.items():
        for time_period, sequences in sequences_by_time.items():
            frequent_itemsets = apply_fpgrowth_to_sequences(
                sequences, min_support=min_support
            )
            patterns_by_incident_type[incident_type][time_period] = frequent_itemsets

    return patterns_by_incident_type


# Create features based on the identified patterns
def create_features_from_patterns(data, patterns_by_incident_type):
    feature_list = []

    for _, row in data.iterrows():
        incident_type = row["incident_type"]
        feature_row = {}

        # Retrieve patterns for the current incident type
        patterns_by_time = patterns_by_incident_type[incident_type]

        for time_period, itemsets in patterns_by_time.items():
            # Combine event and speed into the same representation
            event_speed_set = set(
                f"{event}_{discretize_speed(speed)}"
                for event, speed in zip(
                    row["events_sequence"], row["train_kph_sequence"]
                )
            )

            for _, itemset_row in itemsets.iterrows():
                itemset = set(itemset_row["itemsets"])
                # Check if the itemset is a subset of the combined event-speed sequence
                pattern_present = 1 if itemset.issubset(event_speed_set) else 0
                feature_row[f"{time_period}_pattern_{itemset}"] = pattern_present

        feature_list.append(feature_row)

    # Convert list of feature dictionaries to DataFrame
    new_features_df = pd.DataFrame(feature_list)
    # Fill NaN values with 0
    new_features_df = new_features_df.fillna(0)

    return pd.concat([data.reset_index(drop=True), new_features_df], axis=1)


def classification_report_to_dataframe(y_test, y_pred, labels):
    """
    Converts the classification report to a pandas DataFrame.

    Parameters:
    y_test (list or array): True labels.
    y_pred (list or array): Predicted labels.
    labels (list): List of class labels in the order they appear in the report.

    Returns:
    pd.DataFrame: A DataFrame representation of the classification report.
    """
    # Ensure the labels are strings
    labels = [str(label) for label in labels]

    # Get unique classes from y_test and y_pred
    unique_labels = sorted(set(y_test) | set(y_pred))

    # Generate the classification report as a dictionary
    report_dict = classification_report(
        y_test, y_pred, target_names=unique_labels, zero_division=1, output_dict=True
    )

    # Convert the dictionary to a DataFrame
    report_df = pd.DataFrame(report_dict).transpose()

    # Return the DataFrame
    return report_df


def stacking_classification(X, y):
    # Print the original label encoding map
    print("Original label encoding map:")
    class_labels = np.unique(y)
    label_map = {label: idx for idx, label in enumerate(class_labels)}
    print(label_map)

    # Convert to NumPy arrays if necessary
    if not isinstance(X, np.ndarray):
        X = X.values
    if not isinstance(y, np.ndarray):
        y = y.values

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Identify all possible classes in the dataset
    all_classes = np.unique(y)

    # Base classifiers
    base_classifiers = [
        (
            "rf",
            make_pipeline(StandardScaler(), RandomForestClassifier(n_estimators=100)),
        ),
        (
            "gb",
            make_pipeline(
                StandardScaler(), GradientBoostingClassifier(n_estimators=100)
            ),
        ),
        ("mlp", make_pipeline(StandardScaler(), MLPClassifier(max_iter=1000))),
    ]

    # Meta-classifier
    meta_classifier = LogisticRegression()

    # Stacking Classifier
    stacking_clf = StackingClassifier(
        estimators=base_classifiers,
        final_estimator=meta_classifier,
        cv=5,  # 5-fold cross-validation
    )

    # To track the learning curve (log loss)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    train_meta_log_loss = []
    val_meta_log_loss = []

    # Cross-validation for meta-classifier log loss tracking
    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X_train, y_train)):
        print(f"Training on Fold {fold_idx + 1}...")

        X_train_fold, X_val_fold = X_train[train_idx], X_train[val_idx]
        y_train_fold, y_val_fold = y_train[train_idx], y_train[val_idx]

        # Clone the model to avoid fitting on the entire training data each time
        stacking_clf_fold = clone(stacking_clf)

        # Train the stacking classifier on the current fold
        stacking_clf_fold.fit(X_train_fold, y_train_fold)

        # Get the base classifier outputs for train and validation sets
        train_base_probas = np.column_stack(
            [
                clf.predict_proba(X_train_fold)
                for name, clf in stacking_clf_fold.named_estimators_.items()
            ]
        )
        val_base_probas = np.column_stack(
            [
                clf.predict_proba(X_val_fold)
                for name, clf in stacking_clf_fold.named_estimators_.items()
            ]
        )

        # Get meta-classifier predictions
        train_meta_proba = stacking_clf_fold.final_estimator_.predict_proba(
            train_base_probas
        )
        val_meta_proba = stacking_clf_fold.final_estimator_.predict_proba(
            val_base_probas
        )

        # Ensure all classes are represented in the probabilities
        train_meta_proba = ensure_all_classes(train_meta_proba, all_classes)
        val_meta_proba = ensure_all_classes(val_meta_proba, all_classes)

        # Calculate log loss for the meta-classifier
        train_meta_log_loss.append(
            log_loss(y_train_fold, train_meta_proba, labels=all_classes)
        )
        val_meta_log_loss.append(
            log_loss(y_val_fold, val_meta_proba, labels=all_classes)
        )

    # Plot log loss learning curve for the meta-classifier
    plt.figure(figsize=(12, 6))
    plt.plot(
        range(1, len(train_meta_log_loss) + 1),
        train_meta_log_loss,
        label="Train Log Loss (Meta-Classifier)",
    )
    plt.plot(
        range(1, len(val_meta_log_loss) + 1),
        val_meta_log_loss,
        label="Validation Log Loss (Meta-Classifier)",
    )
    plt.xlabel("Fold")
    plt.ylabel("Log Loss")
    plt.title("Meta-Classifier Log Loss Learning Curve")
    plt.legend()
    plt.show()

    # Fit the Stacking Classifier on the full training set
    stacking_clf.fit(X_train, y_train)

    # Get the outputs of the base classifiers on the test set
    base_classifier_probas = np.column_stack(
        [
            clf.predict_proba(X_test)
            for name, clf in stacking_clf.named_estimators_.items()
        ]
    )

    # Use the trained meta-classifier to calculate log loss on the test set
    meta_predictions_proba = stacking_clf.final_estimator_.predict_proba(
        base_classifier_probas
    )
    meta_predictions_proba = ensure_all_classes(meta_predictions_proba, all_classes)

    # Calculate log loss for the meta-classifier on the test set
    meta_log_loss = log_loss(y_test, meta_predictions_proba, labels=all_classes)
    print(f"Meta-Classifier Log Loss (Test): {meta_log_loss}")

    # Calculate accuracy
    stacking_predictions = stacking_clf.predict(X_test)
    accuracy = accuracy_score(y_test, stacking_predictions)
    print(f"Accuracy: {accuracy}")

    # Classification report
    print("\nClassification Report:")
    print(classification_report(y_test, stacking_predictions, labels=all_classes))

    # Plot the confusion matrix
    cm = confusion_matrix(y_test, stacking_predictions, labels=all_classes)
    plt.figure(figsize=(12, 9))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=all_classes,
        yticklabels=all_classes,
    )
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.show()

    return {
        "stacking_model": stacking_clf,
        "predictions": stacking_predictions,
        "meta_log_loss": meta_log_loss,
        "accuracy": accuracy,
        "classification_report": classification_report(
            y_test, stacking_predictions, labels=all_classes
        ),
    }


def ensure_all_classes(pred_proba, all_classes):
    """
    Ensures that the predicted probabilities include all classes.
    Adds zero probabilities for missing classes if necessary.
    """
    num_samples = pred_proba.shape[0]
    num_classes = len(all_classes)

    if pred_proba.shape[1] < num_classes:
        # Add zero-probability columns for missing classes
        missing_classes = set(all_classes) - set(range(pred_proba.shape[1]))
        missing_classes = sorted(list(missing_classes))
        missing_class_probabilities = np.zeros((num_samples, len(missing_classes)))
        pred_proba = np.hstack([pred_proba, missing_class_probabilities])

    return pred_proba


def main():
    # Main script execution
    file_path = "sncb_data_challenge.csv"  # Adjust this path as needed
    df = pd.read_csv(file_path, sep=";", index_col=0)

    # Preprocess data
    df = preprocess_data(df)

    # Perform feature engineering
    df = feature_engineering(df)

    # Apply TF-IDF processing to the 'events_sequence' column and merge the results with the DataFrame
    df = process_event_sequences_to_tfidf_and_merge(df, column_name="events_sequence")

    # Apply the fpgrowth-based workflow
    patterns_by_incident_type = extract_patterns_by_incident_type(df, min_support=0.9)
    df = create_features_from_patterns(df, patterns_by_incident_type)

    # Train and evaluate models
    y = df["incident_type"]
    X = df.loc[:, "vehicle_unique_perc":]

    # Perform stacking classification
    results = stacking_classification(X, y)

    return df, results


if __name__ == "__main__":
    main()
