# -*- coding: utf-8 -*-

# Import necessary libraries
from collections import Counter, defaultdict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from mlxtend.frequent_patterns import fpgrowth
from mlxtend.preprocessing import TransactionEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier

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


# Function to train and evaluate a classification model
def train_and_evaluate_model(df, min_class_size=2):
    """
    Trains and evaluates an XGBoost classification model without SMOTE.
    It performs encoding, model training, and evaluation, including feature importance
    visualization and a confusion matrix.

    Parameters:
    df (pd.DataFrame): The input DataFrame with features and target variable.
    min_class_size (int, optional): Minimum number of samples required for a class to be included in training.
                                    Default is 2.

    Returns:
    tuple: A tuple containing:
        - encoding_map (dict): Mapping of original class labels to encoded integer labels.
        - xgb_clf (XGBClassifier): The trained XGBoost classifier model.
        - classification_report (dict): A dictionary representation of the classification report.
    """

    # Use 'incident_type' as the target variable
    y = df["incident_type"]
    X = df.loc[:, "vehicle_unique_perc":]

    # Encode target labels into integers
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)

    # Display the mapping of original class labels to their encoded values
    encoding_map = {
        class_label: index for index, class_label in enumerate(label_encoder.classes_)
    }
    print("Encoding Map:", encoding_map)

    # Filter out classes with insufficient samples
    class_counts = pd.Series(y_encoded).value_counts()
    valid_classes = class_counts[class_counts >= min_class_size].index

    mask = np.isin(y_encoded, valid_classes)
    X, y_encoded = X[mask], y_encoded[mask]

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.3, random_state=42
    )

    # Calculate class weights to address imbalance
    class_counts = Counter(y_train)
    total_samples = len(y_train)
    class_weights = {
        label: total_samples / count for label, count in class_counts.items()
    }
    sample_weights = np.array([class_weights[label] for label in y_train])

    # Predefined hyperparameters for the XGBoost model
    best_params = {
        "n_estimators": 660,
        "max_depth": 15,
        "learning_rate": 0.04698734918417903,
        "subsample": 0.8502600673011051,
        "colsample_bytree": 0.6898905947681309,
        "gamma": 0.1,
        "reg_alpha": 1.3390663337605688,
        "reg_lambda": 1.3121669381733938,
        "min_child_weight": 2,
        "eval_metric": "logloss",
        "random_state": 42,
        "use_label_encoder": False,
    }

    # Train an XGBoost classifier using early stopping
    xgb_clf = XGBClassifier(**best_params)

    print("Training the XGBoost model with early stopping...")

    # Tracking metrics for plotting
    eval_set = [(X_train, y_train), (X_test, y_test)]
    xgb_clf.set_params(eval_metric="mlogloss")  # Pass eval_metric here

    xgb_clf.fit(
        X_train,
        y_train,
        sample_weight=sample_weights,
        eval_set=eval_set,
        verbose=True,
    )

    # Evaluate the model on the test set
    y_pred_encoded = xgb_clf.predict(X_test)

    # Convert encoded labels back to original labels
    y_test_original = label_encoder.inverse_transform(y_test)
    y_pred_original = label_encoder.inverse_transform(y_pred_encoded)

    print("\nClassification Report without SMOTE:")
    print(
        classification_report(
            y_test_original,
            y_pred_original,
            labels=label_encoder.classes_,
            zero_division=1,
        )
    )

    # # Save the classification report as a DataFrame
    # report_df = classification_report_to_dataframe(
    #     y_test,
    #     y_pred_encoded,
    #     labels=unique_labels,  # Pass unique labels here to avoid mismatch
    # )
    # report_df.to_csv("classification_report.csv", index=True)

    # Plot the confusion matrix
    cm = confusion_matrix(
        y_test_original, y_pred_original, labels=label_encoder.classes_
    )
    plt.figure(figsize=(12, 9))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=label_encoder.classes_,
        yticklabels=label_encoder.classes_,
    )
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.show()

    # Plot learning curves
    results = xgb_clf.evals_result()
    epochs = len(results["validation_0"]["mlogloss"])
    x_axis = range(0, epochs)
    plt.figure(figsize=(12, 6))
    plt.plot(x_axis, results["validation_0"]["mlogloss"], label="Train")
    plt.plot(x_axis, results["validation_1"]["mlogloss"], label="Validation")
    plt.legend()
    plt.ylabel("M Log Loss")
    plt.title("XGBoost M Log Loss")
    plt.show()

    return (
        encoding_map,
        xgb_clf,
        classification_report(
            y_test, y_pred_encoded, zero_division=1, output_dict=True
        ),
    )


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
    encoding_map, model, report = train_and_evaluate_model(df)

    return df, encoding_map, model, report


if __name__ == "__main__":
    main()
