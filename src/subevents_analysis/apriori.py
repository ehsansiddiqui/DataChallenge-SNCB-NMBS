import pandas as pd
from ast import literal_eval
from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.preprocessing import TransactionEncoder
from sklearn.model_selection import KFold
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import numpy as np

# Global Parameters for Consistency
MIN_SUPPORT = 0.3  # Minimum support threshold
MIN_CONFIDENCE = 0.5  # Minimum confidence threshold
K_FOLDS = 5  # Number of folds for cross-validation

# Load and preprocess the data
data = pd.read_csv('sncb_data_challenge.csv', delimiter=';')
data = data[['events_sequence', 'incident_type', 'seconds_to_incident_sequence']]  # Include time column

# Convert 'events_sequence' and 'seconds_to_incident_sequence' to lists safely
data['events_sequence'] = data['events_sequence'].apply(
    lambda x: literal_eval(x) if isinstance(x, str) else x)
data['seconds_to_incident_sequence'] = data['seconds_to_incident_sequence'].apply(
    lambda x: literal_eval(x) if isinstance(x, str) else x
)

# Updated function to filter based on observation-specific time windows
def filter_by_time_window_per_observation(data):
    """
    Filter sequences using a fixed time window [0, 2400] seconds per observation
    based on incident type.

    Parameters:
        data (pd.DataFrame): Input DataFrame with 'events_sequence',
                             'seconds_to_incident_sequence', and 'incident_type'.

    Returns:
        pd.DataFrame: Filtered DataFrame.
    """
    def filter_sequence(row):
        time_sequence = row['seconds_to_incident_sequence']
        event_sequence = row['events_sequence']

        if not isinstance(time_sequence, list) or not isinstance(event_sequence, list):
            return None, None

        # Apply fixed time window: start at 0 seconds, end at 2400 seconds
        window_start, window_end = 0, 2400
        filtered_events = [event for time, event in zip(time_sequence, event_sequence)
                           if window_start <= time <= window_end]
        filtered_times = [time for time in time_sequence if window_start <= time <= window_end]

        return filtered_events, filtered_times

    # Apply filtering function to each row
    data[['events_sequence', 'seconds_to_incident_sequence']] = data.apply(
        lambda row: pd.Series(filter_sequence(row)), axis=1)

    # Drop rows with no valid events after filtering
    data = data.dropna(subset=['events_sequence', 'seconds_to_incident_sequence'])
    data = data[data['events_sequence'].apply(lambda x: len(x) > 0 if isinstance(x, list) else False)]

    return data

# Apply the modified function
data = filter_by_time_window_per_observation(data)

# Feature Engineering Functions
def binary_one_hot_encoding(data, min_support):
    te = TransactionEncoder()
    te_ary = te.fit(data['events_sequence']).transform(data['events_sequence'])
    df = pd.DataFrame(te_ary, columns=te.columns_)

    frequent_itemsets = apriori(df, min_support=min_support, use_colnames=True, max_len=2)

    for itemset in frequent_itemsets['itemsets']:
        itemset_str = '_'.join(sorted(map(str, itemset)))
        df[itemset_str] = df[list(itemset)].all(axis=1).astype(int)

    return pd.concat(
        [data[['incident_type']], df[frequent_itemsets['itemsets'].apply(lambda x: '_'.join(sorted(map(str, x))))]],
        axis=1
    )

def support_confidence_encoding(data, min_support=0.1, min_confidence=0.5):
    """
    Perform support and confidence encoding for each incident type using Apriori and association rules.

    Parameters:
        data (pd.DataFrame): The input DataFrame containing 'events_sequence' and 'incident_type'.
        min_support (float): The minimum support threshold for the Apriori algorithm.
        min_confidence (float): The minimum confidence threshold for association rules.

    Returns:
        pd.DataFrame: A DataFrame with encoded support and confidence values for itemsets and rules.
    """
    # Prepare a final DataFrame to collect results
    all_encoded_data = []

    # Group data by incident type
    grouped_data = data.groupby('incident_type')

    for incident_type, group in grouped_data:
        print(f"Processing incident type: {incident_type}")

        # Use TransactionEncoder to transform sequences
        te = TransactionEncoder()
        te_ary = te.fit(group['events_sequence']).transform(group['events_sequence'])
        df = pd.DataFrame(te_ary, columns=te.columns_)

        # Apply Apriori to find frequent itemsets
        frequent_itemsets = apriori(df, min_support=min_support, use_colnames=True, max_len=2)

        # Skip if no itemsets are found
        if frequent_itemsets.empty:
            print(f"No frequent itemsets found for incident type {incident_type}.")
            continue

        # Generate association rules
        # Pass frequent_itemsets to num_itemsets argument
        rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=min_confidence, num_itemsets=frequent_itemsets)

        # Add support columns for each frequent itemset
        for itemset, support in zip(frequent_itemsets['itemsets'], frequent_itemsets['support']):
            itemset_str = '_'.join(sorted(map(str, itemset))) + '_support'
            df[itemset_str] = support

        # Add confidence columns for each rule
        for _, rule in rules.iterrows():
            antecedent_str = '_'.join(sorted(map(str, rule['antecedents'])))
            consequent_str = '_'.join(sorted(map(str, rule['consequents'])))
            rule_str = f"{antecedent_str}_to_{consequent_str}_confidence"
            df[rule_str] = rule['confidence']

        # Concatenate the support and confidence columns
        support_cols = frequent_itemsets['itemsets'].apply(lambda x: '_'.join(sorted(map(str, x))) + '_support')
        confidence_cols = [
            f"{'_'.join(sorted(map(str, rule['antecedents'])))}_to_{'_'.join(sorted(map(str, rule['consequents'])))}_confidence"
            for _, rule in rules.iterrows()
        ]

        encoded_data = pd.concat([group[['incident_type']].reset_index(drop=True), df[support_cols.tolist() + confidence_cols]], axis=1)
        all_encoded_data.append(encoded_data)

    # Combine all results into a single DataFrame
    final_encoded_data = pd.concat(all_encoded_data, axis=0).reset_index(drop=True)
    return final_encoded_data

# Cross-Validation Implementation
def evaluate_fold(train_data, test_data, min_support, min_confidence):
    te = TransactionEncoder()
    te_ary = te.fit(train_data['events_sequence']).transform(train_data['events_sequence'])
    df_train = pd.DataFrame(te_ary, columns=te.columns_)

    frequent_itemsets = apriori(df_train, min_support=min_support, use_colnames=True, max_len=2)
    if frequent_itemsets.empty:
        return 0

    # Pass the 'frequent_itemsets' DataFrame to 'num_itemsets'
    rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=min_confidence, num_itemsets=frequent_itemsets)
    if rules.empty:
        return 0

    covered_transactions = sum(
        any(rule['antecedents'].issubset(set(t)) for _, rule in rules.iterrows()) for t in test_data['events_sequence'])
    coverage = covered_transactions / len(test_data)
    return coverage


def cross_validate(data, min_support, min_confidence, k_folds):
    kf = KFold(n_splits=k_folds, shuffle=True, random_state=42)
    coverage_scores = []

    for train_idx, test_idx in kf.split(data):
        train_data, test_data = data.iloc[train_idx], data.iloc[test_idx]
        coverage = evaluate_fold(train_data, test_data, min_support, min_confidence)
        coverage_scores.append(coverage)
        print(f"Fold Coverage: {coverage:.2f}")

    average_coverage = sum(coverage_scores) / len(coverage_scores)
    print(f"Average Coverage: {average_coverage:.2f}")
    return average_coverage


# Grid Search for Parameter Tuning
def grid_search(data, param_grid):
    best_params = None
    best_score = -1

    for params in param_grid:
        print(f"Testing parameters: {params}")
        score = cross_validate(data, params['min_support'], params['min_confidence'], K_FOLDS)
        if score > best_score:
            best_score = score
            best_params = params

    print(f"Best Parameters: {best_params}, Best Score: {best_score:.2f}")
    return best_params, best_score


# Define Parameter Grid
param_grid = [
    {'min_support': 0.2, 'min_confidence': 0.5},
    {'min_support': 0.3, 'min_confidence': 0.6},
    {'min_support': 0.4, 'min_confidence': 0.7},
    {'min_support': 0.4, 'min_confidence': 0.8},
    {'min_support': 0.5, 'min_confidence': 0.5},
    {'min_support': 0.5, 'min_confidence': 0.6},
    {'min_support': 0.5, 'min_confidence': 0.7},
    {'min_support': 0.5, 'min_confidence': 0.8},
    {'min_support': 0.5, 'min_confidence': 0.9},
    {'min_support': 0.6, 'min_confidence': 0.7},
    {'min_support': 0.6, 'min_confidence': 0.8},
    {'min_support': 0.7, 'min_confidence': 0.8},
    {'min_support': 0.7, 'min_confidence': 0.9},
    {'min_support': 0.8, 'min_confidence': 0.9}
]

# Perform Grid Search
best_params, best_score = grid_search(data, param_grid)

# Apply Best Parameters to Feature Engineering
# encoded_data_binary = binary_one_hot_encoding(data, min_support=best_params['min_support'])
# encoded_data_support_confidence = support_confidence_encoding(data, min_support=best_params['min_support'], min_confidence=best_params['min_confidence'])

# Apply hard coded parameters
encoded_data_binary = binary_one_hot_encoding(data, min_support=0.5)
encoded_data_support_confidence = support_confidence_encoding(data, min_support=0.5, min_confidence=0.9)

# Save Results to CSV Files
encoded_data_binary.to_csv('encoded_data_binary.csv', index=False)
encoded_data_support_confidence.to_csv('encoded_data_support_confidence.csv', index=False)

print("Encoded data saved to 'encoded_data_binary.csv' and 'encoded_data_support_confidence.csv'.")