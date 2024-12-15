# -*- coding: utf-8 -*-

# Import necessary libraries
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# %% Set a consistent Seaborn style and color palette
sns.set_theme(style="whitegrid")  # Options: "darkgrid", "white", "ticks"
custom_palette = sns.color_palette(
    "Blues_r"
)  # Reverse Blues palette for a unified look
sns.set_palette(custom_palette)


# %% Data preprocessing function
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


# Load the dataset
file_path = "C:/Users/gianm/Documents/UNI/Data Mining/Project/sncb_data_challenge.csv"  # Adjust this path as needed
data = pd.read_csv(file_path, sep=";", index_col=0)
data = preprocess_data(data)

# %% Summary stats

list_columns = [
    "vehicles_sequence",
    "events_sequence",
    "seconds_to_incident_sequence",
    "train_kph_sequence",
    "dj_ac_state_sequence",
    "dj_dc_state_sequence",
]

# Summary statistics for list lengths
print("\nSummary Statistics for Sequence Lengths:")
sequence_length_stats = data[list_columns].applymap(len).describe()

# Print the summary stats
print(sequence_length_stats)

# Save the summary statistics to a CSV file
sequence_length_stats.to_csv("sequence_length_stats.csv", index=True)

# %% Incident type analysis
incident_counts = data["incident_type"].value_counts()

# Sort the incident counts in descending order
incident_counts = incident_counts.sort_values(ascending=True)

# Plot the sorted incident counts
plt.figure(figsize=(10, 6))
sns.barplot(
    x=incident_counts.index,
    y=incident_counts.values,
    palette=custom_palette,
    order=incident_counts.index,
)
plt.title("Frequency of Incident Type", fontsize=16)
plt.xlabel("Incident Type", fontsize=14)
plt.ylabel("Count", fontsize=14)
plt.xticks(fontsize=12)
plt.tight_layout()
plt.show()


# %% Calculate the average number of unique events per incident type
def calculate_unique_events(seq):
    # Convert the sequence of events into a set to find unique events
    return len(set(seq))  # Returns the count of unique events


# Apply this function to the 'events_sequence' column for each incident type
average_unique_events = data.groupby("incident_type")["events_sequence"].apply(
    lambda x: np.mean([calculate_unique_events(seq) for seq in x])
)

# Plot the result
plt.figure(figsize=(10, 6))
average_unique_events.sort_values().plot(kind="bar", color=custom_palette)
plt.title("Average Number of Unique Events by Incident Type", fontsize=16)
plt.xlabel("Incident Type", fontsize=14)
plt.ylabel("Average Number of Unique Events", fontsize=14)
plt.xticks(fontsize=12)
plt.tight_layout()
plt.show()


# %% Calculate average incident length based on time sequences
def calculate_incident_length(seq):
    # Calculate the incident length by subtracting min time from max time in the sequence
    return np.max(seq) - np.min(seq)


# Calculate the incident length for each incident type
incident_lengths = data.groupby("incident_type")["seconds_to_incident_sequence"].apply(
    lambda x: np.mean([calculate_incident_length(seq) for seq in x])
)

plt.figure(figsize=(10, 6))
incident_lengths.sort_values().plot(kind="bar", color=custom_palette)
plt.title("Average Time by Incident Type", fontsize=16)
plt.xlabel("Incident Type", fontsize=14)
plt.ylabel("Average Time (sec)", fontsize=14)
plt.xticks(fontsize=12)
plt.tight_layout()
plt.show()

# %% Explore relationship between average speed and incident type
average_speed_by_incident = data.groupby("incident_type")["train_kph_sequence"].apply(
    lambda x: np.mean([np.mean(seq) for seq in x])
)

plt.figure(figsize=(10, 6))
average_speed_by_incident.sort_values().plot(kind="bar", color=custom_palette)
plt.title("Average Speed by Incident Type", fontsize=16)
plt.xlabel("Incident Type", fontsize=14)
plt.ylabel("Average Speed (kph)", fontsize=14)
plt.xticks(fontsize=12)
plt.tight_layout()
plt.show()

# #%% Explore spatial distribution of incidents
# plt.figure(figsize=(12, 8))
# sns.scatterplot(x=data['approx_lon'], y=data['approx_lat'], hue=data['incident_type'], palette="Set2", s=100)
# plt.title("Spatial Distribution of Incidents", fontsize=16)
# plt.xlabel("Longitude", fontsize=14)
# plt.ylabel("Latitude", fontsize=14)
# plt.legend(title="Incident Type", fontsize=12, title_fontsize=14)
# plt.tight_layout()
# plt.show()

# %% Iterate over each unique incident type

unique_incident_types = data["incident_type"].unique()
# Loop through each incident type
for incident_type in unique_incident_types:
    plt.figure(figsize=(12, 8))

    # Filter rows for the current incident type
    subset = data[data["incident_type"] == incident_type]

    # Plot the relationship for each row
    for idx, row in subset.iterrows():
        plt.scatter(
            row["seconds_to_incident_sequence"],
            row["train_kph_sequence"],
            alpha=0.6,
            label=f"Row {idx}",
        )

    # Add plot details
    plt.title(
        f"Speed vs. Seconds to Incident for Incident Type {incident_type}", fontsize=16
    )
    plt.xlabel("Seconds to Incident", fontsize=14)
    plt.ylabel("Train Speed (kph)", fontsize=14)
    plt.tight_layout()

    # Optionally, you can show the plot
    plt.show()

    # Close the figure to avoid overlapping with the next plot
    plt.close()

# %% Choose all time steps


# Function to plot boxplots for each feature at each time step
def plot_feature_boxplots(data, feature):
    plt.figure(figsize=(14, 8))

    # Extract the values for the feature at each time step
    time_step_values = []
    for _, row in data.iterrows():
        if len(row[feature]) > 0:
            time_step_values.append(
                row[feature]
            )  # Collect all sequences for the feature
        else:
            time_step_values.append(
                [np.nan] * len(row[feature])
            )  # Handle empty sequences if necessary

    # Convert the list of sequences to a DataFrame for plotting
    time_step_values_df = pd.DataFrame(time_step_values)
    time_step_values_df["incident_type"] = data[
        "incident_type"
    ]  # Add the incident_type column

    # Plot the boxplot
    sns.boxplot(
        x="incident_type", y=time_step_values_df.columns[0], data=time_step_values_df
    )

    # Add titles and labels
    plt.title(f"Feature Values Over Time Steps for '{feature}'", fontsize=16)
    plt.xlabel("Incident Type", fontsize=14)
    plt.ylabel(f"Feature Value for '{feature}'", fontsize=14)
    plt.xticks(fontsize=12)
    plt.tight_layout()

    # Show the plot
    plt.show()


# Plot boxplots for each feature
plot_feature_boxplots(data, "train_kph_sequence")
plot_feature_boxplots(data, "dj_ac_state_sequence")
plot_feature_boxplots(data, "dj_dc_state_sequence")


# Function to count unique values for each incident type
def count_unique_values_by_incident_type(data, sequence_col, incident_col):
    # Group the data by incident type
    grouped_data = data.groupby(incident_col)

    # Dictionary to store counts for each incident type
    incident_type_counts = {}

    # Iterate through each group (incident type)
    for incident_type, group in grouped_data:
        # Flatten the sequence for each incident type
        all_values = [item for sublist in group[sequence_col] for item in sublist]

        # Count the occurrences of each unique value in the sequence
        value_counts = pd.Series(all_values).value_counts()

        # Store the value counts for each incident type
        incident_type_counts[incident_type] = value_counts

    return incident_type_counts


# Function to create a single heatmap for each sequence column
def create_single_heatmap(data, sequence_col, incident_col):
    # Count the occurrences of each unique value for each incident type
    incident_type_counts = count_unique_values_by_incident_type(
        data, sequence_col, incident_col
    )

    # Create a DataFrame for the heatmap
    # Each row will represent an incident type, and each column will represent a unique value in the sequence
    heatmap_df = pd.DataFrame(
        incident_type_counts
    ).T  # Transpose to make it suitable for a heatmap

    # Sort the DataFrame to make it easier to visualize
    heatmap_df = heatmap_df.sort_index(
        axis=1, ascending=True
    )  # Sort columns by the sequence values

    # Plot the heatmap
    plt.figure(figsize=(12, 8))
    sns.heatmap(heatmap_df, fmt="d", cmap="YlGnBu", cbar=True)
    plt.title(f"Heatmap of {sequence_col} Counts by Incident Type")
    plt.xlabel("Unique Sequence Values")
    plt.ylabel("Incident Type")

    # Save the heatmap as an image
    plt.show()
    plt.close()  # Close the plot to avoid displaying it after saving


# Create heatmap for vehicles_sequence
create_single_heatmap(data, "vehicles_sequence", "incident_type")

# Create heatmap for events_sequence
create_single_heatmap(data, "events_sequence", "incident_type")


# Function to split sequences based on time periods
def split_by_time_period(data, sequence_col, time_col):
    # Define time periods based on seconds_to_incident_sequence
    before, during, after = [], [], []

    for _, row in data.iterrows():
        sequence = row[sequence_col]
        time_sequence = row[time_col]

        for seq, time in zip(sequence, time_sequence):
            if time < -1000:  # Before the incident
                before.append(seq)
            elif -1000 <= time <= 1000:  # During the incident
                during.append(seq)
            else:  # After the incident
                after.append(seq)

    return before, during, after


# Function to create a heatmap from the sequence data with normalization and logarithmic scale
def create_heatmap(
    data, sequence_col, time_col, normalize=False, log_scale=False, min_threshold=1
):
    # Split the sequence into before, during, and after
    before, during, after = split_by_time_period(data, sequence_col, time_col)

    # Count the occurrences of each unique value in the sequences
    before_counts = pd.Series(before).value_counts()
    during_counts = pd.Series(during).value_counts()
    after_counts = pd.Series(after).value_counts()

    # Create a DataFrame for the heatmap
    heatmap_df = pd.DataFrame(
        {"Before": before_counts, "During": during_counts, "After": after_counts}
    ).fillna(0)  # Fill NaN values with 0

    # Apply minimum threshold (optional)
    heatmap_df = heatmap_df[heatmap_df.sum(axis=1) >= min_threshold]

    # Normalize the counts (optional)
    if normalize:
        # Convert the sum to a 2D numpy array (we use np.newaxis)
        row_sums = heatmap_df.sum(axis=1).values[
            :, np.newaxis
        ]  # Convert to a 2D numpy array
        heatmap_df = heatmap_df / row_sums  # Broadcast division by row sums

    # Apply logarithmic scale (optional)
    if log_scale:
        heatmap_df = np.log1p(
            heatmap_df
        )  # Apply log scale (log(1 + x) to avoid log(0))

    # Plot heatmap
    plt.figure(figsize=(10, 6))
    sns.heatmap(heatmap_df.T, fmt=".2f", cmap="YlGnBu", cbar=True)
    plt.title(f"Heatmap of {sequence_col} Counts by Time Period")
    plt.xlabel("Unique Values")
    plt.ylabel("Time Period")
    plt.show()


# Create heatmap for vehicles_sequence with normalization and log scale
create_heatmap(
    data,
    "vehicles_sequence",
    "seconds_to_incident_sequence",
    normalize=True,
    log_scale=True,
)

# Create heatmap for events_sequence with normalization and log scale
create_heatmap(
    data,
    "events_sequence",
    "seconds_to_incident_sequence",
    normalize=True,
    log_scale=True,
)

# %% Choose a specific time step (e.g., the first or last step)


# # Function to plot boxplot at time step 0
# def plot_feature_boxplots_at_time_0(data, feature, time_step=0):
#     plt.figure(figsize=(14, 8))

#     # Extract feature values at the given time step (time_step=0 here)
#     time_step_values = []
#     for _, row in data.iterrows():
#         # Find the index of the time_step (0 in this case)
#         if time_step in row["seconds_to_incident_sequence"]:
#             idx = row["seconds_to_incident_sequence"].index(time_step)
#             time_step_values.append(
#                 row[feature][idx]
#             )  # Extract the value at the time step

#     # Add incident type to the data for plotting
#     time_step_values_df = pd.DataFrame(time_step_values, columns=[feature])
#     time_step_values_df["incident_type"] = data[
#         "incident_type"
#     ]  # Include incident type

#     # Plot the boxplot
#     sns.boxplot(x="incident_type", y=feature, data=time_step_values_df, palette="Blues")

#     # Add plot details
#     plt.title(f"Feature Values at Time Step {time_step} by Incident Type", fontsize=16)
#     plt.xlabel("Incident Type", fontsize=14)
#     plt.ylabel(f"Feature Value for {feature}", fontsize=14)
#     plt.tight_layout()

#     # Show the plot
#     plt.show()


# # Plot the boxplots for 'vehicles_sequence', 'events_sequence', and 'train_kph_sequence' at time 0
# plot_feature_boxplots_at_time_0(data, "vehicles_sequence", time_step=0)
# plot_feature_boxplots_at_time_0(data, "events_sequence", time_step=0)
# plot_feature_boxplots_at_time_0(data, "train_kph_sequence", time_step=0)


# Function to plot boxplots for 'before', 'during', and 'after' time periods
def plot_feature_boxplots_by_time_period(data, feature):
    # Initialize the plot
    plt.figure(figsize=(14, 8))

    # Initialize lists to store the feature values and their corresponding time period
    feature_values = []
    time_periods = []
    incident_types = []

    # Loop through the data and categorize based on time
    for _, row in data.iterrows():
        if feature == "train_kph_sequence":
            feature_sequence = row["train_kph_sequence"]
        elif feature == "dj_ac_state_sequence":
            feature_sequence = row["dj_ac_state_sequence"]
        elif feature == "dj_dc_state_sequence":
            feature_sequence = row["dj_dc_state_sequence"]
        else:
            continue  # Skip if the feature is not one of these

        # Categorize data into 'before', 'during', and 'after'
        for time, feature_value in zip(
            row["seconds_to_incident_sequence"], feature_sequence
        ):
            if time < -1000:  # Before the incident
                feature_values.append(feature_value)
                time_periods.append("Before")
                incident_types.append(row["incident_type"])
            elif -1000 <= time <= 1000:  # During the incident
                feature_values.append(feature_value)
                time_periods.append("During")
                incident_types.append(row["incident_type"])
            elif time > 1000:  # After the incident
                feature_values.append(feature_value)
                time_periods.append("After")
                incident_types.append(row["incident_type"])

    # Create a DataFrame for plotting
    feature_values_df = pd.DataFrame(
        {
            feature: feature_values,
            "Time Period": time_periods,
            "incident_type": incident_types,
        }
    )

    # Plot the boxplot
    sns.boxplot(
        x="incident_type",
        y=feature,
        hue="Time Period",
        data=feature_values_df,
        palette="Blues",
    )

    # Add plot details
    plt.title(
        "Feature Values by Time Period (Before, During, After) and Incident Type",
        fontsize=16,
    )
    plt.xlabel("Incident Type", fontsize=14)
    plt.ylabel(f"Feature Value for {feature}", fontsize=14)
    plt.tight_layout()

    # Show the plot
    plt.show()


# Plot the boxplots for 'vehicles_sequence', 'events_sequence', and 'train_kph_sequence' based on time periods
plot_feature_boxplots_by_time_period(data, "train_kph_sequence")
