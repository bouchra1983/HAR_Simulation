import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split

# Define paths
data_dir = "/home/ubuntu/har_data/UCI HAR Dataset/"
train_dir = os.path.join(data_dir, "train/")
test_dir = os.path.join(data_dir, "test/")
output_dir = "/home/ubuntu/har_data/federated_har_data/"

# Create output directory
os.makedirs(output_dir, exist_ok=True)

# --- Load Data --- 

# Load feature names
features = pd.read_csv(os.path.join(data_dir, "features.txt"), sep=" ", header=None, names=["feature_id", "feature_name"])

# Check for duplicate feature names and make them unique
feature_names = features["feature_name"].tolist()
unique_feature_names = []
for i, name in enumerate(feature_names):
    # If the name is already in the unique list, append the index to make it unique
    if name in unique_feature_names:
        unique_feature_names.append(f"{name}_{i}")
    else:
        unique_feature_names.append(name)

# Load activity labels
activity_labels = pd.read_csv(os.path.join(data_dir, "activity_labels.txt"), sep=" ", header=None, names=["activity_id", "activity_name"])

# Load training data
X_train = pd.read_csv(os.path.join(train_dir, "X_train.txt"), sep="\s+", header=None, names=unique_feature_names)
y_train = pd.read_csv(os.path.join(train_dir, "y_train.txt"), header=None, names=["activity_id"])
subject_train = pd.read_csv(os.path.join(train_dir, "subject_train.txt"), header=None, names=["subject_id"])

# Load test data
X_test = pd.read_csv(os.path.join(test_dir, "X_test.txt"), sep="\s+", header=None, names=unique_feature_names)
y_test = pd.read_csv(os.path.join(test_dir, "y_test.txt"), header=None, names=["activity_id"])
subject_test = pd.read_csv(os.path.join(test_dir, "subject_test.txt"), header=None, names=["subject_id"])

# --- Combine and Prepare Data --- 

# Combine train and test data
X_combined = pd.concat([X_train, X_test], ignore_index=True)
y_combined = pd.concat([y_train, y_test], ignore_index=True)
subject_combined = pd.concat([subject_train, subject_test], ignore_index=True)

# Adjust activity IDs to be 0-indexed (original are 1-6)
y_combined["activity_id"] = y_combined["activity_id"] - 1

# Combine all into a single DataFrame
combined_data = pd.concat([subject_combined, y_combined, X_combined], axis=1)

# --- Partition Data by Subject (Client) --- 

# Get unique subject IDs (these will be our clients)
client_ids = sorted(combined_data["subject_id"].unique())
num_clients = len(client_ids)
print(f"Total number of subjects (clients): {num_clients}")

# Create data partitions for each client
for client_id in client_ids:
    client_data = combined_data[combined_data["subject_id"] == client_id].copy()
    
    # Separate features (X) and labels (y)
    y_client = client_data["activity_id"]
    X_client = client_data.drop(columns=["subject_id", "activity_id"])
    
    # Split client data into train and test (e.g., 80/20 split within each client)
    # Using stratify ensures that the proportion of activities is maintained in train/test splits
    try:
        X_client_train, X_client_test, y_client_train, y_client_test = train_test_split(
            X_client, y_client, test_size=0.2, random_state=42, stratify=y_client
        )
    except ValueError: # Handle cases where a client might have too few samples for stratification
        print(f"Warning: Client {client_id} has too few samples for stratified split. Using simple split.")
        X_client_train, X_client_test, y_client_train, y_client_test = train_test_split(
            X_client, y_client, test_size=0.2, random_state=42
        )

    # Save data for the client
    client_train_path = os.path.join(output_dir, f"client_{client_id}_train.csv")
    client_test_path = os.path.join(output_dir, f"client_{client_id}_test.csv")
    
    train_df = pd.concat([y_client_train.reset_index(drop=True), X_client_train.reset_index(drop=True)], axis=1)
    test_df = pd.concat([y_client_test.reset_index(drop=True), X_client_test.reset_index(drop=True)], axis=1)
    
    train_df.to_csv(client_train_path, index=False)
    test_df.to_csv(client_test_path, index=False)
    
    print(f"Saved data for client {client_id}: Train ({len(X_client_train)} samples), Test ({len(X_client_test)} samples)")

# Save metadata
activity_labels.to_csv(os.path.join(output_dir, "activity_labels.csv"), index=False)

# Save the mapping between original and unique feature names
feature_mapping = pd.DataFrame({
    'original_name': feature_names,
    'unique_name': unique_feature_names
})
feature_mapping.to_csv(os.path.join(output_dir, "feature_mapping.csv"), index=False)

print(f"\nFederated HAR data prepared and saved in: {output_dir}")
print(f"Number of clients created: {num_clients}")
