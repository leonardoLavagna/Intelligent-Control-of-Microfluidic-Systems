import pandas as pd
import sys
import os
import logging
from imblearn.over_sampling import SMOTENC
from sklearn.preprocessing import LabelEncoder


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from _Files.config import data_path, logs_path, setup_logging


setup_logging(logs_path, 'data_augmenter.log')


file_names = ["data_1.csv"] 
for file_name in file_names:
    file_path = os.path.join(data_path, file_name)
    logging.info(f"Full file path: {file_path}")
    logging.info(f"Starting data augmentation using SMOTE on {file_name}...")
    logging.info(f"---> Loading dataset from {file_path}")
    if not os.path.exists(file_path):
        logging.error(f"File {file_path} not found. Skipping.")
        continue
    # Load the dataset
    df = pd.read_csv(file_path)
    df = df.sample(frac=0.25, random_state=42) 
    logging.info("Dataset loaded successfully")
    # Encode 'AQUEOUS' column
    logging.info("Encoding 'AQUEOUS' column")
    label_encoder = LabelEncoder()
    df["AQUEOUS_ENC"] = label_encoder.fit_transform(df["AQUEOUS"])
    logging.info("Encoding completed: 'MQ' -> 0, 'PBS' -> 1")
    # Define feature columns and categorical column indices for SMOTENC
    feature_cols = ["ESM", "HSPC", "CHOL", "PEG", "TFR", "FRR", "SIZE", "PDI", "AQUEOUS_ENC"]
    categorical_indices = [feature_cols.index("AQUEOUS_ENC")]  # Categorical index for SMOTE
    # Apply SMOTE (SMOTENC because AQUEOUS_ENC is categorical)
    logging.info("Applying SMOTENC to balance dataset")
    smote = SMOTENC(categorical_features=categorical_indices, random_state=42)
    X_resampled, _ = smote.fit_resample(df[feature_cols], df["AQUEOUS_ENC"])
    logging.info("SMOTENC completed successfully")
    # Convert the resampled data back to DataFrame
    df_resampled = pd.DataFrame(X_resampled, columns=feature_cols)
    logging.info("Converted resampled data to DataFrame")
    # Decode 'AQUEOUS' back to original labels
    logging.info("Decoding 'AQUEOUS' column back to original labels")
    df_resampled["AQUEOUS"] = label_encoder.inverse_transform(df_resampled["AQUEOUS_ENC"].astype(int))
    df_resampled.drop(columns=["AQUEOUS_ENC"], inplace=True)
    # Reorder columns to match the original order (after dropping 'CHIP')
    column_order = ["ESM", "HSPC", "CHOL", "PEG", "TFR", "FRR", "AQUEOUS", "SIZE", "PDI"]
    df_resampled = df_resampled[column_order]
    # Ensure "ESM" and "HSPC" remain mutually exclusive
    logging.info("Ensuring 'ESM' and 'HSPC' remain mutually exclusive")
    df_resampled.loc[df_resampled["HSPC"] > 0, "ESM"] = 0
    df_resampled.loc[df_resampled["ESM"] > 0, "HSPC"] = 0
    # Save augmented dataset
    output_file = os.path.join(data_path, f"{file_name.split('.')[0]}_augmented_SMOTE.csv")
    logging.info(f"Saving augmented dataset to {output_file}")
    df_resampled.to_csv(output_file, index=False)
    logging.info(f"Data augmentation completed successfully for {file_name}. Augmented dataset saved.")
    logging.info("...DONE!\n\n")