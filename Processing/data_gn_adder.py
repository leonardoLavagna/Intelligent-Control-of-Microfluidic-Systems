import pandas as pd
import numpy as np
import sys
import os
import logging


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from _Files.config import data_path, logs_path, setup_logging


setup_logging(logs_path, 'data_augmenter.log')


file_names = ["/data_1"] 
numerical_cols = ["ESM", "HSPC", "CHOL", "PEG", "TFR", "FRR", "SIZE", "PDI"]


def add_gaussian_noise(df, numerical_cols, noise_level=0.01):
    """
    Add Gaussian noise to specified numerical columns in a DataFrame.

    Args:
        df (pd.DataFrame): The input DataFrame containing the numerical columns.
        numerical_cols (list): A list of column names (str) to which noise will be added.
        noise_level (float, optional): The proportion of the standard deviation to use as the noise level.
            Defaults to 0.01.

    Returns:
        pd.DataFrame: The modified DataFrame with noise added to the specified columns.
    """
    logging.info(f"Adding Gaussian noise with noise level: {noise_level} to numerical columns.")
    for col in numerical_cols:
        if col == "ESM":
            # Add noise only where HSPC is zero
            noise = np.random.normal(loc=0, scale=df["ESM"].std() * noise_level, size=df["ESM"].shape)
            df.loc[df["HSPC"] == 0, "ESM"] += noise[df["HSPC"] == 0]
        elif col == "HSPC":
            # Add noise only where ESM is zero
            noise = np.random.normal(loc=0, scale=df["HSPC"].std() * noise_level, size=df["HSPC"].shape)
            df.loc[df["ESM"] == 0, "HSPC"] += noise[df["ESM"] == 0]
        else:
            # Add noise for all other columns without restriction
            noise = np.random.normal(loc=0, scale=df[col].std() * noise_level, size=df[col].shape)
            df[col] += noise
    return df


for file_name in file_names:
    logging.info(f"Starting data augmentation using Gaussian noise...".upper())
    file_path = data_path + file_name + ".csv"
    logging.info(f"---> Loading dataset from {file_path}")
    try:
        df = pd.read_csv(file_path)
        df = df.sample(frac=0.25, random_state=42)
        logging.info("Dataset loaded successfully")
        # Add Gaussian noise to the numerical columns
        df = add_gaussian_noise(df, numerical_cols, noise_level=0.01)
        # Ensure "ESM" and "HSPC" remain mutually exclusive after adding noise
        df.loc[df["HSPC"] > 0, "ESM"] = 0
        df.loc[df["ESM"] > 0, "HSPC"] = 0
        logging.info("Ensured mutual exclusivity of 'ESM' and 'HSPC' after noise addition")
        # Reorder columns
        column_order = ["ESM", "HSPC", "CHOL", "PEG", "TFR", "FRR", "AQUEOUS", "SIZE", "PDI"]
        df = df[column_order]
        # Save the augmented dataset
        output_file = data_path + file_name + "_augmented_gn.csv"
        df.to_csv(output_file, index=False)
        logging.info(f"Saved augmented data to {output_file}")
    except Exception as e:
        logging.error(f"Error processing file {file_name}: {str(e)}")
        raise Exception(f"Error processing file {file_name}: {str(e)}")
    logging.info("...DONE!\n\n")