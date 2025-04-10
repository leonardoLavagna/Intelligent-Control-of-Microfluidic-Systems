import os
import sys
import pandas as pd
import json
import numpy as np
import logging


######################################################################################################
# 1. Load Files and Configurations
######################################################################################################
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from _Files.config import *


setup_logging(logs_path, "data_checker.log")


with open(files_path + '/features_bounds.json', 'r') as f:
    feature_bounds = json.load(f)


DATA_PATH = data_path
csv_files = [f for f in os.listdir(DATA_PATH) if f.endswith('.csv')]


######################################################################################################
# 1.Processing and Saving
######################################################################################################
def check_and_replace_with_nan(df, bounds):
    """
    Check DataFrame values against specified bounds and replace out-of-bound values with NaN.
    Args:
        df (pd.DataFrame): The input DataFrame to be checked and modified.
        bounds (dict): A dictionary where keys are column names and values are either:
            - A list with two numeric values [lower, upper] specifying valid bounds.
            - A list of valid categorical values.

    Returns:
        tuple: A tuple containing:
            - pd.DataFrame: The modified DataFrame with out-of-bound values replaced by NaN.
            - int: The total number of values that were replaced.
    """
    total_changes = 0  
    for column in df.columns:
        if column in bounds:
            if isinstance(bounds[column], list):
                if isinstance(bounds[column][0], (int, float)): 
                    lower, upper = bounds[column]
                    changes = df[column].apply(lambda x: x < lower or x > upper).sum()
                    total_changes += changes
                    df[column] = df[column].apply(lambda x: np.nan if x < lower or x > upper else x)
                else:  
                    valid_values = bounds[column]
                    changes = df[column].apply(lambda x: x not in valid_values).sum()
                    total_changes += changes
                    df[column] = df[column].apply(lambda x: np.nan if x not in valid_values else x)
    return df, total_changes  


logging.info("Started checking feature bounds...".upper())
for csv_file in csv_files:
    logging.info(f"Processing file: {csv_file}")
    file_path = os.path.join(DATA_PATH, csv_file)
    try:
        df = pd.read_csv(file_path)
        logging.info(f"Loaded {csv_file} successfully.")
    except Exception as e:
        logging.error(f"Error loading {csv_file}: {str(e)}")
        raise Exception(f"Error loading {csv_file}: {str(e)}")
    # Check and replace out-of-bounds values with NaN, and count changes
    df_modified, changes = check_and_replace_with_nan(df, feature_bounds)
    logging.info(f"Checked and replaced out-of-bounds values for {csv_file}.")
    logging.info(f"Number of values replaced with NaN in {csv_file}: {changes}")
    # Overwrite the existing dataframe with the modified dataframe 
    try:
        df_modified.to_csv(file_path, index=False)  
        logging.info(f"Overwritten the original file with modified data: {file_path}.")
    except Exception as e:
        logging.error(f"Error saving modified file {csv_file}: {str(e)}")
logging.info("...DONE!\n\n")