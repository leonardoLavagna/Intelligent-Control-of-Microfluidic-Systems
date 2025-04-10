import os
import sys
import logging
import json
import pandas as pd
from pathlib import Path
from typing import Dict, Set, Any
from rapidfuzz import process, fuzz
import re
import numpy as np


######################################################################################################
# 1. Load Files and Configurations
######################################################################################################
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from _Files.config import logs_path, files_path, raw_data_path, setup_logging 
from _Files.config import column_names as REQUIRED_COLUMNS


setup_logging(logs_path, "raw_data_checker.log")
change_counter = 0

def load_file(file_path: str) -> pd.DataFrame:
    """
    Load an Excel or CSV file into a DataFrame, handling errors.

    Args:
        file_path (str): The path to the file to be loaded.

    Returns:
        pd.DataFrame: A DataFrame containing the file's contents, or an empty DataFrame if an error occurs.

    Raises:
        Exception: For any unexpected errors.   
    """
    try:
        if file_path.endswith(".xlsx"):
            return pd.read_excel(file_path)
        elif file_path.endswith(".csv"):
            return pd.read_csv(file_path)
        else:
            raise ValueError(f"Unsupported file format: {file_path}")
    except Exception as e:
        logging.error(f"An unexpected error occurred: {e}")
        raise Exception(f"An unexpected error occurred: {e}")


def load_reference_data(file_path: str) -> Set[str]:
    """
    Load reference file and return a set of column names.

    Args:
        file_path (str): The path to the reference file to be loaded.

    Returns:
        Set[str]: A set containing the column names from the loaded reference file.
    
    Raises:
        Exception: For any unexpected errors.
    """
    try:
        df_ref = load_file(file_path)  
        if df_ref is None or not hasattr(df_ref, 'columns'):
            logging.error("Loaded data is not a valid DataFrame or contains no columns.")
            raise ValueError("Loaded data is not a valid DataFrame or contains no columns.")
        columns = set(df_ref.columns)
        return columns
    except FileNotFoundError:
        logging.error(f"An unexpected error occurred: {e}")
        raise Exception(f"An unexpected error occurred: {e}")
    except ValueError as e:
        logging.error(f"An unexpected error occurred: {e}")
        raise Exception(f"An unexpected error occurred: {e}")
    except Exception as e:
        logging.error(f"An unexpected error occurred: {e}")
        raise Exception(f"An unexpected error occurred: {e}")


def load_feature_bounds(file_path: str) -> Dict[str, Any]:
    """
    Load feature bounds from a JSON file.

    Args:
        file_path (str): The path to the feature bounds JSON file.

    Returns:
        Dict[str, Any]: A dictionary containing feature bounds information.
    
    Raises:
        Exception: If there is an error loading the JSON file.
    """
    try:
        with open(file_path, 'r') as file:
            return json.load(file)
    except Exception as e:
        logging.error(f"An unexpected error occurred: {e}")
        raise Exception(f"An unexpected error occurred: {e}")


with open(files_path + '/features_names_mappings.json', 'r') as file:
    CUSTOM_RENAMES = json.load(file)


FEATURE_BOUNDS = load_feature_bounds(files_path + '/features_bounds.json')


######################################################################################################
# 2. Column Processing Functions
######################################################################################################
def find_fuzzy_column_matches(df_new: pd.DataFrame) -> Dict[str, str]:
    """
    Find fuzzy-matched columns that need renaming.

    Args:
        df_new (pd.DataFrame): The DataFrame to process for fuzzy column matches.

    Returns:
        Dict[str, str]: A dictionary with the columns to rename as keys and their best fuzzy matches as values.
    """
    return {
        col: best_match[0]
        for col in df_new.columns
        if (best_match := process.extractOne(col, REQUIRED_COLUMNS, scorer=fuzz.partial_ratio)) and best_match[1] >= 85
    }


def rename_columns(df_new: pd.DataFrame) -> None:
    global change_counter 
    # Custom renames
    for col, new_name in CUSTOM_RENAMES.items():
        if col in df_new.columns and new_name != col:
            df_new.rename(columns={col: new_name}, inplace=True)
            logging.warning(f"Renamed '{col}' to '{new_name}' based on custom renaming rules.")
            change_counter += 1  
    # Fuzzy matches
    fuzzy_matches = find_fuzzy_column_matches(df_new)
    for old_col, new_col in fuzzy_matches.items():
        if old_col != new_col:
            df_new.rename(columns={old_col: new_col}, inplace=True)
            logging.warning(f"Renaming '{old_col}' to '{new_col}' based on fuzzy matching.")
            change_counter += 1  
    # Case-insensitive matches (avoid redundant logs)
    for col in df_new.columns:
        for ref_col in REQUIRED_COLUMNS:
            if col.lower() == ref_col.lower() and col != ref_col:
                df_new.rename(columns={col: ref_col}, inplace=True)
                change_counter += 1  


######################################################################################################
# 3. Data Validation Functions
######################################################################################################
def validate_feature_values(df_new: pd.DataFrame) -> None:
    global change_counter 
    logged_warnings = set()
    for feature, bounds in FEATURE_BOUNDS.items():
        if feature in df_new.columns:
            if isinstance(bounds, list) and len(bounds) == 2 and all(isinstance(x, (int, float)) for x in bounds):
                # Numerical Feature Bounds Check
                min_val, max_val = bounds
                out_of_bounds = (df_new[feature] < min_val) | (df_new[feature] > max_val)
                if out_of_bounds.any():
                    warning_message = f"{feature}: Replacing out-of-bounds values with NaN."
                    if warning_message not in logged_warnings:
                        logging.warning(warning_message)
                        logged_warnings.add(warning_message)
                    df_new.loc[out_of_bounds, feature] = np.nan
                    change_counter += out_of_bounds.sum() 
            elif isinstance(bounds, list) and all(isinstance(x, str) for x in bounds):
                # Categorical Feature Check
                for i, value in df_new[feature].items():
                    if pd.notna(value) and value not in bounds:
                        match = process.extractOne(value, bounds, scorer=fuzz.partial_ratio)
                        if match and match[1] >= 85:
                            corrected_value = match[0]
                            warning_message = f"{feature}: Replacing invalid '{value}' with '{corrected_value}' (fuzzy match)."
                            if warning_message not in logged_warnings:
                                logging.warning(warning_message)
                                logged_warnings.add(warning_message)
                            df_new.at[i, feature] = corrected_value
                            change_counter += 1  
                        else:
                            warning_message = f"{feature}: '{value}' is invalid, replacing with NaN."
                            if warning_message not in logged_warnings:
                                logging.warning(warning_message)
                                logged_warnings.add(warning_message)
                            df_new.at[i, feature] = np.nan
                            change_counter += 1 


######################################################################################################
# 4. File Processing Functions
######################################################################################################
def read_log_file(file_path: str) -> str:
    """
    Read the content of the log file.

    Args:
        file_path (str): The path to the log file.

    Returns:
        str: The content of the log file.
    """
    with open(file_path, 'r') as file:
        return file.read()


def get_processed_files(file_path: str) -> Set[str]:
    """
    Extract filenames that have already been processed from the log file.

    Args:
        file_path (str): The path to the log file.

    Returns:
        Set[str]: A set of filenames that have already been processed.
    """
    log_content = read_log_file(file_path)
    file_names = re.findall(r'---> Processing _Raw_Data/(.*?\.xlsx)\.\.\.', log_content)
    return set(file_names)


def process_file(file_path: str) -> None:
    """
    Process a single file and apply changes as needed.

    Args:
        file_path (str): The path to the file to be processed.

    Returns:
        None.
    """
    df_new = load_file(file_path)
    if df_new.empty:
        logging.error(f"Skipping {file_path} due to loading failure.")
        return
    logging.info(f"---> Processing {file_path}...")
    rename_columns(df_new)
    validate_feature_values(df_new)
    output_file_path = Path(raw_data_path) / f"{Path(file_path).stem}.csv"
    df_new.to_csv(output_file_path, index=False)
    logging.info(f"Processed data saved to {output_file_path}")


def process_all_files(folder_path: str, log_file_path: str) -> None:
    """
    Process all files in the folder, avoiding reprocessing files already processed.

    Args:
        folder_path (str): The folder path where the files are located.
        log_file_path (str): The path to the log file to check already processed files.

    Returns:
        None.
    """
    global change_counter  
    processed_files = get_processed_files(log_file_path)
    files = [f for f in os.listdir(folder_path) if f.endswith('.xlsx')]
    files_to_process = [f for f in files if f not in processed_files]
    if not files_to_process:
        logging.info("No new files to process.")
        return
    for file in files_to_process:
        process_file(os.path.join(folder_path, file))


######################################################################################################
# 5. Pipeline Execution
######################################################################################################
def run_pipeline() -> None:
    """
    Main function to execute the data-checking pipeline.

    Args:
        None.

    Returns:
        None.
    """
    global change_counter  
    log_file_path = logs_path + "/raw_data_checker.log"
    logging.info("CHECKING RAW DATA...")
    process_all_files(raw_data_path, log_file_path)
    logging.info(f"Total changes made: {change_counter}")
    logging.info("...DONE!\n\n")


if __name__ == "__main__":
    try:
        run_pipeline()
    except Exception as e:
        logging.error(f"An unexpected error occurred: {e}")
        raise Exception(f"An unexpected error occurred: {e}")
