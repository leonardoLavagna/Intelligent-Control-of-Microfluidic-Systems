import os
import sys
import re
import pandas as pd
import logging


######################################################################################################
# 1. Load Files and Configurations
######################################################################################################
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from _Files.config import logs_path, data_path, setup_logging


setup_logging(logs_path, "data_merger.log")


def read_csv_file(file_path: str) -> pd.DataFrame:
    """
    Load a CSV file into a DataFrame, handling errors.

    Args:
        file_path (str): The path to the CSV file to be loaded.

    Returns:
        pd.DataFrame: A DataFrame containing the CSV file's contents, or an empty DataFrame if an error occurs.

    Raises:
        Exception: For any unexpected errors when loading the file.   
    """
    try:
        df = pd.read_csv(file_path)
        df = df.drop(columns=["CHIP"], errors='ignore')
        return df
    except Exception as e:
        logging.error(f"An unexpected error occurred: {e}")
        raise Exception(f"An unexpected error occurred: {e}")


######################################################################################################
# 2.  Processing and Saving
######################################################################################################
def get_merged_files_from_logs() -> set:
    """
    Get a set of files that have already been merged by reading the log file.

    Returns:
        set: A set of already merged CSV files' names (without extensions).
    """
    merged_files = set()
    try:
        with open(os.path.join(logs_path, "data_merger.log"), "r") as log_file:
            log_content = log_file.read()
            merged_files = {f"data_{num}.csv" for num in re.findall(r'\bdata_(\d+)\.csv\b', log_content)}
    except FileNotFoundError:
        logging.warning(f"Log file not found at {logs_path}. Assuming no files are merged yet.")
    return merged_files


def merge_csv_files(directory: str, merged_files: set) -> pd.DataFrame:
    """
    Merge all CSV files from a specified directory, skipping files with 'validation' in the name
    and files that have already been merged.

    Args:
        directory (str): The directory where the CSV files are stored.
        merged_files (set): Set of already merged file names to skip.

    Returns:
        pd.DataFrame: A DataFrame containing the merged data from all valid CSV files.

    Raises:
        ValueError: If no valid CSV files are found in the directory.
    """
    all_data_frames = []
    for file in os.listdir(directory):
        if 'validation' in file.lower() or not file.endswith('.csv') or file in merged_files:
            logging.info(f"Skipping {file} (already merged or invalid)...")
            continue
        file_path = os.path.join(directory, file)
        df = read_csv_file(file_path)
        if not df.empty:
            all_data_frames.append(df)
            logging.info(f"---> Reading {file}...")
    if all_data_frames:
        return pd.concat(all_data_frames, ignore_index=True)
    else:
        logging.warning("No valid CSV files found.")
        raise ValueError("No valid CSV files found.")


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean the DataFrame by dropping duplicates and unnecessary columns.

    Args:
        df (pd.DataFrame): The DataFrame to be cleaned.

    Returns:
        pd.DataFrame: The cleaned DataFrame with duplicates and unnecessary columns removed.
    """
    if df.empty:
        return df
    df = df.drop_duplicates()
    logging.info(f"Dropped {df.duplicated().sum()} duplicates.")
    unnecessary_columns = ['ID', 'ML', 'OUTPUT']
    df = df.drop(columns=[col for col in unnecessary_columns if col in df.columns], errors='ignore')
    for column in unnecessary_columns:
        if column in df.columns:
            logging.info(f"Deleted '{column}' feature.")
    return df


def save_combined_data(df: pd.DataFrame, count: int) -> None:
    """
    Save the merged data to a new CSV file with a dynamically generated name.

    Args:
        df (pd.DataFrame): The DataFrame containing the data to be saved.
        count (int): The current count used for naming the file.

    Returns:
        None: The function saves the data to a file but does not return anything.
    """
    if not df.empty:
        file_name = f"data_{count + 1}.csv"
        df.to_csv(os.path.join(data_path, file_name), index=False)
        logging.info(f"Successfully saved to '{file_name}'.")


def check_and_save_complete_data(df: pd.DataFrame, count: int) -> None:
    """
    Check if the merge process is complete and log the final status.

    Args:
        df (pd.DataFrame): The final DataFrame to check and save.
        count (int): The current count used to track the merging progress.

    Returns:
        None: Logs the completion status but does not return anything.
    """
    if df.empty:
        return
    with open(os.path.join(logs_path, "data_merger.log"), "r") as log_file:
        log_content = log_file.read()
        data_numbers = [int(num) for num in re.findall(r'\bdata_(\d+)\.csv\b', log_content)]
    if max(data_numbers, default=0) == count + 1:
        logging.info("... DONE!\n\n")
    else:
        logging.warning("Not all files have been merged yet.")


######################################################################################################
# 3. Pipeline
######################################################################################################
def run_merger_pipeline() -> None:
    """
    Execute the entire data merging pipeline, including loading, cleaning, and saving the data.

    Args:
        None: This function runs the pipeline to process and merge the data.

    Returns:
        None: The function executes the merging process but does not return anything.
    """
    logging.info("STARTING TO MERGE CSV FILES...")
    # Get already merged files from logs
    merged_files = get_merged_files_from_logs()
    # Merge only the files that haven't been merged yet
    complete_data = merge_csv_files(data_path, merged_files)
    complete_data = clean_data(complete_data)
    # Count how many files have been saved so far
    with open(os.path.join(logs_path, "data_merger.log"), "r") as log_file:
        log_content = log_file.read()
        count = log_content.count("\n\n")
    save_combined_data(complete_data, count)
    check_and_save_complete_data(complete_data, count)


if __name__ == "__main__":
    try:
        run_merger_pipeline()
    except Exception as e:
        logging.error(f"An unexpected error occurred: {e}")
        raise Exception(f"An unexpected error occurred: {e}")
