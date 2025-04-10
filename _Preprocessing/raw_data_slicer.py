import pandas as pd
import os
import sys
import logging
from typing import Optional


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from _Files.config import raw_data_path, data_path, logs_path, setup_logging


setup_logging(logs_path, "raw_data_slicer.log")


def process_data(file_path: str, aggregation_column: str) -> None:
    """
    Process a single CSV file by grouping data based on the given aggregation column.
    It saves the aggregated results into the specified directory (data_path or raw_data_path).

    Args:
        file_path (str): The path to the CSV file to process.
        aggregation_column (str): The column name to group by during aggregation.

    Returns:
        None.
    
    Raises:
        Exception: For any unexpected errors.   
    """
    logging.info("Starting raw data aggregation...".upper())
    try:
        logging.info(f'---> Reading {file_path}...')
        df = pd.read_csv(file_path)
        logging.info(f'Data loaded successfully with {df.shape[0]} rows and {df.shape[1]} columns.')
        logging.info('Dropping rows with missing values...')
        df = df.dropna()
        logging.info(f'Data after dropping missing values: {df.shape[0]} rows.')
        if aggregation_column not in df.columns:
            logging.error(f'Column "{aggregation_column}" not found in {file_path}. Skipping file.')
            return
        logging.info(f'Aggregating data by {aggregation_column}')
        groups = df.groupby(aggregation_column)
        base_filename = os.path.basename(file_path)
        micromixer_names = ["seed", "extension", "validation"]
        for key, group in groups:
            logging.info(f'Aggregating for {aggregation_column}: {key} - {group.shape[0]} rows.')
            base_filename = os.path.basename(file_path)
            output_filename = f'{base_filename.replace(".csv", "")}_{key}.csv'
            if key == "Micromixer":
                for name in micromixer_names:
                    if "seed" in base_filename.lower():
                        output_filename = "seed.csv"
                    elif "extension" in base_filename.lower():
                        output_filename = "extension.csv"
                    elif "validation" in base_filename.lower():
                        output_filename = "validation.csv"
                    else:
                        output_filename = f'{name}.csv' 
                    output_dir = data_path  
                    output_path = os.path.join(output_dir, output_filename)
                    group.to_csv(output_path, index=False)
            else:
                output_dir = raw_data_path
            output_path = os.path.join(output_dir, output_filename) 
            group.to_csv(output_path, index=False)
            logging.info(f'Saved aggregated data.')
    except Exception as e:
        logging.error(f"An unexpected error occurred: {e}")
        raise Exception(f"An unexpected error occurred: {e}")


def process_all_files(directory_path: str = raw_data_path, aggregation_column: str = 'CHIP') -> None:
    """
    Process all CSV files in the specified directory by aggregating them
    based on the given aggregation column.

    Args:
        directory_path (str, optional): The directory containing the CSV files to process. Defaults to raw_data_path.
        aggregation_column (str, optional): The column name to group by during aggregation. Defaults to 'CHIP'.

    Returns:
        None.
    """
    for file_name in os.listdir(directory_path):
        file_path = os.path.join(directory_path, file_name)
        if file_name.endswith('.csv'):
            logging.info(f"---> Processing file: {file_name}")
            process_data(file_path, aggregation_column)
        else:
            logging.warning(f"Skipping non-CSV file: {file_name}")


if __name__ == "__main__":
    try:
        logging.info("Starting data slicing pipeline...")
        process_all_files(raw_data_path, 'CHIP')
        logging.info('...DONE!\n\n')
    except Exception as e:
        logging.error(f"An unexpected error occurred: {e}")
        raise Exception(f"An unexpected error occurred: {e}")