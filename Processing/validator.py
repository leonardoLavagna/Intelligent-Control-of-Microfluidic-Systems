import pandas as pd
import pickle
import torch
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
import logging
import os
import sys


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from _Files.config import models_path, data_path, plots_path, logs_path, setup_logging


def determine_target_from_filename(filename):
    """
    Determines the target variable(s) from the model filename.
    """
    name = filename.lower()
    if 'vae_encoder' in name or 'best_latent_regressor' in name:
        return None  
    has_size = 'size' in name
    has_pdi = 'pdi' in name
    if has_size and has_pdi:
        return ['SIZE', 'PDI']
    elif has_size:
        return ['SIZE']
    elif has_pdi:
        return ['PDI']
    return None


def validate_model(model_file, target_variables):
    """
    Validates a single model file using the appropriate target variable(s).
    """
    model_path = os.path.join(models_path, model_file)
    file_path = os.path.join(data_path, 'validation.csv')

    try:
        logging.info(f"VALIDATING {model_file}...")
        # Load validation data
        df = pd.read_csv(file_path)
        # Handle case with two target variables
        if len(target_variables) == 2:
            X_val = df.drop(columns=target_variables)
            y_true = df[target_variables]
        else:
            target_variable = target_variables[0]
            X_val = df.drop(columns=[target_variable])
            y_true = df[target_variable]
        # Load the model
        if model_file.endswith('.pkl'):
            with open(model_path, 'rb') as f:
                model = pickle.load(f)
        elif model_file.endswith('.pth'):
            logging.warning(f"Skipping PyTorch model {model_file}.")
            return
        else:
            logging.warning(f"Unknown model type for file {model_file}, skipping.")
            return
        # Predict
        y_pred = model.predict(X_val)
        # Plot
        plt.figure(figsize=(8, 6))
        if isinstance(y_true, pd.DataFrame):
            for i, col in enumerate(y_true.columns):
                plt.subplot(1, 2, i+1)
                plt.scatter(y_true[col], y_pred[:, i], alpha=0.6, edgecolors='k')
                plt.plot([y_true[col].min(), y_true[col].max()],
                         [y_true[col].min(), y_true[col].max()], 'r--', lw=2)
                plt.xlabel(f'True {col}')
                plt.ylabel(f'Predicted {col}')
                plt.title(f'{col} (R² = {r2_score(y_true[col], y_pred[:, i]):.2f})')
                plt.grid(True)
            plt.tight_layout()
        else:
            plt.scatter(y_true, y_pred, alpha=0.6, edgecolors='k')
            plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--', lw=2)
            plt.xlabel(f'True {target_variables[0]}')
            plt.ylabel(f'Predicted {target_variables[0]}')
            plt.title(f'{model_file} (R² = {r2_score(y_true, y_pred):.2f})')
            plt.grid(True)
            plt.tight_layout()
        # Save plot
        filename_safe = model_file.replace('.pkl', '').replace('.pth', '') + '_plot.png'
        plot_path = os.path.join(plots_path, filename_safe)
        plt.savefig(plot_path)
        plt.close()
        logging.info(f"Plot saved: {plot_path}")
    except Exception as e:
        logging.error(f"Failed to validate {model_file}: {e}")


def validate_all_models():
    """
    Loops through all models in the models folder and validates them.
    """
    setup_logging(logs_path, 'validatator.log')
    model_files = os.listdir(models_path)

    for model_file in model_files:
        targets = determine_target_from_filename(model_file)
        if targets:
            validate_model(model_file, targets)
        else:
            logging.warning(f"Skipping file: {model_file}")


if __name__ == "__main__":
    try:
        validate_all_models()
    except Exception as e:
        logging.error(f"An unexpected error occurred: {e}")
        raise
    logging.info("...DONE!\n\n")