import numpy as np
import pandas as pd
import os
import sys
import logging
from joblib import load
from sklearn.metrics import r2_score
import pickle
from sklearn.metrics import mean_squared_error, mean_absolute_error

######################################################################################################
# 1. Load Files and Configurations
######################################################################################################
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from _Files.config import logs_path, data_path, models_path, setup_logging


setup_logging(logs_path, "mixture_of_experts.log")
logging.info("Starting the Iterative Refinement model training and evaluation process...".upper())
file_name = "data_1"
file_path = data_path + f"/{file_name}.csv"
logging.info(f"---> Loading training dataset from {file_path}")
DATA = pd.read_csv(file_path).dropna()
validation_file_path = data_path + "/validation.csv"
logging.info(f"---> Loading validation dataset from {validation_file_path}")
validation_data = pd.read_csv(validation_file_path).dropna()


X_train = DATA.drop(columns=["SIZE", "PDI"])
y_train_size = DATA["SIZE"]
y_train_pdi = DATA["PDI"]
X_validation = validation_data.drop(columns=["SIZE", "PDI"])

rf_path = models_path + '/best_random_forest_size_pdi_data_1.pkl'
xgb_path_size = models_path + '/best_xgboost_model__data_1_size.pkl'
xgb_path_pdi = models_path + '/best_xgboost_model__data_1_pdi.pkl'
logging.info(f"---> Loading pre-trained models from {rf_path}, {xgb_path_size}, {xgb_path_pdi}")
rf_model = load(rf_path)
xgb_size = load(xgb_path_size)
xgb_pdi = load(xgb_path_pdi)


######################################################################################################
# 2. Mixture of Experts (Iterative Refinement)
######################################################################################################
def iterative_refinement(X_raw, num_epochs=2):
    logging.info(f"Starting iterative refinement with {num_epochs} epochs.")
    X = X_raw.copy()
    # Initial predictions using weak model
    logging.info("Making initial predictions using Random Forest model.")
    initial_preds = rf_model.predict(X)
    size_pred = initial_preds[:, 0].ravel() 
    pdi_pred = initial_preds[:, 1].ravel() 
    # Iterative refinement
    for epoch in range(num_epochs):
        logging.info(f"Epoch {epoch+1}/{num_epochs}")
        # Predict SIZE using PDI
        X_with_pdi = X.copy()
        X_with_pdi['PDI'] = pdi_pred
        size_pred = xgb_size.predict(X_with_pdi).ravel() 
        # Predict PDI using new SIZE
        X_with_size = X.copy()
        X_with_size['SIZE'] = size_pred
        pdi_pred = xgb_pdi.predict(X_with_size).ravel()  

    logging.info("Completed iterative refinement.")
    return pd.DataFrame({
        'Refined_SIZE': size_pred,
        'Refined_PDI': pdi_pred
    })


# Apply iterative refinement to the validation set
logging.info("Starting iterative refinement for validation dataset.")
refined_validation_preds = iterative_refinement(X_validation, num_epochs=5)
y_val_size = validation_data["SIZE"]
y_val_pdi = validation_data["PDI"]


# Evaluate performance on validation data
validation_r2_size = r2_score(y_val_size, refined_validation_preds['Refined_SIZE'])
validation_r2_pdi = r2_score(y_val_pdi, refined_validation_preds['Refined_PDI'])
validation_mse_size = mean_squared_error(y_val_size, refined_validation_preds['Refined_SIZE'])
validation_mse_pdi = mean_squared_error(y_val_pdi, refined_validation_preds['Refined_PDI'])
validation_mae_size = mean_absolute_error(y_val_size, refined_validation_preds['Refined_SIZE'])
validation_mae_pdi = mean_absolute_error(y_val_pdi, refined_validation_preds['Refined_PDI'])
logging.info("Validation completed successfully.")
logging.info(f"Validation R² for SIZE: {validation_r2_size}")
logging.info(f"Validation R² for PDI: {validation_r2_pdi}")
logging.info(f"Validation MSE for SIZE: {validation_mse_size}")
logging.info(f"Validation MSE for PDI: {validation_mse_pdi}")
logging.info(f"Validation MAE for SIZE: {validation_mae_size}")
logging.info(f"Validation MAE for PDI: {validation_mae_pdi}")


######################################################################################################
# 3. Save Refined Model
######################################################################################################
model_path = os.path.join(models_path, f"refined_model_size_pdi.pkl")
logging.info(f"Saving the refined model at {model_path}")
with open(model_path, "wb") as f:
    pickle.dump({
        'rf_model': rf_model,
        'xgb_size': xgb_size,
        'xgb_pdi': xgb_pdi
    }, f)
logging.info("Refined model saved successfully.")
logging.info("...VALIDATION DONE!\n\n")
