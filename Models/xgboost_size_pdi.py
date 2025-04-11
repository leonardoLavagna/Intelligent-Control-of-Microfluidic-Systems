import pandas as pd
import pickle
import os
import sys
import logging
from sklearn.model_selection import train_test_split, GridSearchCV
from xgboost import XGBRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error


######################################################################################################
# 1. Load Files and Configurations
######################################################################################################
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from _Files.config import logs_path, data_path, models_path, setup_logging


file_name = "data_3"
file_path = os.path.join(data_path, f"{file_name}.csv")
DATA = pd.read_csv(file_path).dropna()


target_variables = ["SIZE", "PDI"]


setup_logging(logs_path, f"xgboost_multi_target.log")
logging.info("STARTING THE XGBOOST MODEL TRAINING PROCESS FOR PREDICTING SIZE AND PDI...")
logging.info(f"---> Loading dataset from {file_path}")


######################################################################################################
# 2. Modeling Function with Grid Search
######################################################################################################
def run_model(seed, best_model_info):
    logging.info(f"STARTING THE XGBOOST MODEL TRAINING PROCESS FOR SEED {seed}...")
    # Prepare features and targets
    features = DATA.drop(columns=target_variables)
    targets = DATA[target_variables]
    categorical_columns = features.select_dtypes(include=["object"]).columns
    # Define preprocessor
    preprocessor = ColumnTransformer(
        transformers=[("cat", OneHotEncoder(drop="first", handle_unknown='ignore'), categorical_columns)],
        remainder="passthrough"
    )

    # Define pipeline
    pipeline = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("regressor", MultiOutputRegressor(XGBRegressor(random_state=seed)))
    ])
    # Hyperparameter grid
    param_grid = {
        'regressor__estimator__n_estimators': [100, 200],
        'regressor__estimator__max_depth': [5, 10],
        'regressor__estimator__learning_rate': [0.01, 0.1],
        'regressor__estimator__subsample': [0.8, 1.0]
    }
    # Grid Search
    grid_search = GridSearchCV(pipeline, param_grid, cv=3, scoring='neg_mean_squared_error', n_jobs=-1)
    grid_search.fit(features, targets)
    best_pipeline = grid_search.best_estimator_
    # Train/Test evaluation
    X_train, X_test, y_train, y_test = train_test_split(features, targets, test_size=0.2, random_state=seed)
    best_pipeline.fit(X_train, y_train)
    # Predictions and evaluation metrics
    y_pred_train = best_pipeline.predict(X_train)
    y_pred_test = best_pipeline.predict(X_test)
    train_r2 = r2_score(y_train, y_pred_train)
    test_r2 = r2_score(y_test, y_pred_test)
    train_mse = mean_squared_error(y_train, y_pred_train)
    test_mse = mean_squared_error(y_test, y_pred_test)
    train_mae = mean_absolute_error(y_train, y_pred_train)
    test_mae = mean_absolute_error(y_test, y_pred_test)
    logging.info(f"Model training completed successfully for seed {seed}.")
    logging.info(f"Training R-squared: {train_r2}, Validation R-squared: {test_r2}")
    logging.info(f"Training Mean Squared Error: {train_mse}, Validation Mean Squared Error: {test_mse}")
    logging.info(f"Training Mean Absolute Error: {train_mae}, Validation Mean Absolute Error: {test_mae}")
    # Validation
    val_path = os.path.join(data_path, "validation.csv")
    logging.info(f"---> Loading validation dataset from {val_path}")
    VALIDATION_DATA = pd.read_csv(val_path)
    validation_features = VALIDATION_DATA.drop(columns=target_variables, errors="ignore")
    validation_targets = VALIDATION_DATA[target_variables]
    validation_preds = best_pipeline.predict(validation_features)
    validation_r2 = r2_score(validation_targets, validation_preds)
    validation_mse = mean_squared_error(validation_targets, validation_preds)
    validation_mae = mean_absolute_error(validation_targets, validation_preds)
    logging.info(f"Validation R-squared for seed {seed}: {validation_r2}")
    logging.info(f"Validation Mean Squared Error for seed {seed}: {validation_mse}")
    logging.info(f"Validation Mean Absolute Error for seed {seed}: {validation_mae}")
    # Update best model if needed
    if train_r2 > best_model_info["best_train_r2"] and validation_r2 > best_model_info["best_val_r2"]:
        best_model_info["best_train_r2"] = train_r2
        best_model_info["best_val_r2"] = validation_r2
        best_model_info["best_seed"] = seed
        best_model_info["best_model"] = best_pipeline
        logging.info(f"New best model found for seed {seed} with Validation R-squared: {test_r2}")
    logging.info("...DONE!\n\n")


######################################################################################################
# 3. Full Pipeline Over Seeds
######################################################################################################
best_model_info = {
    "best_val_r2": -float("inf"),
    "best_train_r2": -float("inf"),
    "best_seed": None,
    "best_model": None
}


seeds = [42, 279, 897, 103, 432, 780, 562, 951, 233, 682]
for seed in seeds:
    run_model(seed, best_model_info)
# Save the best model
if best_model_info["best_model"] is not None:
    model_save_path = os.path.join(models_path, f"best_xgboost_size_pdi_{file_name}.pkl")
    with open(model_save_path, "wb") as f:
        pickle.dump(best_model_info["best_model"], f)
    logging.info(f"Best model saved to {model_save_path}")


logging.info("Training complete for all seeds.")
logging.info(f"Best model is from seed {best_model_info['best_seed']} with Validation R-squared: {best_model_info['best_val_r2']}")
