import pandas as pd
import pickle
import os
import sys
import logging
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.ensemble import RandomForestRegressor  
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
file_path = data_path + f"/{file_name}.csv"
DATA = pd.read_csv(file_path).dropna()
target_variable = "SIZE"  


setup_logging(logs_path, f"random_forest_one_target_{target_variable.lower()}.log")  
logging.info(f"Starting the RandomForest model training process for predicting {target_variable}...".upper())
logging.info(f"---> Loading dataset from {file_path}")


######################################################################################################
# 2. Modeling with Cross Validation and Grid Search
######################################################################################################
def run_model(seed, best_model_info):
    """
    Train and evaluate a RandomForest model using cross-validation and hyperparameter tuning.

    Args:
        seed (int): Random seed for reproducibility of train-test splits and model training.
        best_model_info (dict): A dictionary to track and store information about the best model.
            Should contain the following keys:
            - "best_train_r2" (float)
            - "best_val_r2" (float)
            - "best_seed" (int)
            - "best_model" (sklearn estimator or None)

    Returns:
        None
    """
    logging.info(f"Starting the RandomForest model training process for seed {seed}...".upper())  
    # Prepare features and targets
    features = DATA.drop(columns=[target_variable])
    targets = DATA[[target_variable]]
    categorical_columns = features.select_dtypes(include=["object"]).columns
    numerical_columns = features.select_dtypes(include=["float64", "int64"]).columns
    preprocessor = ColumnTransformer(
        transformers=[("cat", OneHotEncoder(drop="first"), categorical_columns)], 
        remainder="passthrough"
    )
    # Define the pipeline
    pipeline = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("regressor", MultiOutputRegressor(RandomForestRegressor(random_state=seed)))  
    ])
    # Grid search
    param_grid = {
        'regressor__estimator__n_estimators': [50, 100, 200],
        'regressor__estimator__max_depth': [3, 5, 7],
        'regressor__estimator__min_samples_split': [2, 5, 10],
        'regressor__estimator__min_samples_leaf': [1, 2, 4]
    }
    grid_search = GridSearchCV(pipeline, param_grid, cv=5, scoring='neg_mean_squared_error', n_jobs=-1)
    # Cross Validation
    cross_val_score(pipeline, features, targets, cv=5, scoring='neg_mean_squared_error')
    # Train with the best parameters
    grid_search.fit(features, targets)
    best_pipeline = grid_search.best_estimator_
    X_train, X_test, y_train, y_test = train_test_split(features, targets, test_size=0.2, random_state=seed)
    best_pipeline.fit(X_train, y_train)
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
    new_file_path = data_path + "/validation.csv"
    logging.info(f"---> Loading validation dataset from {new_file_path}")
    VALIDATION_DATA = pd.read_csv(new_file_path)
    validation_features = VALIDATION_DATA.drop(columns=[target_variable], errors="ignore")
    validation_targets = VALIDATION_DATA[[target_variable]]
    validation_predictions = best_pipeline.predict(validation_features)
    validation_r2 = r2_score(validation_targets, validation_predictions)
    validation_mse = mean_squared_error(validation_targets, validation_predictions)
    validation_mae = mean_absolute_error(validation_targets, validation_predictions)
    logging.info(f"Validation R-squared for seed {seed}: {validation_r2}")
    logging.info(f"Validation Mean Squared Error for seed {seed}: {validation_mse}")
    logging.info(f"Validation Mean Absolute Error for seed {seed}: {validation_mae}")
    if train_r2 > best_model_info["best_train_r2"] and validation_r2 > best_model_info["best_val_r2"]:
        best_model_info["best_train_r2"] = train_r2
        best_model_info["best_val_r2"] = validation_r2
        best_model_info["best_seed"] = seed
        best_model_info["best_model"] = best_pipeline
        logging.info(f"New best model found for seed {seed} with Training R-squared: {train_r2} and Validation R-squared: {validation_r2}")
    logging.info("...DONE!\n\n")
    # Save the best model after training
    if best_model_info["best_model"] is not None:
        model_save_path = models_path + f"/best_random_forest_model_{file_name}_{target_variable.lower()}.pkl"
        with open(model_save_path, 'wb') as model_file:
            pickle.dump(best_model_info["best_model"], model_file)
        logging.info(f"Best model saved to {model_save_path}")


######################################################################################################
# 3. Complete pipeline
######################################################################################################
best_model_info = {
    "best_train_r2": -float("inf"),  
    "best_val_r2": -float("inf"),    
    "best_seed": None,
    "best_model": None
}


seeds = [42, 279, 897, 103, 432, 780, 562, 951, 233, 682]
for seed in seeds:
    run_model(seed, best_model_info)


logging.info("Training complete for all seeds.")
logging.info(f"Best model is from seed {best_model_info['best_seed']} with Training R-squared: {best_model_info['best_train_r2']} and Validation R-squared: {best_model_info['best_val_r2']}")
