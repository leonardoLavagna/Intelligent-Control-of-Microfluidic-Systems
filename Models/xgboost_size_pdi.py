import pandas as pd
import pickle
import os
import sys
import logging
from sklearn.model_selection import train_test_split
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


file_path = data_path + "/data_1.csv"
DATA = pd.read_csv(file_path)


setup_logging(logs_path, "xgboost_size_pdi.log")
logging.info("Starting the XGBoost model training and evaluation process...".upper())
logging.info(f"---> Loading dataset from {file_path}")


######################################################################################################
# 2. Modeling
######################################################################################################
features = DATA.drop(columns=["SIZE", "PDI"])
targets = DATA[["SIZE", "PDI"]]
categorical_columns = features.select_dtypes(include=["object"]).columns
numerical_columns = features.select_dtypes(include=["float64", "int64"]).columns
# Define the preprocessing pipeline
preprocessor = ColumnTransformer(
    transformers=[("cat", OneHotEncoder(drop="first"), categorical_columns)], 
    remainder="passthrough"
)
X_train, X_test, y_train, y_test = train_test_split(features, targets, test_size=0.2, random_state=42)
xgboost_model = Pipeline(steps=[("preprocessor", preprocessor), ("regressor", MultiOutputRegressor(XGBRegressor(n_estimators=100, random_state=42)))])
xgboost_model.fit(X_train, y_train)
# Predict and evaluate on the test set
y_pred = xgboost_model.predict(X_test)
r2 = r2_score(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
logging.info("Model training completed successfully.")
logging.info(f"R-squared: {r2}")
logging.info(f"Mean Squared Error: {mse}")  
logging.info(f"Mean Absolute Error: {mae}")
# Save the model
model_path = os.path.join(models_path, "xgboost_micromixer_size_pdi.pkl")
with open(model_path, "wb") as f:
    pickle.dump(xgboost_model, f)
logging.info(f"Model saved successfully at {model_path}")


######################################################################################################
# 3. Validation
######################################################################################################
new_file_path = data_path + "/validation.csv"
logging.info(f"---> Loading validation dataset from {new_file_path}")
# Load validation data
VALIDATION_DATA = pd.read_csv(new_file_path)
# Separate features and targets
validation_features = VALIDATION_DATA.drop(columns=["SIZE", "PDI"], errors="ignore")
validation_targets = VALIDATION_DATA[["SIZE", "PDI"]]
# Load the trained model
with open(model_path, "rb") as f:
    loaded_model = pickle.load(f)
# Predict on validation data
validation_predictions = loaded_model.predict(validation_features)
# Evaluate on validation data
validation_r2 = r2_score(validation_targets, validation_predictions)
validation_mse = mean_squared_error(validation_targets, validation_predictions)
validation_mae = mean_absolute_error(validation_targets, validation_predictions)
# Log validation results
logging.info("Model validation completed successfully.")
logging.info(f"Validation R-squared: {validation_r2}")
logging.info(f"Validation Mean Squared Error: {validation_mse}")
logging.info(f"Validation Mean Absolute Error: {validation_mae}")
logging.info("...VALIDATION DONE!\n\n")