import pandas as pd
import pickle
import os
import sys
import logging
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import numpy as np


######################################################################################################
# 1. Load Files and Configurations
######################################################################################################
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from _Files.config import logs_path, data_path, models_path, setup_logging


file_path = data_path + "/data_1.csv"
DATA = pd.read_csv(file_path)


setup_logging(logs_path, "random_forest_size_pdi.log")
logging.info("Starting the Random Forest model training and evaluation process...")
logging.info(f"---> Loading dataset from {file_path}")


######################################################################################################
# 2. Modeling
######################################################################################################
features = DATA.drop(columns=["SIZE", "PDI"])
targets = DATA[["SIZE", "PDI"]]
categorical_columns = features.select_dtypes(include=["object"]).columns
numerical_columns = features.select_dtypes(include=["float64", "int64"]).columns
# Preprocessing Pipeline
preprocessor = ColumnTransformer(
    transformers=[("cat", OneHotEncoder(drop="first"), categorical_columns)], 
    remainder="passthrough"
)
# Model Pipeline with Cross-Validation
model = Pipeline(steps=[
    ("preprocessor", preprocessor), 
    ("regressor", MultiOutputRegressor(RandomForestRegressor(n_estimators=200, max_depth=10, random_state=42)))
])
kf = KFold(n_splits=5, shuffle=True, random_state=42)
cv_r2_scores = cross_val_score(model, features, targets, cv=kf, scoring='r2')
cv_mse_scores = cross_val_score(model, features, targets, cv=kf, scoring='neg_mean_squared_error')
cv_mae_scores = cross_val_score(model, features, targets, cv=kf, scoring='neg_mean_absolute_error')
logging.info("Cross-validation results:")
logging.info(f"Average R-squared: {np.mean(cv_r2_scores):.4f}")
logging.info(f"Average Mean Squared Error: {-np.mean(cv_mse_scores):.4f}")
logging.info(f"Average Mean Absolute Error: {-np.mean(cv_mae_scores):.4f}")
# Train Final Model on Full Training Set
X_train, X_test, y_train, y_test = train_test_split(features, targets, test_size=0.2, random_state=42)
model.fit(X_train, y_train)
# Predict on Test Set
y_pred = model.predict(X_test)
r2 = r2_score(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
logging.info("Final model evaluation:")
logging.info(f"R-squared: {r2:.4f}")
logging.info(f"Mean Squared Error: {mse:.4f}")
logging.info(f"Mean Absolute Error: {mae:.4f}")
# Save Model
model_path = os.path.join(models_path, "random_forest_size_pdi.pkl")
with open(model_path, "wb") as f:
    pickle.dump(model, f)
logging.info(f"Model saved successfully at {model_path}")


######################################################################################################
# 3. Validation
######################################################################################################
new_file_path = data_path + "/validation.csv"
logging.info(f"---> Loading validation dataset from {new_file_path}")
VALIDATION_DATA = pd.read_csv(new_file_path)
validation_features = VALIDATION_DATA.drop(columns=["SIZE", "PDI"], errors="ignore")
validation_targets = VALIDATION_DATA[["SIZE", "PDI"]]
# Predict on Validation Data
validation_predictions = model.predict(validation_features)
# Evaluate Validation Performance
validation_r2 = r2_score(validation_targets, validation_predictions)
validation_mse = mean_squared_error(validation_targets, validation_predictions)
validation_mae = mean_absolute_error(validation_targets, validation_predictions)
logging.info("Model validation completed successfully.")
logging.info(f"Validation R-squared: {validation_r2:.4f}")
logging.info(f"Validation Mean Squared Error: {validation_mse:.4f}")
logging.info(f"Validation Mean Absolute Error: {validation_mae:.4f}")
logging.info("...DONE!\n\n")