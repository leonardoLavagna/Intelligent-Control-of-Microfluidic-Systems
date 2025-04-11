import pandas as pd
import pickle
import os
import sys
import logging
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.multioutput import MultiOutputRegressor
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error


######################################################################################################
# 1. Load Files and Configurations
######################################################################################################
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from _Files.config import logs_path, data_path, models_path, setup_logging


file_name = "data_3"
file_path = data_path + f"/{file_name}.csv"
DATA = pd.read_csv(file_path).dropna()


setup_logging(logs_path, "mlp_size_pdi.log")
logging.info("Starting the MLP model training and evaluation process...".upper())
logging.info(f"---> Loading dataset from {file_path}")


######################################################################################################
# 2. Modeling (MLP with Grid Search)
######################################################################################################
features = DATA.drop(columns=["SIZE", "PDI"])
targets = DATA[["SIZE", "PDI"]]
categorical_columns = features.select_dtypes(include=["object"]).columns
numerical_columns = features.select_dtypes(include=["float64", "int64"]).columns
# Define preprocessing pipeline
preprocessor = ColumnTransformer(
    transformers=[("cat", OneHotEncoder(drop="first"), categorical_columns)],
    remainder="passthrough"
)
X_train, X_test, y_train, y_test = train_test_split(
    features, targets, test_size=0.2, random_state=42
)
# Define MLP model
mlp = MultiOutputRegressor(MLPRegressor(max_iter=1000, random_state=42))
pipeline = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("regressor", mlp)
])
# Define hyperparameter grid
param_grid = {
    "regressor__estimator__hidden_layer_sizes": [(50,), (100,), (50, 50)],
    "regressor__estimator__activation": ["relu", "tanh"],
    "regressor__estimator__alpha": [0.0001, 0.001],
    "regressor__estimator__learning_rate_init": [0.001, 0.01]
}
# Grid Search
grid_search = GridSearchCV(pipeline, param_grid, cv=5, scoring="neg_mean_squared_error", n_jobs=-1)
grid_search.fit(X_train, y_train)
# Predict and evaluate on test data
y_pred = grid_search.predict(X_test)
r2 = r2_score(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
logging.info("MLP model training and grid search completed successfully.")
logging.info(f"Best Parameters: {grid_search.best_params_}")
logging.info(f"Test R-squared: {r2}")
logging.info(f"Test Mean Squared Error: {mse}")
logging.info(f"Test Mean Absolute Error: {mae}")
# Save the best model
model_path = os.path.join(models_path, f"mlp_size_pdi_{file_name}.pkl")
with open(model_path, "wb") as f:
    pickle.dump(grid_search.best_estimator_, f)
logging.info(f"Best model saved successfully at {model_path}")


######################################################################################################
# 3. Validation
######################################################################################################
new_file_path = data_path + "/validation.csv"
logging.info(f"---> Loading validation dataset from {new_file_path}")
# Load validation data 
VALIDATION_DATA = pd.read_csv(new_file_path)
validation_features = VALIDATION_DATA.drop(columns=["SIZE", "PDI"], errors="ignore")
validation_targets = VALIDATION_DATA[["SIZE", "PDI"]]
# Load the trained model
with open(model_path, "rb") as f:
    loaded_model = pickle.load(f)
# Predict and evaluate
validation_predictions = loaded_model.predict(validation_features)
validation_r2 = r2_score(validation_targets, validation_predictions)
validation_mse = mean_squared_error(validation_targets, validation_predictions)
validation_mae = mean_absolute_error(validation_targets, validation_predictions)
logging.info("Model validation completed successfully.")
logging.info(f"Validation R-squared: {validation_r2}")
logging.info(f"Validation Mean Squared Error: {validation_mse}")
logging.info(f"Validation Mean Absolute Error: {validation_mae}")
logging.info("...VALIDATION DONE!\n\n")