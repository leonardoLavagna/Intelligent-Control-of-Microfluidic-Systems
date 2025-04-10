import os
import sys
import logging
import pickle
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.neural_network import MLPRegressor


######################################################################################################
# 1. Load data and Configurations
######################################################################################################
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from _Files.config import logs_path, data_path, models_path, setup_logging


setup_logging(logs_path, "vae_regressor_size.log")


file_path = os.path.join(data_path, "data_1.csv")
DATA = pd.read_csv(file_path)
logging.info(f"STARTING VAE + REGRESSOR TRAINING for {file_path}...")


######################################################################################################
# 2. Model setup
######################################################################################################
features = DATA.drop(columns=["SIZE"])
targets = DATA[["SIZE"]]
categorical_columns = features.select_dtypes(include=["object"]).columns
numerical_columns = features.select_dtypes(include=["float64", "int64"]).columns
# Preprocessing pipeline
preprocessor = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(drop="first", handle_unknown="ignore"), categorical_columns),
        ("num", StandardScaler(), numerical_columns)
    ]
)
X_preprocessed = preprocessor.fit_transform(features)
X_train, X_test, y_train, y_test = train_test_split(X_preprocessed, targets, test_size=0.2, random_state=42)
input_dim = X_train.shape[1]
latent_dim = 20
# Convert to PyTorch tensors
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test.values, dtype=torch.float32)


######################################################################################################
# 3. Define VAE model with adjustments
######################################################################################################
class VAE(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super(VAE, self).__init__()
        # Encoder network
        self.fc1 = nn.Linear(input_dim, 128)  
        self.dropout1 = nn.Dropout(0.2)  
        self.fc21 = nn.Linear(128, latent_dim)  
        self.fc22 = nn.Linear(128, latent_dim)  
        # Decoder network
        self.fc3 = nn.Linear(latent_dim, 128)  
        self.dropout2 = nn.Dropout(0.2)  
        self.fc4 = nn.Linear(128, input_dim)  
    
    def encode(self, x):
        h1 = torch.relu(self.fc1(x))
        h1 = self.dropout1(h1)  
        z_mean = self.fc21(h1)
        z_log_var = self.fc22(h1)
        return z_mean, z_log_var
    
    def reparameterize(self, z_mean, z_log_var):
        std = torch.exp(0.5 * z_log_var)
        eps = torch.randn_like(std)
        return z_mean + eps * std
    
    def decode(self, z):
        h3 = torch.relu(self.fc3(z))
        h3 = self.dropout2(h3)  
        return torch.sigmoid(self.fc4(h3))
    
    def forward(self, x):
        z_mean, z_log_var = self.encode(x)
        z = self.reparameterize(z_mean, z_log_var)
        return self.decode(z), z_mean, z_log_var

def vae_loss(recon_x, x, z_mean, z_log_var):
    MSE = nn.functional.mse_loss(recon_x, x.view(-1, input_dim), reduction='sum')
    KL = torch.sum(0.5 * (z_log_var.exp() + z_mean.pow(2) - 1 - z_log_var))
    return MSE + KL

vae = VAE(input_dim, latent_dim)
optimizer = optim.Adam(vae.parameters(), lr=1e-3)
# Learning rate scheduler 
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
# Train 
num_epochs = 1000 
for epoch in range(num_epochs):
    vae.train()
    optimizer.zero_grad()
    recon_batch, z_mean, z_log_var = vae(X_train_tensor)
    loss = vae_loss(recon_batch, X_train_tensor, z_mean, z_log_var)
    loss.backward()
    optimizer.step()
    # Step the scheduler
    scheduler.step()
    if epoch % 100 == 0:
        logging.info(f"Epoch {epoch}, Loss {loss.item()}")
logging.info("VAE training completed.")


######################################################################################################
# 4. Encode latent features and train MLP for regression
######################################################################################################
vae.eval()
with torch.no_grad():
    X_train_latent = vae.encode(X_train_tensor)[0] 
    X_test_latent = vae.encode(X_test_tensor)[0]
# Convert the latent vectors to numpy for regression
X_train_latent = X_train_latent.numpy()
X_test_latent = X_test_latent.numpy()
# Define the hyperparameter grid for MLPRegressor
param_grid = {
    'hidden_layer_sizes': [(100,), (100, 50), (150, 75), (50, 25)],
    'activation': ['relu', 'tanh'],
    'solver': ['adam', 'sgd'],
    'max_iter': [500, 1000],
    'alpha': [0.0001, 0.001, 0.01]
}
# Set up GridSearchCV
grid_search = GridSearchCV(estimator=MLPRegressor(random_state=42),
                           param_grid=param_grid,
                           scoring='r2',   # Or 'neg_mean_squared_error' for MSE
                           cv=3,           # 3-fold cross-validation
                           n_jobs=-1,      # Use all CPU cores
                           verbose=2)      # Show detailed output
# Fit GridSearchCV to the latent training data
grid_search.fit(X_train_latent, y_train)
# Get the best parameters and best score
best_params = grid_search.best_params_
best_score = grid_search.best_score_
logging.info(f"Best Parameters: {best_params}")
logging.info(f"Best Cross-validation R2: {best_score}")
# Retrain the regressor with the best parameters
best_regressor = grid_search.best_estimator_
# Predict and evaluate on the validation set
val_path = os.path.join(data_path, "validation.csv")
VAL = pd.read_csv(val_path)
val_features = VAL.drop(columns=["SIZE"], errors="ignore")
val_targets = VAL[["SIZE"]]
X_val_preprocessed = preprocessor.transform(val_features)
X_val_tensor = torch.tensor(X_val_preprocessed, dtype=torch.float32)
vae.eval()
with torch.no_grad():
    X_val_latent = vae.encode(X_val_tensor)[0]
X_val_latent = X_val_latent.numpy()
val_predictions = best_regressor.predict(X_val_latent)
val_r2 = r2_score(val_targets, val_predictions)
val_mse = mean_squared_error(val_targets, val_predictions)
val_mae = mean_absolute_error(val_targets, val_predictions)
logging.info("Validation completed.")
logging.info(f"Validation R2: {val_r2}")
logging.info(f"Validation MSE: {val_mse}")
logging.info(f"Validation MAE: {val_mae}")
# Save the best regressor
with open(os.path.join(models_path, "best_latent_regressor.pkl"), "wb") as f:
    pickle.dump(best_regressor, f)
# Save the VAE model
torch.save(vae.state_dict(), os.path.join(models_path, "vae_encoder.pth"))
logging.info("...DONE!\n")