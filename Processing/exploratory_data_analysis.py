import pandas as pd
import matplotlib.pyplot as plt
import logging
import io
import os
import sys
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import umap
import plotly.graph_objects as go


######################################################################################################
# 1. Load Files and Configurations
######################################################################################################
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from _Files.config import *


setup_logging(logs_path, "exploratory_data_analysis.log")



DATA_PATH = data_path
PLOT_DIR = plots_path
seed_filename= "seed.csv"
csv_files = [f for f in os.listdir(DATA_PATH) if f.endswith('.csv')]
data_dict = {}
for file in csv_files:
    file_path = os.path.join(DATA_PATH, file)
    data_dict[file] = pd.read_csv(file_path).dropna()


######################################################################################################
# 2. Processing and Saving  Functions
######################################################################################################
def save_plot(fig, plot_name):
    """Save plots with proper filename handling."""
    fig.savefig(f"{PLOT_DIR}/{plot_name}.png")
    plt.close()
    logging.info("---> Plot saved at: %s", PLOT_DIR)


def standardize_data(data):
    """Standardize the data for PCA and UMAP."""
    scaler = StandardScaler()
    return scaler.fit_transform(data)


def data_explorer(data, filename):
    """Perform exploratory data analysis (EDA) on the given dataset."""
    buffer = io.StringIO()
    data.info(buf=buffer)
    logging.info(f"Dataset Info for {filename}:\n{buffer.getvalue()}")
    logging.info(f"Summary Statistics for {filename}:\n{data.describe().to_string()}")
    logging.info(f"First 5 Rows for {filename}:\n{data.head().to_string()}")
    data.hist(bins=30, figsize=(12, 10))
    plt.suptitle(f"Feature Distributions for {filename}")
    save_plot(plt, f"{filename}_complete_data_histogram")
    logging.info("---> Feature distributions at: %s", f"{PLOT_DIR}/{filename}_complete_data_histogram.png")


def correlation_heatmap(data, filename):
    """Get correlation heatmap."""
    numeric_data = data.select_dtypes(include=['number']).dropna()
    corr_matrix = numeric_data.corr()
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f", ax=ax)
    plt.title(f"Feature Correlation Heatmap for {filename}")
    save_plot(fig, f"{filename}_correlation_heatmap")
    logging.info("---> Correlation heatmap at: %s", f"{PLOT_DIR}/{filename}_correlation_heatmap.png")


def plot_numeric_distributions(seed_data, comparison_data, seed_filename, comparison_filename):
    """Plot and save the KDE distributions for all numeric columns, comparing datasets."""
    numeric_seed_data = seed_data.select_dtypes(include=['number']).dropna()
    numeric_comparison_data = comparison_data.select_dtypes(include=['number']).dropna()
    for column in numeric_seed_data.columns:
        plt.figure(figsize=(6, 4))
        sns.kdeplot(numeric_seed_data[column], label=f"{seed_filename} (Seed)", shade=True, color="skyblue")
        sns.kdeplot(numeric_comparison_data[column], label=f"{comparison_filename} (Comparison)", shade=True, color="orange")
        plt.title(f"Feature Distribution: {column}")
        plt.legend()
        save_plot(plt, f"{column}_{seed_filename}_vs_{comparison_filename}_distribution_comparison")
        logging.info("---> Distribution plot saved for: %s", column)


def categorize_size(size):
    """Categorize size into Small (S), Medium (M), and Large (L)."""
    if size < 100:
        return "S"
    elif 100 <= size < 200:
        return "M"
    else:
        return "L"


def categorize_pdi(pdi):
    """Categorize PDI into High Molecular Dispersion (HMD), Medium Dispersion (MD), and Poor Dispersion (PLD)."""
    if pdi < 0.1:
        return "HMD"
    elif 0.1 <= pdi < 0.25:
        return "MD"
    else:
        return "PLD"


def sankey_diagram(data, source="HSPC", target="SIZE", filename="sankey_plot"):
    """Generate and save an alluvial (Sankey) plot."""
    logging.info("Calculating Sankey plot for %s...", filename)
    data = data.copy().dropna()
    data["SIZE"] = data["SIZE"].apply(categorize_size)
    data["PDI"] = data["PDI"].apply(categorize_pdi)
    value_counts = data.groupby([source, target]).size().reset_index(name='value')
    categories = list(set(value_counts[source]).union(set(value_counts[target])))
    category_to_index = {category: i for i, category in enumerate(categories)}
    sources = value_counts[source].map(category_to_index).tolist()
    targets = value_counts[target].map(category_to_index).tolist()
    values = value_counts['value'].tolist()
    fig = go.Figure(data=[go.Sankey(
        node=dict(pad=15, thickness=20, line=dict(color="black", width=0.5), label=categories),
        link=dict(source=sources, target=targets, value=values)
    )])
    fig.update_layout(title_text=f"Sankey Plot for {filename}")
    fig.write_html(f"{PLOT_DIR}/{filename}_sankey_plot.html")
    logging.info(f"Sankey plot saved as HTML for {filename}.")
    logging.info("---> Alluvial plot at: %s", f"{PLOT_DIR}/{filename}_sankey_plot.html")


def feature_importance(data, target_features=["SIZE", "PDI"], filename="feature_importance"):
    """Train a Random Forest model and save feature importance plots."""
    logging.info(f"Calculating feature importance for {filename}...")
    numeric_data = data.select_dtypes(include=['number'])
    X = numeric_data.drop(columns=target_features)
    for target_feature in target_features:
        y = numeric_data[target_feature]
        rf = RandomForestRegressor(n_estimators=100, random_state=42)
        rf.fit(X, y)
        feature_importances = pd.DataFrame({
            'Feature': X.columns,
            'Importance': rf.feature_importances_
        })
        feature_importances = feature_importances.sort_values(by='Importance', ascending=False)
        fig = plt.figure(figsize=(10, 6))
        sns.barplot(x='Importance', y='Feature', data=feature_importances, palette='viridis', hue='Feature', legend=False)
        plt.title(f"Feature Importance for {target_feature} in {filename}")
        save_plot(fig, f"{filename}_feature_importance_{target_feature}")
        logging.info("---> Feature importance at: %s", f"{PLOT_DIR}/{filename}_feature_importance_{target_feature}.png")


def pca_umap_clustering(data, filename="umap_clustering"):
    """Perform PCA and UMAP with K-Means clustering for any dataset."""
    logging.info(f"Performing PCA and UMAP clustering for {filename}...")
    numeric_data = data.select_dtypes(include=['number'])
    standardized_data = standardize_data(numeric_data)
    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(standardized_data)
    df_pca = pd.DataFrame(pca_result, columns=["PC1", "PC2"])
    plt.figure(figsize=(8, 6))
    sns.kdeplot(x=df_pca["PC1"], y=df_pca["PC2"], fill=True, cmap="Blues", thresh=0.05)
    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    plt.title(f"2D Density Plot of PCA Transformed Data for {filename}")
    save_plot(plt, f"{filename}_pca_density_plot")
    logging.info("---> PCA density plot at: %s", f"{PLOT_DIR}/{filename}_pca_density_plot.png")
    umap_model = umap.UMAP(n_components=2, random_state=42)
    umap_result = umap_model.fit_transform(standardized_data)
    kmeans = KMeans(n_clusters=3, random_state=42)
    clusters = kmeans.fit_predict(umap_result)
    df_umap = pd.DataFrame(umap_result, columns=["UMAP1", "UMAP2"])
    df_umap["Cluster"] = clusters
    fig = plt.figure(figsize=(10, 8))
    sns.scatterplot(x=df_umap["UMAP1"], y=df_umap["UMAP2"], hue=df_umap["Cluster"], palette="tab10", alpha=0.7)
    plt.title(f"UMAP Projection with K-Means Clustering for {filename}")
    save_plot(fig, f"{filename}_umap_clustering")
    logging.info("---> UMAP clustering at: %s", f"{PLOT_DIR}/{filename}_umap_clustering.png")


######################################################################################################
# 3. Pipeline
######################################################################################################
def run_pipeline():
    """Main function to execute the exploratory data analysis pipeline."""
    logging.info("Starting Exploratory Data Analysis (EDA)...".upper())
    for filename, data in data_dict.items():
        logging.info(f"Exploring {filename} dataset...")
        data_explorer(data=data, filename=filename)
    seed_data = data_dict.get(seed_filename) 
    for filename, data in data_dict.items():
        if filename != seed_filename:
            logging.info(f"Plotting numeric distributions ({seed_filename} vs {filename})...")
            plot_numeric_distributions(seed_data=seed_data, comparison_data=data, 
                                       seed_filename=seed_filename, comparison_filename=filename)
    for filename, data in data_dict.items():
        correlation_heatmap(data=data, filename=filename)
    for filename, data in data_dict.items():
        sankey_diagram(data=data, filename=filename)
    for filename, data in data_dict.items():
        feature_importance(data=data, filename=filename)
    for filename, data in data_dict.items():
        pca_umap_clustering(data=data, filename=filename)


if __name__ == "__main__":
    run_pipeline()
    logging.info("...Done!\n\n".upper())
