# # -*- coding: utf-8 -*-
# """
# RegressorDetect_with_HyperparamTuning_AddPoints3and6.ipynb

# This script demonstrates:
# - More complex NN architectures (Point #3: Complexity / Underfitting)
# - Adjusting alpha & early stopping (Point #6: Regularization or early stopping)
# """

# import os
# import ast
# import pandas as pd
# import numpy as np
# import seaborn as sns
# import matplotlib.pyplot as plt

# from sklearn.model_selection import train_test_split, GridSearchCV
# from sklearn.neural_network import MLPRegressor
# from sklearn.metrics import mean_squared_error, r2_score
# from sklearn.preprocessing import StandardScaler
# from sklearn.decomposition import PCA
# from sklearn.pipeline import Pipeline

# # --------------------------------------------------------------------------------------
# # 1. Create directories for saving outputs
# # --------------------------------------------------------------------------------------
# RESULTS_DIR = "mlp_results"
# PLOTS_DIR = os.path.join(RESULTS_DIR, "plots")
# METRICS_CSV = os.path.join(RESULTS_DIR, "model_metrics.csv")

# # Create if they don't exist
# os.makedirs(PLOTS_DIR, exist_ok=True)

# # --------------------------------------------------------------------------------------
# # 2. Load Data
# # --------------------------------------------------------------------------------------
# df = pd.read_csv('/home/user/Desktop/CODE/Individual_Regressor/predictRegressor/updated_file_Predict.csv')
# print("Data Loaded. Head of the DataFrame:")
# print(df.head())

# # --------------------------------------------------------------------------------------
# # 3. Basic EDA and filtering
# # --------------------------------------------------------------------------------------
# print("\nDataFrame Columns:")
# print(df.columns)

# # Plot the distribution of Processing Time
# plt.figure(figsize=(10, 6))
# sns.histplot(df['Processing Time'], kde=True)
# plt.title('Distribution of Processing Time')
# plt.xlabel('Processing Time')
# plt.ylabel('Frequency')
# plt.savefig(os.path.join(PLOTS_DIR, 'processing_time_distribution_initial.png'))
# plt.show()

# # Print distribution counts
# distribution_counts = df['Processing Time'].value_counts().sort_index()
# print("Distribution Counts:")
# print(distribution_counts)

# # Filter out Processing Time > 20
# df = df[df['Processing Time'] <= 20]

# # Plot the distribution again after filtering
# plt.figure(figsize=(10, 6))
# sns.histplot(df['Processing Time'], kde=True)
# plt.title('Distribution of Processing Time (Filtered <= 20)')
# plt.xlabel('Processing Time')
# plt.ylabel('Frequency')
# plt.savefig(os.path.join(PLOTS_DIR, 'processing_time_distribution_filtered.png'))
# plt.show()

# # Print new distribution counts
# distribution_counts = df['Processing Time'].value_counts().sort_index()
# print("Distribution Counts after filtering:")
# print(distribution_counts)

# # --------------------------------------------------------------------------------------
# # 4. Drop unwanted columns
# # --------------------------------------------------------------------------------------
# columns_to_drop = [
#     'Image No',
#     'Iteration',
#     'Execution Time (seconds)',
#     'File',
#     'Duration'
# ]
# df.drop(columns=columns_to_drop, inplace=True, errors='ignore')

# # --------------------------------------------------------------------------------------
# # 5. Separate features (X) and target (y)
# # --------------------------------------------------------------------------------------
# y = df['Processing Time']
# X = df.drop(columns=['Processing Time'])

# print("\nFeatures (X) head:")
# print(X.head())
# print("\nTarget (y) head:")
# print(y.head())

# # --------------------------------------------------------------------------------------
# # 6. Data Cleaning / Feature Engineering
# # --------------------------------------------------------------------------------------
# print("\nData Types of X:")
# print(X.dtypes)

# # Split 'Image Pixel' into width and height, then create image_size
# X[['width', 'height']] = X['Image Pixel'].str.split('x', expand=True).astype(int)
# X['image_size'] = X['width'] * X['height']
# X.drop(columns=['Image Pixel'], inplace=True)

# # Optionally drop columns if not needed
# # (Removing 'Current Script' and 'Total CPU Cores' if they exist)
# X.drop(columns=['Current Script','Total CPU Cores'], inplace=True, errors='ignore')

# # One-hot encoding of categorical columns
# categorical_data = ['CPU Model','GPU Model']
# encoder = pd.get_dummies(X[categorical_data], drop_first=False)
# X = pd.concat([X, encoder], axis=1)
# X.drop(columns=categorical_data, inplace=True)

# # Count occurrences of p, s, d in 'Combination'
# X['num_p'] = X['Combination'].str.count('p')
# X['num_s'] = X['Combination'].str.count('s')
# X['num_d'] = X['Combination'].str.count('d')
# X.drop(columns=['Combination'], inplace=True)

# # Convert CPU Usage Per Core from string to list
# X['CPU Usage Per Core'] = X['CPU Usage Per Core'].apply(
#     lambda x: ast.literal_eval(x) if isinstance(x, str) else x
# )

# # Expand CPU usage into separate columns
# cpu_columns = [f'cpu_core_{i+1}' for i in range(32)]
# X[cpu_columns] = pd.DataFrame(X['CPU Usage Per Core'].tolist(), columns=cpu_columns, index=X.index)
# X.drop(columns=['CPU Usage Per Core'], inplace=True)

# print("\nFeature columns after transformations:")
# print(X.columns)

# # --------------------------------------------------------------------------------------
# # 7. Dimensionality Reduction (PCA on CPU usage)
# # --------------------------------------------------------------------------------------
# scaler = StandardScaler()
# X_cpu_scaled = scaler.fit_transform(X[cpu_columns])

# pca_optimal = PCA(n_components=0.95, random_state=42)
# X_cpu_pca_optimal = pca_optimal.fit_transform(X_cpu_scaled)

# print(f"\nNumber of PCA components to retain 95% variance: {pca_optimal.n_components_}")

# # Create PCA component names
# pca_optimal_columns = [f'cpu_pca_optimal_{i+1}' for i in range(X_cpu_pca_optimal.shape[1])]
# cpu_pca_optimal_df = pd.DataFrame(X_cpu_pca_optimal, columns=pca_optimal_columns, index=X.index)

# # Concatenate the PCA components with the original DataFrame
# X = pd.concat([X, cpu_pca_optimal_df], axis=1)

# # Drop the original CPU usage columns
# X.drop(cpu_columns, axis=1, inplace=True)

# # (Optional) visualize PCA variance
# explained_variance = pca_optimal.explained_variance_ratio_
# cumulative_variance = explained_variance.cumsum()

# plt.figure(figsize=(10,6))
# plt.bar(range(1, len(explained_variance)+1), explained_variance, alpha=0.5,
#         align='center', label='Individual explained variance')
# plt.step(range(1, len(cumulative_variance)+1), cumulative_variance,
#          where='mid', label='Cumulative explained variance')
# plt.xlabel('Principal Components')
# plt.ylabel('Explained Variance Ratio')
# plt.legend(loc='best')
# plt.title('Explained Variance by Principal Components')
# plt.savefig(os.path.join(PLOTS_DIR, 'pca_explained_variance.png'))
# plt.show()

# # --------------------------------------------------------------------------------------
# # 8. Final Train-Test Split
# # --------------------------------------------------------------------------------------
# X_train, X_test, y_train, y_test = train_test_split(
#     X, y, test_size=0.2, random_state=42
# )
# print("\nTrain-Test Split:")
# print(f"Training Samples: {X_train.shape[0]}")
# print(f"Testing Samples: {X_test.shape[0]}")

# # --------------------------------------------------------------------------------------
# # 9. Hyperparameter Tuning (Addressing 3rd & 6th points)
# #    - 3rd point (Complexity): More layers (e.g., (200, 100, 50)), higher max_iter
# #    - 6th point (Regularization, Early Stopping): alpha includes near-zero, turn ES on/off
# # --------------------------------------------------------------------------------------
# pipeline = Pipeline([
#     ('scaler', StandardScaler()),
#     ('mlp', MLPRegressor(
#         activation='relu',
#         solver='adam',
#         random_state=42,
#         # The following defaults can be overridden by param_grid
#         # We'll keep them here for clarity
#         max_iter=200,
#         early_stopping=True,
#         n_iter_no_change=20,
#         validation_fraction=0.1
#     ))
# ])

# param_grid = {
#     # More complex hidden layer configurations
#     'mlp__hidden_layer_sizes': [
#         (50,),
#         (100,),
#         (100, 50),
#         (100, 100),
#         (100, 50, 25),
#         (200, 100, 50)   # deeper architecture to address underfitting
#     ],
#     # Include alpha=0.0 or very small alpha to reduce regularization
#     'mlp__alpha': [0.0, 1e-5, 1e-4, 1e-3, 1e-2],
#     # Test different maximum iterations to allow more training
#     'mlp__max_iter': [200, 500],
#     # Option to turn early stopping off to see if it over-regularizes
#     'mlp__early_stopping': [False, True]
# }

# from sklearn.model_selection import GridSearchCV

# grid_search = GridSearchCV(
#     estimator=pipeline,
#     param_grid=param_grid,
#     scoring='neg_mean_squared_error',
#     n_jobs=-1,
#     cv=3,
#     verbose=2
# )

# print("\nStarting Grid Search for Hyperparameter Tuning...")
# grid_search.fit(X_train, y_train)
# print("\nGrid Search Complete.")

# print(f"Best Parameters: {grid_search.best_params_}")
# print(f"Best CV Score (Negative MSE): {grid_search.best_score_}")

# # Retrieve best estimator
# best_model = grid_search.best_estimator_

# # --------------------------------------------------------------------------------------
# # 10. Final Evaluation on Test Set
# # --------------------------------------------------------------------------------------
# y_pred = best_model.predict(X_test)
# mse = mean_squared_error(y_test, y_pred)
# r2 = r2_score(y_test, y_pred)

# print("\nEvaluation on Test Set using the Best Model:")
# print(f"Mean Squared Error (MSE): {mse:.4f}")
# print(f"R² Score: {r2:.4f}")

# # --------------------------------------------------------------------------------------
# # 11. Save Model Metrics & Plots
# # --------------------------------------------------------------------------------------
# # Create a DataFrame with best hyperparameters and results
# metrics_df = pd.DataFrame({
#     'hidden_layer_sizes': [grid_search.best_params_['mlp__hidden_layer_sizes']],
#     'alpha': [grid_search.best_params_['mlp__alpha']],
#     'max_iter': [grid_search.best_params_['mlp__max_iter']],
#     'early_stopping': [grid_search.best_params_['mlp__early_stopping']],
#     'Best_CV_Score_(Neg_MSE)': [grid_search.best_score_],
#     'Test_MSE': [mse],
#     'Test_R2': [r2]
# })

# # Append or write new CSV
# if os.path.isfile(METRICS_CSV):
#     metrics_df.to_csv(METRICS_CSV, mode='a', header=False, index=False)
# else:
#     metrics_df.to_csv(METRICS_CSV, index=False)

# print(f"\nMetrics saved to: {METRICS_CSV}")

# # Plot Actual vs Predicted
# plt.figure(figsize=(8,6))
# sns.scatterplot(x=y_test, y=y_pred, alpha=0.5)
# plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
# plt.xlabel('Actual Processing Time')
# plt.ylabel('Predicted Processing Time')
# plt.title('Actual vs. Predicted Processing Time (Best Model)')
# plot_path = os.path.join(PLOTS_DIR, 'actual_vs_predicted.png')
# plt.savefig(plot_path)
# plt.show()

# # If you want to see the loss curve of the *best* MLP, access it from the pipeline:
# if hasattr(best_model.named_steps['mlp'], 'loss_curve_'):
#     plt.figure(figsize=(8,6))
#     plt.plot(best_model.named_steps['mlp'].loss_curve_)
#     plt.xlabel('Iterations')
#     plt.ylabel('Loss')
#     plt.title('Training Loss Curve (Best Model)')
#     loss_plot_path = os.path.join(PLOTS_DIR, 'training_loss_curve.png')
#     plt.savefig(loss_plot_path)
#     plt.show()

# print("\nAll plots saved in 'mlp_results/plots' and metrics in CSV file.")



#####################################################################################################################################################################
                        ############################  ABOVE CODE WORKING FINE BUT NOW BELOW THS CREATE  A SUBFOLDER AT EVERY RUN  #############################
##################################################################################################################################################################



# -*- coding: utf-8 -*-
"""
RegressorDetect_with_CustomRunFolders.ipynb

This script demonstrates:
1) More complex NN architectures (Point #3: Complexity / Underfitting)
2) Adjusting alpha & early stopping (Point #6: Regularization or early stopping)
3) Creating a new run folder each time by checking which run number is missing.
4) Storing all plots in that run folder without showing them inline.
"""

import os
import re
import ast
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline

# --------------------------------------------------------------------------------------
# 1. Find the next available run folder
# --------------------------------------------------------------------------------------
BASE_RESULTS_DIR = "mlp_results"
if not os.path.exists(BASE_RESULTS_DIR):
    os.makedirs(BASE_RESULTS_DIR)

# Parse existing "run_X" folders to find the highest run number
existing_runs = []
for entry in os.listdir(BASE_RESULTS_DIR):
    if os.path.isdir(os.path.join(BASE_RESULTS_DIR, entry)) and entry.startswith("run_"):
        match = re.match(r"run_(\d+)$", entry)
        if match:
            existing_runs.append(int(match.group(1)))

if existing_runs:
    next_run_number = max(existing_runs) + 1
else:
    next_run_number = 1

RUN_FOLDER_NAME = f"run_{next_run_number}"
RESULTS_DIR = os.path.join(BASE_RESULTS_DIR, RUN_FOLDER_NAME)
os.makedirs(RESULTS_DIR, exist_ok=True)

PLOTS_DIR = os.path.join(RESULTS_DIR, "plots")
os.makedirs(PLOTS_DIR, exist_ok=True)

METRICS_CSV = os.path.join(RESULTS_DIR, "model_metrics.csv")

# --------------------------------------------------------------------------------------
# 2. Load Data
# --------------------------------------------------------------------------------------
df = pd.read_csv('/home/user/Desktop/CODE/Individual_Regressor/predictRegressor/updated_file_Predict.csv')
print("Data Loaded. Head of the DataFrame:")
print(df.head())

# --------------------------------------------------------------------------------------
# 3. Basic EDA and filtering
# --------------------------------------------------------------------------------------
print("\nDataFrame Columns:")
print(df.columns)

plt.figure(figsize=(10, 6))
sns.histplot(df['Processing Time'], kde=True)
plt.title('Distribution of Processing Time')
plt.xlabel('Processing Time')
plt.ylabel('Frequency')
plt.savefig(os.path.join(PLOTS_DIR, 'processing_time_distribution_initial.png'))
plt.close()  # Do NOT show inline

distribution_counts = df['Processing Time'].value_counts().sort_index()
print("Distribution Counts:")
print(distribution_counts)

# Filter out Processing Time > 20
df = df[df['Processing Time'] <= 20]

plt.figure(figsize=(10, 6))
sns.histplot(df['Processing Time'], kde=True)
plt.title('Distribution of Processing Time (Filtered <= 20)')
plt.xlabel('Processing Time')
plt.ylabel('Frequency')
plt.savefig(os.path.join(PLOTS_DIR, 'processing_time_distribution_filtered.png'))
plt.close()  # Do NOT show inline

distribution_counts = df['Processing Time'].value_counts().sort_index()
print("Distribution Counts after filtering:")
print(distribution_counts)

# --------------------------------------------------------------------------------------
# 4. Drop unwanted columns
# --------------------------------------------------------------------------------------
columns_to_drop = [
    'Image No',
    'Iteration',
    'Execution Time (seconds)',
    'File',
    'Duration'
]
df.drop(columns=columns_to_drop, inplace=True, errors='ignore')

# --------------------------------------------------------------------------------------
# 5. Separate features (X) and target (y)
# --------------------------------------------------------------------------------------
y = df['Processing Time']
X = df.drop(columns=['Processing Time'])

print("\nFeatures (X) head:")
print(X.head())
print("\nTarget (y) head:")
print(y.head())

# --------------------------------------------------------------------------------------
# 6. Data Cleaning / Feature Engineering
# --------------------------------------------------------------------------------------
print("\nData Types of X:")
print(X.dtypes)

# Split 'Image Pixel' into width and height, then create image_size
X[['width', 'height']] = X['Image Pixel'].str.split('x', expand=True).astype(int)
X['image_size'] = X['width'] * X['height']
X.drop(columns=['Image Pixel'], inplace=True)

# Remove columns if not needed
X.drop(columns=['Current Script','Total CPU Cores'], inplace=True, errors='ignore')

# One-hot encoding of categorical columns
categorical_data = ['CPU Model','GPU Model']
encoder = pd.get_dummies(X[categorical_data], drop_first=False)
X = pd.concat([X, encoder], axis=1)
X.drop(columns=categorical_data, inplace=True)

# Count occurrences of p, s, d in 'Combination'
X['num_p'] = X['Combination'].str.count('p')
X['num_s'] = X['Combination'].str.count('s')
X['num_d'] = X['Combination'].str.count('d')
X.drop(columns=['Combination'], inplace=True)

# Convert CPU Usage Per Core from string to list
X['CPU Usage Per Core'] = X['CPU Usage Per Core'].apply(
    lambda x: ast.literal_eval(x) if isinstance(x, str) else x
)

# Expand CPU usage into separate columns
cpu_columns = [f'cpu_core_{i+1}' for i in range(32)]
X[cpu_columns] = pd.DataFrame(X['CPU Usage Per Core'].tolist(), columns=cpu_columns, index=X.index)
X.drop(columns=['CPU Usage Per Core'], inplace=True)

print("\nFeature columns after transformations:")
print(X.columns)

# --------------------------------------------------------------------------------------
# 7. Dimensionality Reduction (PCA on CPU usage)
# --------------------------------------------------------------------------------------
scaler = StandardScaler()
X_cpu_scaled = scaler.fit_transform(X[cpu_columns])

pca_optimal = PCA(n_components=0.95, random_state=42)
X_cpu_pca_optimal = pca_optimal.fit_transform(X_cpu_scaled)

print(f"\nNumber of PCA components to retain 95% variance: {pca_optimal.n_components_}")

pca_optimal_columns = [f'cpu_pca_optimal_{i+1}' for i in range(X_cpu_pca_optimal.shape[1])]
cpu_pca_optimal_df = pd.DataFrame(X_cpu_pca_optimal, columns=pca_optimal_columns, index=X.index)

# Concatenate
X = pd.concat([X, cpu_pca_optimal_df], axis=1)
# Drop original CPU usage columns
X.drop(cpu_columns, axis=1, inplace=True)

# (Optional) visualize PCA variance
explained_variance = pca_optimal.explained_variance_ratio_
cumulative_variance = explained_variance.cumsum()

plt.figure(figsize=(10,6))
plt.bar(range(1, len(explained_variance)+1), explained_variance, alpha=0.5,
        align='center', label='Individual explained variance')
plt.step(range(1, len(cumulative_variance)+1), cumulative_variance,
         where='mid', label='Cumulative explained variance')
plt.xlabel('Principal Components')
plt.ylabel('Explained Variance Ratio')
plt.legend(loc='best')
plt.title('Explained Variance by Principal Components')
plt.savefig(os.path.join(PLOTS_DIR, 'pca_explained_variance.png'))
plt.close()  # Do NOT show inline

# --------------------------------------------------------------------------------------
# 8. Final Train-Test Split
# --------------------------------------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
print("\nTrain-Test Split:")
print(f"Training Samples: {X_train.shape[0]}")
print(f"Testing Samples: {X_test.shape[0]}")

# --------------------------------------------------------------------------------------
# 9. Hyperparameter Tuning (Points #3 and #6)
# --------------------------------------------------------------------------------------
from sklearn.pipeline import Pipeline

pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('mlp', MLPRegressor(
        activation='relu',
        solver='adam',
        random_state=42,
        max_iter=200,
        early_stopping=True,
        n_iter_no_change=20,
        validation_fraction=0.1
    ))
])

param_grid = {
    'mlp__hidden_layer_sizes': [
        (50,),
        (100,),
        (100, 50),
        (100, 100),
        (100, 50, 25),
        (200, 100, 50)
    ],
    'mlp__alpha': [0.0, 1e-5, 1e-4, 1e-3, 1e-2],
    'mlp__max_iter': [200, 500],
    'mlp__early_stopping': [False, True]
}

from sklearn.model_selection import GridSearchCV

grid_search = GridSearchCV(
    estimator=pipeline,
    param_grid=param_grid,
    scoring='neg_mean_squared_error',
    n_jobs=-1,
    cv=3,
    verbose=2
)

print("\nStarting Grid Search for Hyperparameter Tuning...")
grid_search.fit(X_train, y_train)
print("\nGrid Search Complete.")

print(f"Best Parameters: {grid_search.best_params_}")
print(f"Best CV Score (Negative MSE): {grid_search.best_score_}")

# Retrieve best estimator
best_model = grid_search.best_estimator_

# --------------------------------------------------------------------------------------
# 10. Final Evaluation on Test Set
# --------------------------------------------------------------------------------------
y_pred = best_model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("\nEvaluation on Test Set using the Best Model:")
print(f"Mean Squared Error (MSE): {mse:.4f}")
print(f"R² Score: {r2:.4f}")

# --------------------------------------------------------------------------------------
# 11. Save Model Metrics & Plots
# --------------------------------------------------------------------------------------
metrics_df = pd.DataFrame({
    'hidden_layer_sizes': [grid_search.best_params_['mlp__hidden_layer_sizes']],
    'alpha': [grid_search.best_params_['mlp__alpha']],
    'max_iter': [grid_search.best_params_['mlp__max_iter']],
    'early_stopping': [grid_search.best_params_['mlp__early_stopping']],
    'Best_CV_Score_(Neg_MSE)': [grid_search.best_score_],
    'Test_MSE': [mse],
    'Test_R2': [r2]
})

# Write CSV in the newly created run folder (no appending, each run has its own CSV)
metrics_df.to_csv(METRICS_CSV, index=False)

print(f"\nMetrics saved to: {METRICS_CSV}")

# Actual vs. Predicted Plot
plt.figure(figsize=(8,6))
sns.scatterplot(x=y_test, y=y_pred, alpha=0.5)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
plt.xlabel('Actual Processing Time')
plt.ylabel('Predicted Processing Time')
plt.title('Actual vs. Predicted Processing Time (Best Model)')
plot_path = os.path.join(PLOTS_DIR, 'actual_vs_predicted.png')
plt.savefig(plot_path)
plt.close()  # Do NOT show inline

# Training Loss Curve (if available)
if hasattr(best_model.named_steps['mlp'], 'loss_curve_'):
    plt.figure(figsize=(8,6))
    plt.plot(best_model.named_steps['mlp'].loss_curve_)
    plt.xlabel('Iterations')
    plt.ylabel('Loss')
    plt.title('Training Loss Curve (Best Model)')
    loss_plot_path = os.path.join(PLOTS_DIR, 'training_loss_curve.png')
    plt.savefig(loss_plot_path)
    plt.close()  # Do NOT show inline

print(f"\nAll plots saved in: {PLOTS_DIR}")
print(f"Run #{next_run_number} complete. Check folder '{RUN_FOLDER_NAME}' for outputs.")