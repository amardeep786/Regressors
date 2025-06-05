import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, KFold, ParameterGrid, cross_val_score
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt
import seaborn as sns
import joblib  # for saving the model

# ---------------------
# 1) LOAD AND CLEAN THE DATA
# ---------------------
df = pd.read_csv('/home/user/Desktop/CODE/Individual_Regressor/updated_file_Detect.csv')

# Drop columns we don't need
columns_to_drop = [
    'Image No','Iteration','Execution Time (seconds)',
    'CPU Usage Per Core','File','Duration'
]
df = df.drop(columns=columns_to_drop)

# Separate features and target
y = df['Processing Time']
X = df.drop(columns=['Processing Time'])

# Parse 'Image Pixel' if it exists (e.g. '460x680' -> width=460, height=680)
if 'Image Pixel' in X.columns:
    X[['width', 'height']] = X['Image Pixel'].str.split('x', expand=True).astype(int)
    X.drop(columns=['Image Pixel'], inplace=True)

# Drop "Current Script" if present
if 'Current Script' in X.columns:
    X.drop(columns=['Current Script'], inplace=True)

# Encode categorical columns if they exist
categorical_data = []
if 'CPU Model' in X.columns:
    categorical_data.append('CPU Model')
if 'GPU Model' in X.columns:
    categorical_data.append('GPU Model')

if len(categorical_data) > 0:
    encoder = pd.get_dummies(X[categorical_data])
    X = pd.concat([X, encoder], axis=1).drop(columns=categorical_data)

# Convert 'Combination' column to length of the string (if present and is string-based)
if 'Combination' in X.columns and X['Combination'].dtype == object:
    X['Combination'] = X['Combination'].str.len()

# Handle missing values (mean imputation)
imputer = SimpleImputer(strategy='mean')
X_imputed = imputer.fit_transform(X)

# ---------------------
# 2) TRAIN-TEST SPLIT
# ---------------------
X_train, X_test, y_train, y_test = train_test_split(
    X_imputed, y, test_size=0.2, random_state=42
)

# ---------------------
# 3) DEFINE THE PIPELINE
# ---------------------
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    # Use early stopping to speed up training on unpromising combos
    ('mlp', MLPRegressor(
        max_iter=500,
        random_state=42,
        early_stopping=True,
        validation_fraction=0.1
    ))
])

# ---------------------
# 4) SPECIFY PARAMETER GRID
# ---------------------
param_grid = {
    'mlp__hidden_layer_sizes': [(50, 50), (100, 50)],  # Adjust as needed
    'mlp__activation': ['relu', 'tanh'],
    'mlp__solver': ['adam'],         # or ['adam', 'sgd'] if you want both
    'mlp__learning_rate_init': [0.001, 0.01]
}

# ---------------------
# 5) SETUP PARTIAL-RESULT STORAGE
# ---------------------
partial_results_path = '/home/user/Desktop/CODE/Individual_Regressor/partial_results.csv'

# Create a fresh CSV with a header
with open(partial_results_path, 'w') as f:
    f.write("hidden_layer_sizes,activation,solver,learning_rate_init,mse_mean,mse_std\n")

# ---------------------
# 6) MANUAL SEARCH LOOP
# ---------------------
kf = KFold(n_splits=3, shuffle=True, random_state=42)  # 3-fold CV for speed
best_mse = float('inf')
best_params = None

for params in ParameterGrid(param_grid):
    # Update pipeline parameters
    pipeline.set_params(**params)

    # Evaluate via cross-validation
    cv_scores = cross_val_score(
        pipeline,
        X_train,
        y_train,
        scoring='neg_mean_squared_error',
        cv=kf,
        n_jobs=-1
    )
    mse_scores = -cv_scores  # because it's neg_mean_squared_error
    mse_mean = mse_scores.mean()
    mse_std = mse_scores.std()

    # Store partial results immediately
    with open(partial_results_path, 'a') as f:
        f.write(f"{params['mlp__hidden_layer_sizes']},"
                f"{params['mlp__activation']},"
                f"{params['mlp__solver']},"
                f"{params['mlp__learning_rate_init']},"
                f"{mse_mean},{mse_std}\n")

    # Track best so far
    if mse_mean < best_mse:
        best_mse = mse_mean
        best_params = params

print("Best hyperparameters (CV):", best_params)
print("Best CV MSE:", best_mse)

# ---------------------
# 7) RETRAIN ON BEST PARAMS, EVALUATE ON TEST
# ---------------------
pipeline.set_params(**best_params)
pipeline.fit(X_train, y_train)
y_pred = pipeline.predict(X_test)
mse_test = mean_squared_error(y_test, y_pred)
print("Final Test MSE:", mse_test)

# ---------------------
# 8) SAVE THE TRAINED MODEL
# ---------------------
model_path = '/home/user/Desktop/CODE/Individual_Regressor/final_model.joblib'
joblib.dump(pipeline, model_path)
print(f"Model saved to: {model_path}")

# ---------------------
# 9) CREATE & SAVE HEATMAP
# ---------------------
# Let's read the partial results CSV
results_df = pd.read_csv(partial_results_path)

# Convert hidden_layer_sizes to string for pivoting
results_df['hidden_layer_sizes'] = results_df['hidden_layer_sizes'].astype(str)

# Example pivot: hidden_layer_sizes vs learning_rate_init => MSE
pivoted = results_df.pivot_table(
    index='hidden_layer_sizes',
    columns='learning_rate_init',
    values='mse_mean'
)

plt.figure(figsize=(8, 6))
sns.heatmap(pivoted, annot=True, fmt=".3f", cmap='viridis')
plt.title("Mean MSE: Hidden Layers vs. Learning Rate")
plt.ylabel("hidden_layer_sizes")
plt.xlabel("learning_rate_init")
plt.tight_layout()

heatmap_path = '/home/user/Desktop/CODE/Individual_Regressor/partial_results_heatmap.png'
plt.savefig(heatmap_path, dpi=150)
plt.close()
print(f"Saved heatmap to: {heatmap_path}")

# ---------------------
# 10) PREDICTED VS. ACTUAL PLOT
# ---------------------
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred, alpha=0.5, label='Predicted vs Actual')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 
         'r--', label='Ideal Fit')
plt.xlabel("Actual Processing Time (in ms)")
plt.ylabel("Predicted Processing Time (in ms)")
# plt.title("Predicted vs. Actual on Test Set")
plt.legend()
plt.grid(True)

pred_vs_actual_path = '/home/user/Desktop/CODE/Individual_Regressor/predicted_vs_actual.png'
plt.savefig(pred_vs_actual_path, dpi=150)
plt.close()
print(f"Saved Predicted vs. Actual plot to: {pred_vs_actual_path}")

# ---------------------
# 11) SAVE FINAL RESULTS (MSE + Some Predictions)
# ---------------------
final_results_path = '/home/user/Desktop/CODE/Individual_Regressor/final_results.csv'

# Maybe store a small sample of predictions side-by-side with actual
comparison_df = pd.DataFrame({
    'Actual': y_test,
    'Predicted': y_pred
}).reset_index(drop=True)

# We'll store MSE at the top, then predictions below (this is one approach)
with open(final_results_path, 'w') as f:
    # Write a small header
    f.write(f"Final Test MSE: {mse_test}\n\n")
    # Write 5 sample predictions
    f.write("Sample Predictions (first 5 rows):\n")
comparison_df.head(5).to_csv(final_results_path, mode='a', index=False)

print(f"All done! Final MSE and predictions stored in: {final_results_path}")