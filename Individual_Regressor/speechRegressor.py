#!/usr/bin/python3
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt
import os
import joblib

# Constants
RESULTS_DIR = 'SPEECH_RESULTS'
DATA_FILE = '/home/user/Desktop/CODE/Individual_Regressor/filtered_file_speech.csv'
TARGET_COLUMN = 'Processing Time'
COLUMNS_TO_DROP = [
    'Image No', 'Image Pixel', 'Execution Time (seconds)',
    'CPU Usage Per Core', 'File'
]
CATEGORICAL_COLUMNS = ['Current Script', 'CPU Model', 'GPU Model', 'Combination']
RANDOM_STATE = 42

# Create results directory
os.makedirs(RESULTS_DIR, exist_ok=True)

def load_data(file_path):
    """Load the dataset from the specified CSV file."""
    try:
        df = pd.read_csv(file_path)
        print(f"Data loaded successfully. Shape: {df.shape}")
        return df
    except FileNotFoundError:
        print(f"Error: File '{file_path}' not found.")
        exit(1)
    except Exception as e:
        print(f"Error loading data: {str(e)}")
        exit(1)

# def preprocess_data(df):
#     """
#     Preprocess the input DataFrame by encoding categorical features 
#     and separating features and target.
#     """
#     try:
#         # Drop unnecessary columns
#         df = df.drop(columns=COLUMNS_TO_DROP, errors='ignore')
        
#         # Handle categorical variables with Label Encoding
#         label_encoders = {}
#         for col in ['Current Script', 'CPU Model', 'GPU Model']:
#             if col in df.columns:
#                 le = LabelEncoder()
#                 df[col] = le.fit_transform(df[col].astype(str))
#                 label_encoders[col] = le
        
#         # One-hot encode the Combination column
#         if 'Combination' in df.columns:
#             combination_dummies = pd.get_dummies(
#                 df['Combination'].astype(str), 
#                 prefix='Combination',
#                 drop_first=True  # Drop first column to avoid multicollinearity
#             )
#             df = pd.concat([df.drop('Combination', axis=1), combination_dummies], axis=1)
        
#         # Split features and target
#         X = df.drop(TARGET_COLUMN, axis=1)
#         y = df[TARGET_COLUMN]
        
#         print("Preprocessing completed successfully")
#         print(f"Features shape: {X.shape}")
#         print("Features included:", ', '.join(X.columns))
        
#         return X, y, label_encoders
    
#     except Exception as e:
#         print(f"Error in preprocessing: {str(e)}")
#         raise
from sklearn.impute import SimpleImputer

def preprocess_data(df):
    """
    Preprocess the input DataFrame by encoding categorical features 
    and separating features and target, and handling missing values.
    """
    try:
        # Drop unnecessary columns
        df = df.drop(columns=COLUMNS_TO_DROP, errors='ignore')
        
        # Handle categorical variables with Label Encoding
        label_encoders = {}
        for col in ['Current Script', 'CPU Model', 'GPU Model']:
            if col in df.columns:
                le = LabelEncoder()
                df[col] = le.fit_transform(df[col].astype(str))
                label_encoders[col] = le
        
        # One-hot encode the Combination column
        if 'Combination' in df.columns:
            combination_dummies = pd.get_dummies(
                df['Combination'].astype(str), 
                prefix='Combination',
                drop_first=True  # Drop first column to avoid multicollinearity
            )
            df = pd.concat([df.drop('Combination', axis=1), combination_dummies], axis=1)
        
        # Handle missing values in numeric columns using mean imputation
        imputer = SimpleImputer(strategy='mean')
        df_imputed = pd.DataFrame(imputer.fit_transform(df), columns=df.columns)
        
        # Split features and target
        X = df_imputed.drop(TARGET_COLUMN, axis=1)
        y = df_imputed[TARGET_COLUMN]
        
        print("Preprocessing completed successfully")
        print(f"Features shape: {X.shape}")
        print("Features included:", ', '.join(X.columns))
        
        return X, y, label_encoders
    
    except Exception as e:
        print(f"Error in preprocessing: {str(e)}")
        raise  # This stops further execution and re-raises the exception
    
def train_and_evaluate(X_train, X_test, y_train, y_test):
    """Train the model using GridSearchCV and evaluate its performance."""
    try:
        # Scale the features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # Define parameter grid for hyperparameter tuning
        param_grid = {
            'hidden_layer_sizes': [(50,), (100,), (50, 25), (100, 50), (100, 75, 50)],
            'activation': ['relu', 'tanh'],
            'learning_rate_init': [0.001, 0.01, 0.0001],
            'max_iter': [2000],  # Increased max iterations
            'alpha': [0.0001, 0.001, 0.01],
            'batch_size': ['auto', 32, 64]
        }

        # Create and train model with GridSearchCV
        mlp = MLPRegressor(random_state=RANDOM_STATE, early_stopping=True)
        grid_search = GridSearchCV(
            mlp, 
            param_grid, 
            cv=5, 
            scoring=['neg_mean_squared_error', 'r2'],
            refit='neg_mean_squared_error',
            n_jobs=-1,
            verbose=1
        )
        
        print("Starting model training...")
        grid_search.fit(X_train_scaled, y_train)
        print("Model training completed")

        # Get best model
        best_model = grid_search.best_estimator_

        # Make predictions
        y_train_pred = best_model.predict(X_train_scaled)
        y_test_pred = best_model.predict(X_test_scaled)

        # Calculate metrics
        metrics = {
            'Train MSE': mean_squared_error(y_train, y_train_pred),
            'Test MSE': mean_squared_error(y_test, y_test_pred),
            'Train RMSE': np.sqrt(mean_squared_error(y_train, y_train_pred)),
            'Test RMSE': np.sqrt(mean_squared_error(y_test, y_test_pred)),
            'Train R2': r2_score(y_train, y_train_pred),
            'Test R2': r2_score(y_test, y_test_pred),
            'Best Parameters': grid_search.best_params_
        }

        return best_model, scaler, y_test, y_test_pred, metrics

    except Exception as e:
        print(f"Error in model training: {str(e)}")
        raise

def save_results(y_test, y_test_pred, metrics, best_model, scaler):
    """Save predictions, model, scaler, metrics, and plots."""
    try:
        # Save predictions
        results = pd.DataFrame({
            'True_Processing_Time': y_test,
            'Predicted_Processing_Time': y_test_pred,
            'Absolute_Error': np.abs(y_test - y_test_pred)
        })
        results.to_csv(os.path.join(RESULTS_DIR, 'predictions.csv'), index=False)

        # Save model and scaler
        joblib.dump(best_model, os.path.join(RESULTS_DIR, 'best_model.joblib'))
        joblib.dump(scaler, os.path.join(RESULTS_DIR, 'scaler.joblib'))

        # Save metrics
        with open(os.path.join(RESULTS_DIR, 'metrics.txt'), 'w') as f:
            for metric_name, value in metrics.items():
                f.write(f'{metric_name}: {value}\n')

        # Create and save visualization
        plt.figure(figsize=(12, 8))
        plt.scatter(y_test, y_test_pred, alpha=0.5, label='Predictions')
        plt.plot([y_test.min(), y_test.max()], 
                [y_test.min(), y_test.max()], 
                'r--', 
                lw=2, 
                label='Perfect Prediction')
        plt.xlabel('True Processing Time')
        plt.ylabel('Predicted Processing Time')
        plt.title('True vs Predicted Processing Time')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(RESULTS_DIR, 'prediction_plot.png'), dpi=300)
        plt.close()

        print("Results saved successfully in", RESULTS_DIR)

    except Exception as e:
        print(f"Error saving results: {str(e)}")
        raise

def main():
    try:
        print("Starting speech processing time prediction pipeline...")
        
        # Load and preprocess data
        df = load_data(DATA_FILE)
        X, y, _ = preprocess_data(df)

        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=RANDOM_STATE
        )

        # Train and evaluate model
        best_model, scaler, y_test, y_test_pred, metrics = train_and_evaluate(
            X_train, X_test, y_train, y_test
        )

        # Save results
        save_results(y_test, y_test_pred, metrics, best_model, scaler)

        # Print summary
        print("\n=== Training Summary ===")
        print(f"Number of features: {X.shape[1]}")
        print(f"Training samples: {X_train.shape[0]}")
        print(f"Test samples: {X_test.shape[0]}")
        print("\nBest Parameters:", metrics['Best Parameters'])
        print(f"\nModel Performance:")
        print(f"Train RMSE: {metrics['Train RMSE']:.4f}")
        print(f"Test RMSE: {metrics['Test RMSE']:.4f}")
        print(f"Train R2: {metrics['Train R2']:.4f}")
        print(f"Test R2: {metrics['Test R2']:.4f}")

    except Exception as e:
        print(f"Error in main execution: {str(e)}")
        raise

if __name__ == "__main__":
    main()