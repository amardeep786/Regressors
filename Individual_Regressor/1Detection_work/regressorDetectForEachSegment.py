# import os
# import re
# import ast
# import joblib
# import pandas as pd
# import numpy as np
# import seaborn as sns
# import matplotlib.pyplot as plt

# from sklearn.model_selection import train_test_split
# from sklearn.neural_network import MLPRegressor
# from sklearn.metrics import mean_squared_error, r2_score
# from sklearn.preprocessing import StandardScaler
# from sklearn.decomposition import PCA
# from sklearn.pipeline import Pipeline

# def main():
#     """
#     This script:
#       1. Loads the dataset and replicates the same data cleaning & feature engineering steps.
#       2. Splits into train/test.
#       3. Constructs the best model (without re-searching).
#       4. Fits the best model (or loads if you have a saved model).
#       5. Segregates the test data in [0-5), [5-10), ... bins and computes MSE, R² for each.
#       6. Saves each bin's results to individual CSV files.
#     """

#     # --------------------------------------------------------------------------------------
#     # 1. Load Data
#     # --------------------------------------------------------------------------------------
#     df = pd.read_csv('/home/user/Desktop/CODE/Individual_Regressor/07_11_2025_predict_work/updated_file_Predict.csv')
#     print("Data Loaded. Head of the DataFrame:")
#     print(df.head())

#     # --------------------------------------------------------------------------------------
#     # 2. Basic EDA & Filtering (optional: same as your code)
#     # --------------------------------------------------------------------------------------
#     # Filter out Processing Time > 20 (Adjust or remove if your final model was trained with a different cutoff)
#     df = df[df['Processing Time'] <= 20]

#     # --------------------------------------------------------------------------------------
#     # 3. Drop unwanted columns
#     # --------------------------------------------------------------------------------------
#     columns_to_drop = [
#         'Image No',
#         'Iteration',
#         'Execution Time (seconds)',
#         'File',
#         'Duration'
#     ]
#     df.drop(columns=columns_to_drop, inplace=True, errors='ignore')

#     # --------------------------------------------------------------------------------------
#     # 4. Separate features (X) and target (y)
#     # --------------------------------------------------------------------------------------
#     y = df['Processing Time']
#     X = df.drop(columns=['Processing Time'])

#     # --------------------------------------------------------------------------------------
#     # 5. Data Cleaning / Feature Engineering
#     # --------------------------------------------------------------------------------------
#     # Split 'Image Pixel' into width, height => image_size
#     X[['width', 'height']] = X['Image Pixel'].str.split('x', expand=True).astype(int)
#     X['image_size'] = X['width'] * X['height']
#     X.drop(columns=['Image Pixel'], inplace=True)

#     # Remove columns if not needed
#     X.drop(columns=['Current Script','Total CPU Cores'], inplace=True, errors='ignore')

#     # One-hot encoding of categorical columns
#     categorical_data = ['CPU Model', 'GPU Model']
#     encoder = pd.get_dummies(X[categorical_data], drop_first=False)
#     X = pd.concat([X, encoder], axis=1)
#     X.drop(columns=categorical_data, inplace=True)

#     # Count occurrences of p, s, d in 'Combination'
#     X['num_p'] = X['Combination'].str.count('p')
#     X['num_s'] = X['Combination'].str.count('s')
#     X['num_d'] = X['Combination'].str.count('d')
#     X.drop(columns=['Combination'], inplace=True)

#     # Convert CPU Usage Per Core from string to list
#     X['CPU Usage Per Core'] = X['CPU Usage Per Core'].apply(
#         lambda x: ast.literal_eval(x) if isinstance(x, str) else x
#     )

#     # Expand CPU usage into separate columns
#     cpu_columns = [f'cpu_core_{i+1}' for i in range(32)]
#     X[cpu_columns] = pd.DataFrame(X['CPU Usage Per Core'].tolist(), columns=cpu_columns, index=X.index)
#     X.drop(columns=['CPU Usage Per Core'], inplace=True)

#     # --------------------------------------------------------------------------------------
#     # 6. Dimensionality Reduction (PCA on CPU usage) - same as original
#     # --------------------------------------------------------------------------------------
#     scaler = StandardScaler()
#     X_cpu_scaled = scaler.fit_transform(X[cpu_columns])

#     pca_optimal = PCA(n_components=0.95, random_state=42)
#     X_cpu_pca_optimal = pca_optimal.fit_transform(X_cpu_scaled)

#     pca_optimal_columns = [f'cpu_pca_optimal_{i+1}' for i in range(X_cpu_pca_optimal.shape[1])]
#     cpu_pca_optimal_df = pd.DataFrame(X_cpu_pca_optimal, columns=pca_optimal_columns, index=X.index)

#     # Concatenate
#     X = pd.concat([X, cpu_pca_optimal_df], axis=1)
#     # Drop original CPU usage columns
#     X.drop(cpu_columns, axis=1, inplace=True)

#     # --------------------------------------------------------------------------------------
#     # 7. Final Train-Test Split
#     # --------------------------------------------------------------------------------------
#     X_train, X_test, y_train, y_test = train_test_split(
#         X, y, test_size=0.2, random_state=42
#     )
#     print("\nTrain-Test Split:")
#     print(f"Training Samples: {X_train.shape[0]}")
#     print(f"Testing Samples: {X_test.shape[0]}")

#     # --------------------------------------------------------------------------------------
#     # 8. Construct the Best Model *Without* GridSearch (use best hyperparams)
#     #    If you already saved a trained model, you can skip the .fit(...) and just load it.
#     # --------------------------------------------------------------------------------------
#     best_model_pipeline = Pipeline([
#         ('scaler', StandardScaler()),
#         ('mlp', MLPRegressor(
#             hidden_layer_sizes=(100,),
#             alpha=0.01,
#             max_iter=200,
#             early_stopping=False,
#             solver='adam',
#             random_state=42
#         ))
#     ])

#     # NOTE: If you have a saved model from before, do something like:
#     # best_model_pipeline = joblib.load("best_mlp_pipeline.joblib")
#     # and remove the fit step below.

#     print("\nFitting the pipeline with known best hyperparameters (this does not re-run GridSearchCV).")
#     best_model_pipeline.fit(X_train, y_train)
#     print("Done fitting the best model.\n")

#     # --------------------------------------------------------------------------------------
#     # 9. Evaluate on *segregated* Test Data Bins
#     #    E.g., [0-5), [5-10), [10-15), [15-20), ... up to [40-45)
#     # --------------------------------------------------------------------------------------
#     # If your data truly only goes up to 20ms, you might not need bins beyond 20ms.
#     # But we'll illustrate the general idea up to 45ms.
#     bins = [
#         (0, 5),
#         (5, 10),
#         (10, 15),
#         (15, 20),
#         (20, 25),
#         (25, 30),
#         (30, 35),
#         (35, 40),
#         (40, 45)
#     ]

#     # Make an output directory for bin results if desired
#     SEGREGATED_RESULTS_DIR = "segregated_test_results"
#     os.makedirs(SEGREGATED_RESULTS_DIR, exist_ok=True)

#     for (lower_bound, upper_bound) in bins:
#         # Create a mask for the test set where y_test is within [lower_bound, upper_bound)
#         mask = (y_test >= lower_bound) & (y_test < upper_bound)
#         X_test_bin = X_test[mask]
#         y_test_bin = y_test[mask]

#         # If there's no data in this bin, skip
#         if len(X_test_bin) == 0:
#             print(f"No test samples in range [{lower_bound},{upper_bound}). Skipping.")
#             continue

#         # Predict
#         y_pred_bin = best_model_pipeline.predict(X_test_bin)

#         # Compute metrics
#         mse_bin = mean_squared_error(y_test_bin, y_pred_bin)
#         r2_bin = r2_score(y_test_bin, y_pred_bin)

#         print(f"Range [{lower_bound},{upper_bound}) => #Samples: {len(X_test_bin)}")
#         print(f"  MSE: {mse_bin:.4f}")
#         print(f"  R² : {r2_bin:.4f}")

#         # Save each bin result to a CSV
#         bin_results_df = pd.DataFrame({
#             'Range': [f"{lower_bound}_{upper_bound}"],
#             'Num_Samples': [len(X_test_bin)],
#             'MSE': [mse_bin],
#             'R2': [r2_bin]
#         })
#         out_csv_path = os.path.join(SEGREGATED_RESULTS_DIR, f"bin_{lower_bound}_{upper_bound}_results.csv")
#         bin_results_df.to_csv(out_csv_path, index=False)
#         print(f"  -> Results saved to {out_csv_path}\n")

# if __name__ == "__main__":
#     main()

# -----------------------------------------------------------------------------------------------------------------------------------------------------------
##   saves plot tooo
# -------------------------------------------------------------------------------------------------------------------------------------------------------------


# import os
# import ast
# import joblib
# import pandas as pd
# import numpy as np
# import seaborn as sns
# import matplotlib.pyplot as plt

# from sklearn.model_selection import train_test_split
# from sklearn.neural_network import MLPRegressor
# from sklearn.metrics import mean_squared_error, r2_score
# from sklearn.preprocessing import StandardScaler
# from sklearn.decomposition import PCA
# from sklearn.pipeline import Pipeline

# def main():
#     """
#     This script:
#       1. Loads the dataset and replicates the same data cleaning & feature engineering steps.
#       2. Splits into train/test.
#       3. Constructs the best model (without re-searching).
#       4. Fits the best model (or loads if you have a saved model).
#       5. Segregates the test data in [0-5), [5-10), [10-15), etc. bins and computes MSE, R² for each.
#       6. Saves each bin’s results in a CSV, plus an 'Actual vs. Predicted' scatter plot for each bin.
#     """

#     import re  # if needed, though not strictly used here

#     # --------------------------------------------------------------------------------------
#     # 1. Load Data
#     # --------------------------------------------------------------------------------------
#     df = pd.read_csv('/home/user/Desktop/CODE/Individual_Regressor/predictRegressor/updated_file_Predict.csv')
#     print("Data Loaded. Head of the DataFrame:")
#     print(df.head())

#     # --------------------------------------------------------------------------------------
#     # 2. Basic EDA & Filtering (optional: same as your code)
#     # --------------------------------------------------------------------------------------
#     # Filter out Processing Time > 20 (Adjust or remove if your final model was trained with a different cutoff)
#     df = df[df['Processing Time'] <= 20]

#     # --------------------------------------------------------------------------------------
#     # 3. Drop unwanted columns
#     # --------------------------------------------------------------------------------------
#     columns_to_drop = [
#         'Image No',
#         'Iteration',
#         'Execution Time (seconds)',
#         'File',
#         'Duration'
#     ]
#     df.drop(columns=columns_to_drop, inplace=True, errors='ignore')

#     # --------------------------------------------------------------------------------------
#     # 4. Separate features (X) and target (y)
#     # --------------------------------------------------------------------------------------
#     y = df['Processing Time']
#     X = df.drop(columns=['Processing Time'])

#     # --------------------------------------------------------------------------------------
#     # 5. Data Cleaning / Feature Engineering
#     # --------------------------------------------------------------------------------------
#     # Split 'Image Pixel' into width, height => image_size
#     X[['width', 'height']] = X['Image Pixel'].str.split('x', expand=True).astype(int)
#     X['image_size'] = X['width'] * X['height']
#     X.drop(columns=['Image Pixel'], inplace=True)

#     # Remove columns if not needed
#     X.drop(columns=['Current Script','Total CPU Cores'], inplace=True, errors='ignore')

#     # One-hot encoding of categorical columns
#     categorical_data = ['CPU Model', 'GPU Model']
#     encoder = pd.get_dummies(X[categorical_data], drop_first=False)
#     X = pd.concat([X, encoder], axis=1)
#     X.drop(columns=categorical_data, inplace=True)

#     # Count occurrences of p, s, d in 'Combination'
#     X['num_p'] = X['Combination'].str.count('p')
#     X['num_s'] = X['Combination'].str.count('s')
#     X['num_d'] = X['Combination'].str.count('d')
#     X.drop(columns=['Combination'], inplace=True)

#     # Convert CPU Usage Per Core from string to list
#     X['CPU Usage Per Core'] = X['CPU Usage Per Core'].apply(
#         lambda x: ast.literal_eval(x) if isinstance(x, str) else x
#     )

#     # Expand CPU usage into separate columns
#     cpu_columns = [f'cpu_core_{i+1}' for i in range(32)]
#     X[cpu_columns] = pd.DataFrame(X['CPU Usage Per Core'].tolist(), columns=cpu_columns, index=X.index)
#     X.drop(columns=['CPU Usage Per Core'], inplace=True)

#     # --------------------------------------------------------------------------------------
#     # 6. Dimensionality Reduction (PCA on CPU usage) - same as original
#     # --------------------------------------------------------------------------------------
#     # from sklearn.preprocessing import StandardScaler
#     # scaler = StandardScaler()
#     # X_cpu_scaled = scaler.fit_transform(X[cpu_columns])

#     # pca_optimal = PCA(n_components=0.95, random_state=42)
#     # X_cpu_pca_optimal = pca_optimal.fit_transform(X_cpu_scaled)

#     # pca_optimal_columns = [f'cpu_pca_optimal_{i+1}' for i in range(X_cpu_pca_optimal.shape[1])]
#     # cpu_pca_optimal_df = pd.DataFrame(X_cpu_pca_optimal, columns=pca_optimal_columns, index=X.index)

#     # # Concatenate
#     # X = pd.concat([X, cpu_pca_optimal_df], axis=1)
#     # # Drop original CPU usage columns
#     # X.drop(cpu_columns, axis=1, inplace=True)

#     # --------------------------------------------------------------------------------------
#     # 7. Final Train-Test Split
#     # --------------------------------------------------------------------------------------
#     X_train, X_test, y_train, y_test = train_test_split(
#         X, y, test_size=0.2, random_state=42
#     )
#     print("\nTrain-Test Split:")
#     print(f"Training Samples: {X_train.shape[0]}")
#     print(f"Testing Samples: {X_test.shape[0]}")

#     # --------------------------------------------------------------------------------------
#     # 8. Construct the Best Model *Without* GridSearch (use best hyperparams)
#     #    If you already saved a trained model, you can skip the .fit(...) and just load it.
#     # --------------------------------------------------------------------------------------
#     best_model_pipeline = Pipeline([
#         ('scaler', StandardScaler()),
#         ('mlp', MLPRegressor(
#             hidden_layer_sizes=(100,),
#             alpha=0.01,
#             max_iter=200,
#             early_stopping=False,
#             solver='adam',
#             random_state=42
#         ))
#     ])

#     # NOTE: If you have a saved model from before, do something like:
#     # best_model_pipeline = joblib.load("best_mlp_pipeline.joblib")
#     # and remove the fit step below.

#     print("\nFitting the pipeline with known best hyperparameters (this does not re-run GridSearchCV).")
#     best_model_pipeline.fit(X_train, y_train)
#     print("Done fitting the best model.\n")

#     # --------------------------------------------------------------------------------------
#     # 9. Evaluate on *segregated* Test Data Bins + Plot Actual vs Predicted
#     # --------------------------------------------------------------------------------------
#     # If your data truly only goes up to 20ms, no need for bins beyond 20ms.
#     # But we illustrate up to 20 here. Adjust as you see fit.
#     bins = [
#         (0, 5),
#         (5, 10),
#         (10, 15),
#         (15, 20),
#         # You could go beyond 20 if you had data, e.g. (20, 25), ...
#     ]

#     SEGREGATED_RESULTS_DIR = "segregated_test_results2_without_PCA"
#     os.makedirs(SEGREGATED_RESULTS_DIR, exist_ok=True)

#     for (lower_bound, upper_bound) in bins:
#         # Create a mask for the test set where y_test is within [lower_bound, upper_bound)
#         mask = (y_test >= lower_bound) & (y_test < upper_bound)
#         X_test_bin = X_test[mask]
#         y_test_bin = y_test[mask]

#         # If there's no data in this bin, skip
#         if len(X_test_bin) == 0:
#             print(f"No test samples in range [{lower_bound},{upper_bound}). Skipping.")
#             continue

#         # Predict
#         y_pred_bin = best_model_pipeline.predict(X_test_bin)

#         # Compute metrics
#         mse_bin = mean_squared_error(y_test_bin, y_pred_bin)
#         r2_bin = r2_score(y_test_bin, y_pred_bin)

#         print(f"Range [{lower_bound},{upper_bound}) => #Samples: {len(X_test_bin)}")
#         print(f"  MSE: {mse_bin:.4f}")
#         print(f"  R² : {r2_bin:.4f}")

#         # 9.1) Save each bin result to CSV
#         bin_results_df = pd.DataFrame({
#             'Range': [f"{lower_bound}_{upper_bound}"],
#             'Num_Samples': [len(X_test_bin)],
#             'MSE': [mse_bin],
#             'R2': [r2_bin]
#         })
#         out_csv_path = os.path.join(SEGREGATED_RESULTS_DIR, f"bin_{lower_bound}_{upper_bound}_results.csv")
#         bin_results_df.to_csv(out_csv_path, index=False)
#         print(f"  -> Results saved to {out_csv_path}")

#         # 9.2) Plot Actual vs. Predicted
#         plt.figure(figsize=(6, 5))
#         sns.scatterplot(x=y_test_bin, y=y_pred_bin, alpha=0.5)
#         plt.plot(
#             [y_test_bin.min(), y_test_bin.max()],
#             [y_test_bin.min(), y_test_bin.max()],
#             'r--',
#             label='Ideal: Actual=Predicted'
#         )
#         plt.xlabel('Actual Processing Time')
#         plt.ylabel('Predicted Processing Time')
#         plt.title(f'Actual vs. Predicted ({lower_bound}-{upper_bound} ms)')
#         plt.legend()
#         # Save the plot
#         plot_filename = f"actual_vs_predicted_{lower_bound}_{upper_bound}.png"
#         plot_path = os.path.join(SEGREGATED_RESULTS_DIR, plot_filename)
#         plt.savefig(plot_path, dpi=100, bbox_inches='tight')
#         plt.close()

#         print(f"  -> Plot saved to {plot_path}\n")

# if __name__ == "__main__":
#     main()


####    SAVES CSV TOO #################################################################################################


import os
import ast
import joblib
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline

def main():
    """
    This script:
      1. Loads the dataset & replicates data cleaning & feature engineering steps.
      2. Splits into train/test.
      3. Constructs (or loads) the best model (without re-running GridSearch).
      4. Fits the best model on the training set (unless already loaded a saved model).
      5. Segregates the test data in [0-5), [5-10), [10-15), ... bins.
      6. For each bin:
         - Computes MSE, R² for that subset
         - Saves metrics to a CSV
         - Saves an Actual-vs-Predicted scatter plot
         - Saves a CSV with Actual and Predicted values
    """

    # --------------------------------------------------------------------------------------
    # 1. Load Data
    # --------------------------------------------------------------------------------------
    df = pd.read_csv('/home/user/Desktop/CODE/Individual_Regressor/07_11_2025_detection_work/updated_file_Detect.csv')
    print("Data Loaded. Head of the DataFrame:")
    print(df.head())

    # --------------------------------------------------------------------------------------
    # 2. (Optional) Filter out rows with Processing Time > 20
    #    (Adjust based on how you trained your final model)
    # --------------------------------------------------------------------------------------
    df = df[df['Processing Time'] <= 20]

    # --------------------------------------------------------------------------------------
    # 3. Drop unwanted columns
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
    # 4. Separate features (X) and target (y)
    # --------------------------------------------------------------------------------------
    y = df['Processing Time']
    X = df.drop(columns=['Processing Time'])

    # --------------------------------------------------------------------------------------
    # 5. Data Cleaning / Feature Engineering
    # --------------------------------------------------------------------------------------
    # 5.1) Split 'Image Pixel' into width, height => image_size
    X[['width', 'height']] = X['Image Pixel'].str.split('x', expand=True).astype(int)
    X['image_size'] = X['width'] * X['height']
    X.drop(columns=['Image Pixel'], inplace=True)

    # 5.2) Remove columns if not needed
    X.drop(columns=['Current Script','Total CPU Cores'], inplace=True, errors='ignore')

    # 5.3) One-hot encoding of categorical columns
    categorical_data = ['CPU Model', 'GPU Model']
    encoder = pd.get_dummies(X[categorical_data], drop_first=False)
    X = pd.concat([X, encoder], axis=1)
    X.drop(columns=categorical_data, inplace=True)

    # 5.4) Count occurrences of p, s, d in 'Combination'
    X['num_p'] = X['Combination'].str.count('p')
    X['num_s'] = X['Combination'].str.count('s')
    X['num_d'] = X['Combination'].str.count('d')
    X.drop(columns=['Combination'], inplace=True)

    # 5.5) Convert CPU Usage Per Core from string to list
    X['CPU Usage Per Core'] = X['CPU Usage Per Core'].apply(
        lambda x: ast.literal_eval(x) if isinstance(x, str) else x
    )

    # 5.6) Expand CPU usage into separate columns
    cpu_columns = [f'cpu_core_{i+1}' for i in range(32)]
    X[cpu_columns] = pd.DataFrame(X['CPU Usage Per Core'].tolist(), columns=cpu_columns, index=X.index)
    X.drop(columns=['CPU Usage Per Core'], inplace=True)

    # --------------------------------------------------------------------------------------
    # 6. Dimensionality Reduction (PCA on CPU usage) - same as original
    # --------------------------------------------------------------------------------------
    scaler = StandardScaler()
    X_cpu_scaled = scaler.fit_transform(X[cpu_columns])

    pca_optimal = PCA(n_components=0.95, random_state=42)
    X_cpu_pca_optimal = pca_optimal.fit_transform(X_cpu_scaled)

    pca_optimal_columns = [f'cpu_pca_optimal_{i+1}' for i in range(X_cpu_pca_optimal.shape[1])]
    cpu_pca_optimal_df = pd.DataFrame(X_cpu_pca_optimal, columns=pca_optimal_columns, index=X.index)

    # Concatenate
    X = pd.concat([X, cpu_pca_optimal_df], axis=1)
    # Drop original CPU usage columns
    X.drop(cpu_columns, axis=1, inplace=True)

    # --------------------------------------------------------------------------------------
    # 7. Train-Test Split
    # --------------------------------------------------------------------------------------
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    print("\nTrain-Test Split:")
    print(f"Training Samples: {X_train.shape[0]}")
    print(f"Testing Samples: {X_test.shape[0]}")

    # --------------------------------------------------------------------------------------
    # 8. Construct (or load) the Best Model
    # --------------------------------------------------------------------------------------
    best_model_pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('mlp', MLPRegressor(
            hidden_layer_sizes=(100,50),
            alpha=0.01,
            max_iter=200,
            early_stopping=False,
            solver='adam',
            random_state=42
        ))
    ])

    # If you have a saved model, e.g.:
    # best_model_pipeline = joblib.load("best_mlp_pipeline.joblib")
    # then comment out the .fit(...) call below.

    print("\nFitting the pipeline with the known best hyperparameters.")
    best_model_pipeline.fit(X_train, y_train)
    print("Done fitting.\n")

    # --------------------------------------------------------------------------------------
    # 9. Evaluate on segregated Test Data Bins, save results + scatter plot + predicted CSV
    # --------------------------------------------------------------------------------------
    bins = [
        (0, 5),
        (5, 10),
        (10, 15),
        (15, 20),
        # Add more if needed
    ]

    SEGREGATED_RESULTS_DIR = "segregated_test_results4_detect"
    os.makedirs(SEGREGATED_RESULTS_DIR, exist_ok=True)

    for (lower_bound, upper_bound) in bins:
        # Filter the test data for this bin
        mask = (y_test >= lower_bound) & (y_test < upper_bound)
        X_test_bin = X_test[mask]
        y_test_bin = y_test[mask]

        if len(X_test_bin) == 0:
            print(f"No test samples in range [{lower_bound},{upper_bound}). Skipping.")
            continue

        # Predict
        y_pred_bin = best_model_pipeline.predict(X_test_bin)

        # Compute metrics
        mse_bin = mean_squared_error(y_test_bin, y_pred_bin)
        r2_bin = r2_score(y_test_bin, y_pred_bin)

        print(f"Range [{lower_bound},{upper_bound}) => #Samples: {len(X_test_bin)}")
        print(f"  MSE: {mse_bin:.4f}")
        print(f"  R² : {r2_bin:.4f}")

        # 9.1) Save bin-level metrics to CSV
        bin_results_df = pd.DataFrame({
            'Range': [f"{lower_bound}_{upper_bound}"],
            'Num_Samples': [len(X_test_bin)],
            'MSE': [mse_bin],
            'R2': [r2_bin]
        })
        out_csv_path = os.path.join(SEGREGATED_RESULTS_DIR, f"bin_{lower_bound}_{upper_bound}_results.csv")
        bin_results_df.to_csv(out_csv_path, index=False)
        print(f"  -> Metrics saved to {out_csv_path}")

        # 9.2) Save Actual vs Predicted scatter plot
        plt.figure(figsize=(6, 5))
        sns.scatterplot(x=y_test_bin, y=y_pred_bin, alpha=0.5)
        plt.plot(
            [y_test_bin.min(), y_test_bin.max()],
            [y_test_bin.min(), y_test_bin.max()],
            'r--',
            label='Ideal: Actual=Predicted'
        )
        plt.xlabel('Actual Processing Time(in ms)')
        plt.ylabel('Predicted Processing Time(in ms)')
        # plt.title(f'Actual vs Predicted ({lower_bound}-{upper_bound} ms)')
        plt.legend()
        plot_filename = f"actual_vs_predicted_{lower_bound}_{upper_bound}.png"
        plot_path = os.path.join(SEGREGATED_RESULTS_DIR, plot_filename)
        plt.savefig(plot_path, dpi=100, bbox_inches='tight')
        plt.close()
        print(f"  -> Scatter plot saved to {plot_path}")

        # 9.3) Save Actual & Predicted values to CSV
        #      So you can inspect them manually for each bin
        predictions_df = pd.DataFrame({
            'Index': y_test_bin.index,
            'Actual': y_test_bin.values,
            'Predicted': y_pred_bin
        })
        preds_csv_path = os.path.join(SEGREGATED_RESULTS_DIR, f"predictions_{lower_bound}_{upper_bound}.csv")
        predictions_df.to_csv(preds_csv_path, index=False)
        print(f"  -> Actual & Predicted saved to {preds_csv_path}\n")

if __name__ == "__main__":
    main()