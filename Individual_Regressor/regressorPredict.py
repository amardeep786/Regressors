import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error,r2_score


df = pd.read_csv('/home/user/Desktop/CODE/Individual_Regressor/filtered_file_predict.csv')
df.head()

print(df.columns)
columns_to_drop = ['Image No','Iteration','Execution Time (seconds)','CPU Usage Per Core','File','Duration']
df = df.drop(columns=columns_to_drop)

y = df['Processing Time']
X = df.drop(columns=['Processing Time'])
X.head()

# Assuming 'Image Pixel' column contains values like '460x680'
X[['width', 'height']] = X['Image Pixel'].str.split('x', expand=True).astype(int)
X.head()
X = X.drop(columns=['Image Pixel'])
X.head()

X = X.drop(columns=['Current Script'])
X.columns

#Now Do encodings of categorical data
categorical_data = ['CPU Model','GPU Model']  #do one hot encoding
encoder = pd.get_dummies(X[categorical_data])

# Concatenate the encoded features with the original DataFrame
X = pd.concat([X, encoder], axis=1)

# Drop the original categorical columns
X = X.drop(columns=categorical_data)
X.head()

X['Combination'] = X['Combination'].str.len()
X.head()

# Model parameters and pipeline
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import  GridSearchCV

# Define the pipeline steps
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('mlp', MLPRegressor(max_iter=500, random_state=42))  # Increased max_iter
])

# Define the parameter grid for GridSearchCV
param_grid = {
    'mlp__hidden_layer_sizes': [(100, 50), (50, 50, 50), (100, 100, 50)],  # Different hidden layer architectures
    'mlp__activation': ['relu', 'tanh'],  # Different activation functions
    'mlp__solver': ['adam', 'sgd'],  # Different solvers
    'mlp__learning_rate_init': [0.001, 0.01, 0.1],  # Different learning rates
}

# Create GridSearchCV object
grid_search = GridSearchCV(pipeline, param_grid, scoring='neg_mean_squared_error', cv=5, verbose=1)
## HANDLE MISSING VALUES
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(strategy='mean')  # Other strategies: 'median', 'most_frequent'

# Create GridSearchCV object
grid_search = GridSearchCV(pipeline, param_grid, scoring='neg_mean_squared_error', cv=5, verbose=1)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Fit the imputer to your training data and transform both training and testing data
X_train = imputer.fit_transform(X_train)
X_test = imputer.transform(X_test)

# Fit the GridSearchCV object to the training data
grid_search.fit(X_train, y_train)

# Get the best model and its parameters
best_model = grid_search.best_estimator_
best_params = grid_search.best_params_

# Make predictions on the testing data using the best model
y_pred = best_model.predict(X_test)


from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
mse = mean_squared_error(y_test, y_pred)
print(f"Best Parameters: {best_params}")
print(f"Mean Squared Error: {mse}")

output_file_path = '/home/user/Desktop/CODE/Individual_Regressor/predictColab1.txt'  # Replace with your desired path
with open(output_file_path, 'w') as f:
    f.write(f"Best Parameters: {best_params}\n")
    f.write(f"Mean Squared Error: {mse}\n")

# Generate true vs. predicted time graph
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred, alpha=0.5)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')  # Diagonal line
plt.xlabel("True Processing Time")
plt.ylabel("Predicted Processing Time")
plt.title("True vs. Predicted Processing Time")
plt.grid(True)