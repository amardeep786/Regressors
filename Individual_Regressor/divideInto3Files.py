import pandas as pd

def partition_file_by_script(file_path):
    try:
        # Read the main file
        df = pd.read_csv(file_path)
        
        # Check if the 'Current Script' column exists
        if 'Current Script' not in df.columns:
            raise ValueError("'Current Script' column not found in the file.")
        
        # Filter rows based on 'Current Script' values
        predict_df = df[df['Current Script'] == 'predict.py']
        detect_df = df[df['Current Script'] == 'detect.py']
        speech_df = df[df['Current Script'] == 'speech.py']
        
        # Generate new file paths
        base_path = file_path.rsplit('.', 1)[0]  # Remove file extension
        predict_file = f"{base_path}_predict.csv"
        detect_file = f"{base_path}_detect.csv"
        speech_file = f"{base_path}_speech.csv"
        
        # Save the partitioned data to new files
        predict_df.to_csv(predict_file, index=False)
        detect_df.to_csv(detect_file, index=False)
        speech_df.to_csv(speech_file, index=False)
        
        print(f"Files created successfully:\n- {predict_file}\n- {detect_file}\n- {speech_file}")
    except Exception as e:
        print(f"An error occurred: {e}")

# Example usage
# Replace 'your_file.csv' with the path to your file
file_path = r'/home/user/Desktop/CODE/Individual_Regressor/filtered_file.csv'
partition_file_by_script(file_path)