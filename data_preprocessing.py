import pandas as pd

def load_and_clean_data(file_path):
    """
    Load the dataset and perform basic cleaning.
    """
    try:
        # Load the dataset with the correct encoding
        data = pd.read_csv(file_path, encoding='ISO-8859-1')  # Common encoding for CSV files
    except UnicodeDecodeError:
        # Try a different encoding if needed
        data = pd.read_csv(file_path, encoding='latin1')

    # Drop duplicates
    data.drop_duplicates(inplace=True)

    # Handle missing values (example: drop rows with NaNs)
    data.dropna(inplace=True)

    print("Data after cleaning:")
    print(data.info())

    return data
