import pandas as pd
from pathlib import Path


def load_application_train(data_path: str):
    
    # Create full path to the CSV file
    path = Path(data_path) / "application_train.csv"

    # Read CSV file into pandas DataFrame
    df = pd.read_csv(path)

    # Return loaded DataFrame
    return df
