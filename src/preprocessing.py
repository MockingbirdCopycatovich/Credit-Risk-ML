def clean_data(df):
    # Remove columns where >50% missing values
    missing_percent = df.isnull().mean()
    cols_to_drop = missing_percent[missing_percent > 0.5].index
    df = df.drop(columns=cols_to_drop)

    # Create AGE
    df["AGE"] = -df["DAYS_BIRTH"] / 365
    df = df.drop(columns=["DAYS_BIRTH"])

    return df
