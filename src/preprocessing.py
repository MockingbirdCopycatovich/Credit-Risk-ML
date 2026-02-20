import pandas as pd 

def clean_data(df):
    # AGE CREATION
    # If the 'AGE' column doesn't exist, create it from 'DAYS_BIRTH'
    # 'DAYS_BIRTH' is negative (days since birth), so we negate it and convert to years
    if "AGE" not in df.columns:
        age_df = pd.DataFrame({"AGE": -df["DAYS_BIRTH"] / 365}, index=df.index)
        df = pd.concat([df.drop(columns=["DAYS_BIRTH"]), age_df], axis=1)

    # MISSING VALUE FLAGS
    # Calculate percentage of missing values per column
    missing_percent = df.isnull().mean()
    # Select columns where more than 30% of values are missing
    cols_with_missing = missing_percent[missing_percent > 0.3].index.tolist()
    
    if cols_with_missing:
        # For each column with many missing values, create a flag column
        # 1 if value is missing, 0 otherwise
        missing_flags = pd.DataFrame(
            {f"FLAG_{col}_MISSING": df[col].isnull().astype(int) 
             for col in cols_with_missing},
            index=df.index
        )
        # Add all missing value flags at once to avoid performance issues
        df = pd.concat([df, missing_flags], axis=1)

    # NEW FEATURES / FEATURE ENGINEERING
    # Creating features that may help the model predict credit default risk
    new_features = pd.DataFrame({
        # How much the client borrowed relative to their income
        "CREDIT_INCOME_RATIO": df["AMT_CREDIT"] / df["AMT_INCOME_TOTAL"],
        # How big the annuity payment is relative to the total credit
        "ANNUITY_CREDIT_RATIO": df["AMT_ANNUITY"] / df["AMT_CREDIT"],
        # Ratio of days employed to age, shows employment stability
        "EMPLOYED_AGE_RATIO": -df["DAYS_EMPLOYED"] / df["AGE"]
    }, index=df.index)

    # Add the new features to the original DataFrame
    df = pd.concat([df, new_features], axis=1)

    return df