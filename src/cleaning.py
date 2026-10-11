import pandas as pd

def convert_datetime_columns(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:   #convert specific columns to pandas datetime format
    df = df.copy()
    for column in columns:
        if column in df.columns:
            df[column] = pd.to_datetime(df[column], errors = 'coerce')
    return df


def convert_numeric_columns(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame: # convert specific columns to pandas numeric format
    df = df.copy()
    for column in columns:
        if column in df.columns:
            df[column] = pd.to_numeric(df[column], errors = 'coerce')
    return df
