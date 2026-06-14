import pandas as pd
from sklearn.preprocessing import OneHotEncoder

def encode_features(df: pd.DataFrame, one_hot_columns: list, binary_map: dict = {}) -> pd.DataFrame:
    """
    Encode categorical features in a DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame containing the features to encode.

    one_hot_columns : list
        List of column names to be one-hot encoded.

    binary_map : dict, default = {}
        Dictionary mapping columns to dictionaries of value maps.
        
    Returns
    -------
    pd.DataFrame
        A new DataFrame with encoded categorical features.
    """
    for column, maps in binary_map.items():
        df[column] = df[column].map(maps)

    encoder = OneHotEncoder(sparse_output = False).set_output(transform = "pandas")
    encoded_columns = encoder.fit_transform(df[one_hot_columns])

    df = pd.concat([df.drop(columns = one_hot_columns), encoded_columns], axis = 1)

    return df

