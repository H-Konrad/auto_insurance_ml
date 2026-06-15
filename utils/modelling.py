import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.base import BaseEstimator
from sklearn.pipeline import Pipeline



def encode_features(df: pd.DataFrame, one_hot_columns: list, binary_map: dict = {}) -> pd.DataFrame:
    """
    Encode categorical features in a DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame containing the features to encode.

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



def split_dataset(df: pd.DataFrame, target: str, test_size: float, stratify: bool = False, random_state: int = 123) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Split a dataset into training and testing sets.

    Parameters
    ----------
    df : pd.DataFrame
        The dataset to split.

    target : str
        Name of the target column. 

    test_size : float
        Proportion of the dataset to split.

    stratify : bool, default = False
        If True, perform a stratified split to keep the target distribution in 
        the training and testing sets.

    random_state : int, default = 123
        Random number to reproduce the data split. 
        
    Returns
    -------
    x_train : pd.DataFrame
        Training features.
    x_test : pd.DataFrame
        Testing features.
    y_train : pd.Series
        Training target values.
    y_test : pd.Series
        Testing target values.

    """
    x = df.drop(columns = [target])
    y = df[target]

    if stratify:
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = test_size, stratify = y, random_state = random_state)
    else:
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = test_size, stratify = None, random_state = random_state)

    return x_train, x_test, y_train, y_test



def create_pipeline(model: BaseEstimator, preprocessor: ColumnTransformer) -> Pipeline:
    """
    Create a preprocessing and modelling pipeline.

    Parameters
    ----------
    model : BaseEstimator
        A scikit-learn estimator.

    preprocessor : ColumnTransformer
        A scikit-learn ColumnTransformer.

    Returns
    -------
    Pipeline
        A scikit-learn Pipeline with preprocessing and modelling steps.
    """
    model = Pipeline(
        steps = [
            ("preprocessor", preprocessor),
            ("model", model)
        ]
    )

    return model

















































    