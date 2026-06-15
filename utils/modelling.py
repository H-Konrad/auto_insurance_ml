import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.base import BaseEstimator
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score, classification_report, roc_curve
from sklearn.model_selection import GridSearchCV


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
    df : pd.DataFrame
        A new DataFrame with encoded categorical features.
    """
    for column, maps in binary_map.items():
        df[column] = df[column].map(maps)

    encoder = OneHotEncoder(sparse_output = False).set_output(transform = "pandas")
    encoded_columns = encoder.fit_transform(df[one_hot_columns])

    df = pd.concat([df.drop(columns = one_hot_columns), encoded_columns], axis = 1)

    return df



def split_dataset(df: pd.DataFrame, target: str, test_size: float, stratify: bool = False, 
                  random_state: int = 123) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
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
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = test_size, 
                                                            stratify = y, random_state = random_state)
    else:
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = test_size, 
                                                            stratify = None, random_state = random_state)

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
    model_pipeline : Pipeline
        A scikit-learn Pipeline with preprocessing and modelling steps.
    """
    model_pipeline = Pipeline(
        steps = [
            ("preprocessor", preprocessor),
            ("model", model)
        ]
    )

    return model_pipeline



def classification_results(model_pipeline: Pipeline, x_train: pd.DataFrame, x_test: pd.DataFrame, 
                           y_train: pd.Series, y_test: pd.Series, report: bool = False) -> dict:
    """
    Runs classification model metrics and creates a dictionary of 
    different model metrics.
    
    Parameters
    ----------
    model_pipeline : Pipeline
        A scikit-learn Pipeline.

    x_train : pd.DataFrame
        Training features.
        
    x_test : pd.DataFrame
        Testing features.
        
    y_train : pd.Series
        Training target values.
        
    y_test : pd.Series
        Testing target values.

    report : bool
        True to print classification report.

    Returns
    -------
    results : dict
        A dictionary of different model metrics.
    """
    y_train_prediction = model_pipeline.predict(x_train)
    y_train_probability = model_pipeline.predict_proba(x_train)
    y_test_prediction = model_pipeline.predict(x_test)
    y_test_probability = model_pipeline.predict_proba(x_test)

    results = {
        "training_accuracy": accuracy_score(y_train, y_train_prediction),
        "training_roc": roc_curve(y_train, y_train_probability[:, 1]),
        "training_confusion_matrix": confusion_matrix(y_train, y_train_prediction),
        "training_classification_report": classification_report(y_train, y_train_prediction, output_dict = True),
        "training_y_pred": y_train_prediction,
        "training_y_prob": y_train_probability,
        "testing_accuracy": accuracy_score(y_test, y_test_prediction),
        "testing_roc": roc_curve(y_test, y_test_probability[:, 1]),
        "testing_confusion_matrix": confusion_matrix(y_test, y_test_prediction),
        "testing_classification_report": classification_report(y_test, y_test_prediction, output_dict = True),
        "testing_y_pred": y_test_prediction,
        "testing_y_prob": y_test_probability
    }
    
    print("TRAINING METRICS")
    print(f"Accuracy: {results['training_accuracy']:.4f}")
    print("Confusion Matrix: \n", results["training_confusion_matrix"])
    if report:
        print("Classification Report: \n", results["training_classification_report"])

    print("TESTING METRICS")
    print(f"Accuracy: {results['testing_accuracy']:.4f}")
    print("Confusion Matrix: \n", results["testing_confusion_matrix"])
    if report:
        print("Classification Report: \n", results["testing_classification_report"])

    return results



def optimise_model(model_pipeline: Pipeline, parameters: dict, scoring: str, cv: int, x_train: pd.DataFrame, 
                   y_train: pd.Series, n_jobs: int = -1) -> GridSearchCV:
    """
    Optimise a model using a grid search.

    Parameters
    ----------
    model_pipeline : Pipeline
        A Scikit-learn pipeline.

    param_grid : dict
        Parameters for the model in the pipeline.

    scoring : str
        Scoring metric.

    cv : int
        Number of cross-validation folds.

    x_train : pd.DataFrame
        Training features.

    y_train : pd.Series
        Training target values.

    n_jobs : int
        Number of CPU cores.

    Returns
    -------
    search : GridSearchCV
        Scikit-learn GridSearchCV object.
    """
    model_gs = GridSearchCV(model_pipeline, parameters, scoring = scoring, cv = cv, n_jobs = n_jobs)
    model_gs.fit(x_train, y_train)

    print("BEST PARAMETERS")
    print(model_gs.best_params_)

    return model_gs
    
    













































    