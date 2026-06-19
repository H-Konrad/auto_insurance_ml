import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split, GridSearchCV, TunedThresholdClassifierCV
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.base import BaseEstimator
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score, classification_report, roc_curve
from sklearn.metrics import root_mean_squared_error, mean_absolute_error, r2_score



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
        Scikit-learn estimator.

    preprocessor : ColumnTransformer
        Scikit-learn ColumnTransformer.

    Returns
    -------
    model_pipeline : Pipeline
        Scikit-learn Pipeline.
    """
    model_pipeline = Pipeline(
        steps = [
            ("preprocessor", preprocessor),
            ("model", model)
        ]
    )

    return model_pipeline



def classification_results(model: Pipeline, x_train: pd.DataFrame, x_test: pd.DataFrame, 
                           y_train: pd.Series, y_test: pd.Series, report: bool = False) -> dict:
    """
    Runs classification model metrics and creates a dictionary of 
    different model metrics.
    
    Parameters
    ----------
    model : Pipeline
        scikit-learn Pipeline.

    x_train : pd.DataFrame
        Training features.
        
    x_test : pd.DataFrame
        Testing features.
        
    y_train : pd.Series
        Training target values.
        
    y_test : pd.Series
        Testing target values.

    report : bool, default = False
        True to print classification report.

    Returns
    -------
    results : dict
        A dictionary of different model metrics.
    """
    y_train_prediction = model.predict(x_train)
    y_train_probability = model.predict_proba(x_train)
    y_test_prediction = model.predict(x_test)
    y_test_probability = model.predict_proba(x_test)

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
        print("Classification Report: \n", classification_report(y_train, y_train_prediction))

    print("\nTESTING METRICS")
    print(f"Accuracy: {results['testing_accuracy']:.4f}")
    print("Confusion Matrix: \n", results["testing_confusion_matrix"])
    if report:
        print("Classification Report: \n", classification_report(y_test, y_test_prediction))

    return results



def optimise_model(model: Pipeline, parameters: dict, scoring: str, cv: int, x_train: pd.DataFrame, 
                   y_train: pd.Series, n_jobs: int = -1) -> GridSearchCV:
    """
    Optimise a model using a grid search.

    Parameters
    ----------
    model : Pipeline
        Scikit-learn Pipeline.

    parameters : dict
        Parameters for the model in the pipeline.

    scoring : str
        Scoring metric.

    cv : int
        Number of cross-validation folds.

    x_train : pd.DataFrame
        Training features.

    y_train : pd.Series
        Training target values.

    n_jobs : int, default = -1
        Number of CPU cores.

    Returns
    -------
    best_model : estimator
        Scikit-learn estimator.
    """
    model_gs = GridSearchCV(model, parameters, scoring = scoring, cv = cv, n_jobs = n_jobs)
    model_gs.fit(x_train, y_train)
    best_model = model_gs.best_estimator_

    print("BEST PARAMETERS")
    print(model_gs.best_params_)

    return best_model
    
    

def optimise_threshold(model: Pipeline | GridSearchCV, scoring: str, cv: int, x_train: pd.DataFrame, 
                       y_train: pd.Series, random_state: int = 123) -> TunedThresholdClassifierCV:
    """
    Optimise a model threshold using TunedThresholdClassifierCV.

    Parameters
    ----------
    model : Pipeline | GridSearchCV
        Scikit-learn Pipeline or GridSearchCV.

    scoring : str
        Scoring metric.

    cv : int
        Number of cross-validation folds.

    x_train : pd.DataFrame
        Training features.

    y_train : pd.Series
        Training target values.

    random_state : int, default = 123
        Random number to reproduce the threshold tuning. 

    Returns
    -------
    model_ttc : GridSearchCV
        Scikit-learn GridSearchCV.
    """
    model_ttc = TunedThresholdClassifierCV(model, scoring = scoring, cv = cv, random_state = random_state)
    model_ttc.fit(x_train, y_train)

    print("THRESHOLD")
    print(f"{model_ttc.best_threshold_:.4f} \n")

    return model_ttc



def store_classification_results(df: pd.DataFrame, file_path: str, dataset_version: str, model_name: str, 
                                 stage: str, threshold: float, results: dict, save: bool = True) -> pd.DataFrame:
    """
    Log evaluation metrics.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing recorded model metrics.
        
    file_path : str
        Path to save results.
        
    dataset_version : str
        Version of the dataset used for training.
        
    model_name : str
        Name of the model being used.
        
    stage : str
        Stage of initial training, optimisation, or threshold tuning.
        
    threshold : float
        Classification probability threshold.
        
    results : dict
        Dictionary containing evaluation outputs.

    save : bool, default = True
        Whether to save the results to disk. 

    Returns
    -------
    metrics_df : pd.DataFrame
        DataFrame new results. 
    """
    better_results = {
        "dataset_version": dataset_version,
        "model": model_name,
        "stage": stage,
        "tuned_threshold": threshold,
        "accuracy": results["testing_accuracy"],
        "true_positive": results["testing_confusion_matrix"][0][0],
        "false_negative": results["testing_confusion_matrix"][0][1],
        "false_positive": results["testing_confusion_matrix"][1][0],
        "true_negative": results["testing_confusion_matrix"][1][1],
        "precision_0": results["testing_classification_report"]["0"]["precision"],
        "recall_0": results["testing_classification_report"]["0"]["recall"],
        "f1_score_0": results["testing_classification_report"]["0"]["f1-score"],
        "precision_1": results["testing_classification_report"]["1"]["precision"],
        "recall_1": results["testing_classification_report"]["1"]["recall"],
        "f1_score_1": results["testing_classification_report"]["1"]["f1-score"],
        "fpr": results["testing_roc"][0], 
        "tpr": results["testing_roc"][1],
        "thresholds": results["testing_roc"][2]
    }

    results_df = pd.DataFrame([better_results])
    metrics_df = pd.concat([df, results_df], ignore_index = True)

    if save:
        metrics_df.to_parquet(file_path, index = False, engine = "pyarrow")

    return metrics_df



def regression_results(model: TransformedTargetRegressor, x_train: pd.DataFrame, 
                       x_test: pd.DataFrame, y_train: pd.Series, y_test: pd.Series) -> dict:
    """
    Runs regression model metrics and creates a dictionary of 
    different model metrics.
    
    Parameters
    ----------
    model : Pipeline
        Scikit-learn TransformedTargetRegressor.

    x_train : pd.DataFrame
        Training features.
        
    x_test : pd.DataFrame
        Testing features.
        
    y_train : pd.Series
        Training target values.
        
    y_test : pd.Series
        Testing target values.

    Returns
    -------
    results : dict
        A dictionary of different model metrics.
    """
    y_train_prediction = model.predict(x_train)
    y_test_prediction = model.predict(x_test)

    results = {
        "training_rmse": root_mean_squared_error(y_train, y_train_prediction),
        "training_mae": mean_absolute_error(y_train, y_train_prediction),
        "training_r2": r2_score(y_train, y_train_prediction),
        "testing_rmse": root_mean_squared_error(y_test, y_test_prediction),
        "testing_mae": mean_absolute_error(y_test, y_test_prediction),
        "testing_r2": r2_score(y_test, y_test_prediction),
    }

    print("TRAINING METRICS")
    print(f"RMSE: {results["training_rmse"]:.0f}")
    print(f"MAE: {results["training_mae"]:.0f}")
    print(f"R^2: {results["training_r2"]:.3f}\n")

    print("TESTING METRICS")
    print(f"RMSE: {results["testing_rmse"]:.0f}")
    print(f"MAE: {results["testing_mae"]:.0f}")
    print(f"R^2: {results["testing_r2"]:.3f}")



def transform_target(model: Pipeline, function: np.ufunc, inverse: np.ufunc) -> TransformedTargetRegressor:
    """
    Applies transformation step to the target variable.
    
    Parameters
    ----------
    model : Pipeline
        Scikit-learn Pipeline.

    function : np.ufunc
        Function applied to the target.
        
    inverse : np.ufunc
        Inverse of the function to get back the original target.

    Returns
    -------
    model_ttr : TransformedTargetRegressor
        Scikit-learn TransformedTargetRegressor.
    """
    model_ttr = TransformedTargetRegressor(
        regressor = model, 
        func = function, 
        inverse_func = inverse
    )

    return model_ttr



def store_regression_results(df: pd.DataFrame, file_path: str, dataset_version: str, model_name: str, 
                             stage: str, results: dict, save: bool = True) -> pd.DataFrame:
    """
    Log evaluation metrics.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing recorded model metrics.
        
    file_path : str
        Path to save results.
        
    dataset_version : str
        Version of the dataset used for training.
        
    model_name : str
        Name of the model being used.
        
    stage : str
        Stage of initial training or optimisation.
        
    results : dict
        Dictionary containing evaluation outputs.

    save : bool, default = True
        Whether to save the results to disk. 

    Returns
    -------
    metrics_df : pd.DataFrame
        DataFrame new results. 
    """
    better_results = {
        "dataset_version": dataset_version,
        "model": model_name,
        "stage": stage,
        "rmse": df["testing_rmse"],
        "mae": df["testing_mae"],
        "r2": df["testing_r2"]
    }

    results_df = pd.DataFrame([better_results])
    metrics_df = pd.concat([df, results_df], ignore_index = True)

    if save:
        metrics_df.to_parquet(file_path, index = False, engine = "pyarrow")

    return metrics_df    