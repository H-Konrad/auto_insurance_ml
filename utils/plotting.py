import pandas as pd
import matplotlib.pyplot as plt



def plot_feature_importance(x_train: pd.DataFrame, dict_for_df: dict, head_n: int = 5, tail_n: int = 5, 
                            plot: bool = False, figsize: tuple = (8, 10)):
    """
    Plots or write the feature importance for models.

    Parameters
    ----------
    x_train : pd.DataFrame
        Training features.
        
    dict_for_df : dict
        Dictionary with model names and feature importance.
        
    head_n : int, default = 5
        Number of the most important features to plot.
        
    tail_n : int, default = 5
        Number of the least important features to plot.
        
    plot : bool, default = False
        If True, plots the feature importance, else prints it.
        
    figsize : tuple, default = (8, 10)
        Figure size.
    """
    features_df = pd.DataFrame(dict_for_df)
    features_df["Features"] = x_train.columns
    if "Logistic Regression" in dict_for_df.keys():  
        features_df["Logistic Regression"] = features_df["Logistic Regression"].abs()
    if "Linear Regression" in dict_for_df.keys():
        features_df["Linear Regression"] = features_df["Linear Regression"].abs()

    if plot:
        fig, axes = plt.subplots(len(dict_for_df), 1, figsize = figsize, constrained_layout = True)
        axes = axes.flatten()
        plt.suptitle("Feature Importance", fontsize = 14)

        for ax, name in zip(axes, dict_for_df.keys()):
            top_features = features_df.nlargest(head_n, name)
            bottom_features = features_df.nsmallest(tail_n, name)
            plotting_df = pd.concat([bottom_features, top_features])

            plotting_df.plot(kind = "barh", x = "Features", y = name, ax = ax)

        plt.show()
    else:
        for name in dict_for_df.keys():
            print(name)
            print(features_df[["Features", name]].sort_values(name).head(10), "\n")
        

    