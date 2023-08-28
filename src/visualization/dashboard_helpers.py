########
# Libs #
########

from IPython.core.display import display, HTML
import matplotlib.pyplot as plt 
import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype
import seaborn as sns
import shap
from sklearn.base import is_classifier

def display_model_params(estimator: object):
    """Display model hyperparameters in Voila dashboard

    Args:

        estimator (object): sklearn estimator object
    """
    params_list = [estimator.get_params()]
    estimator_name = type(estimator).__name__
   
    model_param_df = pd.DataFrame(params_list).transpose()
    model_param_df.columns = [estimator_name]

    display(
                HTML('<h4 style="text-align:center;">Model parameters used</h4>')
            )
    display(model_param_df)
    
def plot_cmatrix(c_matrix_df: pd.DataFrame):
    """Plots confusion matrix based on a dataframe of true and predicted labels for each class.
    For classification problems only. 

    Args:

        c_matrix_df (pd.DataFrame): DataFrame of True and Predicted labels for each class
    """
    plt.figure(figsize=(6,4))
    sns.set(font_scale=1.4) # for label size
    
    sns.heatmap(c_matrix_df, 
                annot=True, 
                annot_kws={"size": 16}, 
                cmap='Blues', 
                fmt='g')
    
    plt.title('Confusion Matrix')
    plt.show()

def plot_roc(roc_curve_df: pd.DataFrame):
    """Plots ROC curve based on a dataframe of true positive rate, false positive rates at different thresholds
    for a given estimator.

    Args:

        roc_curve_df (pd.DataFrame): Dataframe with tpr, fpr, model type as columns
    """
    sns.set_style('whitegrid')
    plt.figure(figsize=(12,8))

    sns.lineplot(data = roc_curve_df, x='fpr', y='tpr', hue='model')
    plt.plot([0, 1], [0, 1], linestyle='--',  label='No model')

    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')

    plt.legend()
    plt.show()
    
def plot_pr(pr_curve_df: pd.DataFrame):
    """Plots precision and recall curves based on a dataframe of precision and recall values at different thresholds 
    for a given estimator.

    Args:

        pr_curve_df (pd.DataFrame): DataFrame with precision, recall, model type, 
        proportion of positive class ('proportion_1') as columns.
    """
    sns.set_style('whitegrid')
    plt.figure(figsize=(12,8))
    
    sns.lineplot(
        data = pr_curve_df, 
        x='recall', 
        y='precision', 
        hue='model'
    )
    sns.lineplot(
        data = pr_curve_df, 
        x='recall', 
        y='proportion_1', 
        label = 'No model', 
        linestyle='--'
    )
    
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.legend()
    plt.show()

def get_shap_values(
    estimator: object, 
    X_train: pd.DataFrame, 
    X_test: pd.DataFrame,
    n_samples: int = None
):
    """Get SHAP(SHapley Additive exPlanations) values for a sample of predictions in the test set.
    If n_samples is None, all of X_train will be used to estimate expected values to explain predictions 

    Args:

        estimator (object): trained estimator from sklearn framework

        X_train (pd.DataFrame): features used for training the model, used to estimate expected values. 
        
        X_test (pd.DataFrame): features used for predictions. 
        
        n_samples (int, optional): Number of samples in X_train used to estimate expected values, and
        number of samples in X_test to explain predictions for. Defaults to None.

    Returns:

        shap_values (array or list): For models with a single output (ie. regression models) this returns 
        a matrix of SHAP values (# samples x # features). 
        Each row sums to the difference between the model output for that sample and the expected value of the model 
        output (which is stored as expected_value attribute of the explainer). 
        For models with vector outputs (ie. classification) this returns a list of such matrices, one for each class.
        
        explainer (SHAP object): SHAP explainer object
    """
    if n_samples:
        X_train = shap.utils.sample(X_train, n_samples)
        X_test = shap.utils.sample(X_test, n_samples)

    if is_classifier(estimator):
        explainer = shap.KernelExplainer(
                estimator.predict_proba, 
                X_train
            )
    else:
        explainer = shap.KernelExplainer(
                estimator.predict, 
                X_train
            )

    shap_values = explainer.shap_values(X_test)

    return shap_values, explainer

def plot_shap_summary(
    estimator: object,
    shap_values: list, 
    X_test: pd.DataFrame,
    n_samples: int = None
):
    """Plots summary plot of shap values for test set. For binary classification problems, only the positive class 
    will be shown. If n_samples is None, SHAP values for all predictions in X_test will be used for the plot. 

    Args:

        estimator (object): trained estimator from sklearn framework
        
        shap_values (np.array or list): SHAP values. 
            - For models with a single output (ie. regression models) this is a matrix of SHAP values 
              (# samples x # features).
            - For models with vector outputs (ie. classification) this is a list of matrices 
              (# samples x # features), one for each class. 
        
        X_test (pd.DataFrame): features from test set

        n_samples (int, optional): Number of samples in X_test to explain predictions. Defaults to None.
    """
    shap.initjs()

    if n_samples:
        X_test = shap.utils.sample(X_test, n_samples) 

    if is_classifier(estimator):
        shap.summary_plot(shap_values[1], X_test)

    else:
        shap.summary_plot(shap_values, X_test)

def plot_shap_force(
    explainer: object, 
    shap_values: list, 
    X_test: pd.DataFrame,
    estimator: object,
    index: int = 0
):
    """Plot force plot of shap values for a particular prediction. For classification problems, only the positive
    class will be explained. 

    Args:

        explainer (object): SHAP Explainer object

        shap_values (np.array or list): SHAP values. 
            - For models with a single output (ie. regression models) this is a matrix of SHAP values 
              (# samples x # features).
            - For models with vector outputs (ie. classification) this is a list of matrices 
              (# samples x # features), one for each class. 
        
        X_test (pd.DataFrame): features from test set to explain predictions

        estimator (object): trained estimator from sklearn framework

        index (int): row index in X_test to visualise. Defaults to 0 (first prediction)
    """
    shap.initjs()

    if is_classifier(estimator):
        shap.force_plot(
            explainer.expected_value[1], 
            shap_values[1][index], 
            X_test[index]
        )
    else:
        shap.force_plot(
            explainer.expected_value, 
            shap_values[index], 
            X_test[index]
        )

def get_feature_importance(
    estimator: object, 
    train_features: list
) -> pd.DataFrame:
    """Get impurity-based feature importance scores from tree based estimators in the sklearn framework.

    Args:

        estimator (object): trained estimator in sklearn framework (tree based model)

        train_features (list): list of features used to train the estimator

    Returns:

        feature_importance (pd.DataFrame): DataFrame with Variable (features), Importance score as columns
    """
    feature_importance = pd.DataFrame(
        {
        'Variable': train_features,
        'Importance score': list(
            estimator.feature_importances_
            )
        }
    )
    sort_importance = feature_importance.sort_values(
        'Importance score', 
        ascending = False
    )
    feature_importance = sort_importance.reset_index(drop=True)

    return feature_importance

def plot_feature_importance(
    estimator: object, 
    feature_importance: pd.DataFrame, 
    plot_nfeatures: int
):
    """Plot impurity-based feature importance scores from Tree based models in sklearn framework.

    Args:

        estimator (object): trained estimator in sklearn framework (tree based model)

        feature_importance (pd.DataFrame): DataFrame with Variable (features), Importance score as columns
        
        plot_nfeatures (int): Top N features to plot
    """
    sns.set_style('whitegrid')
    plt.figure(figsize=(20,8))
    
    ax = sns.barplot(
        data = feature_importance.iloc[ 0 : plot_nfeatures, : ],
        x = 'Variable', 
        y = 'Importance score'
    )
    
    ax.tick_params(axis = 'x', rotation = 45)
    
    plt.title(
        f'Top {plot_nfeatures} importance scores (Mean Decrease in Impurity) from \
        {type(estimator).__name__}'
    )

    for p in ax.patches:
        width = p.get_width()
        height = p.get_height()
        x, y = p.get_xy() 
        ax.annotate(
            '{:.2f}'.format(height), 
            (x + width/2, y + height*1.01), 
            ha='center'
        )

    plt.show() 

def plot_feat_vs_target(
    df: pd.DataFrame, 
    shap_feature_importance: pd.DataFrame, 
    plot_nfeatures: int, 
    target: str,
    estimator: object
):
    """Plot bivariate plots of top N features against target using the entire dataset (df)

    Args:

        df (pd.DataFrame): entire dataset (before train test split) or existing dataset used to instantiate the class.
        
        shap_feature_importance (pd.DataFrame): DataFrame of feature importance scores based on SHAP values.
        
        plot_nfeatures (int): Top N features to plot
        
        target (str): Column name of target variable in df.
        
        estimator (object): trained estimator from sklearn framework.
    """
    # subset top n features
    feature_cols = shap_feature_importance['feature'].tolist()
    col_to_visualise = feature_cols[: plot_nfeatures]

    if not all(
        [col in df.columns.tolist() \
            for col in col_to_visualise]
    ):
        print(
            "Mismatch of columns in df used to instantiate Dashboard class \
            and features used in modelling. Please check feature columns."
        )
        
    n_subplots = int(len(col_to_visualise)/2 +1)

    plt.figure(figsize=(16,30))
    sns.set_theme(style="whitegrid")
    sns.set_palette('Set2')

    if is_classifier(estimator):

        df[target] = df[target].astype('category') # ensure correct data type

        for index, col in enumerate(col_to_visualise):

            plt.subplot(n_subplots, 2, index + 1)
            
            if is_numeric_dtype(df[col]):
                ax = sns.boxplot(
                    data = df, 
                    x = df[col], 
                    y = df[target]
                )

            else:    
                ax = sns.countplot(
                    x = df[col], 
                    data = df, 
                    hue = df[target]
                )

                # label the countplots with percentage
                for p in ax.patches:
                    percentage = '{:.1f}%'.format(
                        100 * p.get_height()/len(df[col])
                    )
                    x = p.get_x() + p.get_width() / 2 - 0.05
                    y = p.get_y() + p.get_height()
                    plt.annotate(percentage, (x, y),ha='center')
                    plt.title(col_to_visualise[index])
                    ax.set_xlabel('')

    # regressor: target is continuous
    else:

        for index, col in enumerate(col_to_visualise):

            plt.subplot(n_subplots, 2, index + 1)

            if is_numeric_dtype(df[col]):
                ax = sns.scatterplot(
                    data = df, 
                    x = df[col], 
                    y = df[target],
                    alpha = 0.6
                )
            else:
                ax = sns.boxplot(
                        data = df, 
                        x = df[col], 
                        y = df[target],
                        orient = 'v'
                    )

    sns.despine(top = True, right = True, left = True) # remove lines
    plt.subplots_adjust(
        left = None, 
        bottom = None, 
        right = None, 
        top = None, 
        wspace = None, 
        hspace = 0.4
    )
            
    plt.show()

def widget_option_unique_values(col: pd.Series) -> list:
    """Gets a list of unique values of a column in a DataFrame, sorts the values 
    and then inserts 'ALL' as the first element of the list. Used to set options for dropdown widget. 

    Args:

        col (pd.Series): Pandas Series of a column in a dataframe to be used for filtering the dataframe by
        using unique values within the column

    Returns:

        unique_list (list): list of unique values of a column in a DataFrame with 'ALL' as the first element.
    """
    unique_list = col.unique().tolist()
    unique_list.sort()
    unique_list.insert(0, 'ALL')

    return unique_list

def get_all_feature_names(df: pd.DataFrame, target: str = None) -> list:
    """Get a list of all feature names in a dataframe.

    Args:

        df (pd.DataFrame): dataframe of features and target variable 

        target (str): name of target column in df

    Returns:

        all_feature_names (list): list of all feature names
    """
    # if using the main df
    if target in df.columns.tolist():

        df = df.loc[ :, ~df.columns.isin([target])]
        all_feature_names = df.columns.tolist()
    
    # if using samples_df with true and predicted labels
    else:
        df = df.loc[ :, ~df.columns.isin(
            [
                'true_label', 
                'predicted_label'
                ]
            )
        ]
        all_feature_names = df.columns.tolist()

    return all_feature_names


def has_feature_importances_attr(estimator: object) -> bool:
    """Helper function to check if current estimator has the fixed attribute feature_importances_. 
    This attribute refers to the impurity-based feature importances within the sklearn framework. 
    Non-tree based estiamtors will result in an AttributeError.

    Args:

        estimator (object): estimator from sklearn framework

    Returns:

        bool: Whether the estimator has the attribute feature_importances_
    """
    
    try:
        estimator.feature_importances_
        return True

    except AttributeError:
        return False

def get_shap_feature_importance(
    estimator: object,
    feature_names: list, 
    shap_values: np.array
) -> pd.DataFrame:
    """Get a DataFrame of feature importance scores based on SHAP values. 
    For binary classification problems, only the positive class will be shown. 

    Args:

        estimator (object): trained model

        feature_names (list): feature names 

        shap_values (np.array_like): SHAP values

    Returns:

        shap_feature_importance (pd.DataFrame): dataframe of SHAP feature importance scores for each feature
    """
    if is_classifier(estimator):
        importance_scores = np.abs(shap_values[1]).mean(0) # for positive class
    
    else:
        importance_scores = np.abs(shap_values).mean(0)

    shap_feature_importance = pd.DataFrame(
    list(
        zip(
            feature_names, 
            importance_scores
        )
    ), 
        columns = ['feature', 'shap_feature_importance']
    )

    shap_feature_importance = shap_feature_importance.sort_values(
        by = 'shap_feature_importance',
        ascending = False
    )
    return shap_feature_importance
