############
# Built-in #
############
from datetime import date
import logging
from pathlib import Path
import pickle

########
# Libs #
########
from category_encoders import one_hot
from flaml import AutoML
import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype
from sklearn.base import is_classifier, is_regressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report, 
    confusion_matrix, 
    roc_auc_score, 
    accuracy_score, 
    precision_score, 
    recall_score, 
    log_loss,
    explained_variance_score, 
    mean_absolute_error, 
    mean_squared_error,
    r2_score,
    roc_curve,
    precision_recall_curve
)
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import (
    LabelEncoder, 
    MinMaxScaler, 
    StandardScaler, 
    RobustScaler
)
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

logger = logging.getLogger(__name__)

class SupervisedModel:

    def __init__(self, 
                 model_type: str, 
                 df: pd.DataFrame, 
                 target: str,
                 cfg: object = None,
                 load_path: str = None): 
        """"Create model pipeline for supervised model

        Args:
            model_type (str): model type to instantiate model class with. 
            df (pd.DataFrame): cleaned dataframe.
            target (str): name of target variable in cleaned dataframe (df)
            config (module/object, optional): object containing all config variables for training pipeline. 
                                              Optional when loading a trained model. Defaults to None. 
            load_path(str, optional): absolute file path of model to load. Defaults to None. 
        """

        self.model_type = model_type
        self.load_path = load_path
        self.df = df # cleaned/preprocessed df
        self.target = target
        
        self.fitted_scaler = None
        self.fitted_onehot_encoder = None
        self.fitted_label_encoder = None
        self.class_labels = None
        self.train_features = None
        self.inference_features = None
        self.estimator = None
        self.trained_estimator = None
        self.train_test_data = None

        # attributes from config
        self.automl_settings = getattr(cfg, 'AUTOML_SETTINGS', None)
        self.test_size = getattr(cfg, 'TEST_SIZE', 0.3)
        self.seed = getattr(cfg, 'SEED', None)
        self.scaler = getattr(cfg, 'SCALER', None)
        self.onehot_cols = getattr(cfg, 'ONEHOT_COLS', None)
        self.params = getattr(cfg, 'PARAMS', None)
        self.model_dir = getattr(cfg, 'MODEL_DIR', None)
        self.n_samples = getattr(cfg, 'N_SAMPLES', 20)

        self.set_estimator()
          
    def set_estimator(self):
        """Sets the estimator of choice."""
        
        try:
            if self.model_type == "RFC":
                self.estimator = RandomForestClassifier

            elif self.model_type == "RFR":
                self.estimator = RandomForestRegressor

            elif self.model_type == "LR":
                self.estimator = LogisticRegression

            elif self.model_type == "DT":
                self.estimator = DecisionTreeClassifier
            
            elif self.model_type == "SVM":
                self.estimator = SVC

            elif self.model_type == "KNN":
                self.estimator = KNeighborsClassifier

            elif self.model_type == "Auto-Classification":
                self.estimator = AutoML
            
            elif self.model_type == "Auto-Regression":
                self.estimator = AutoML

            else:
                raise ValueError("Unsupported model type specified")

        except ValueError as v:
            logger.error(
                f"Model: set_estimator error occured while instantiating the class:{v}"
            )
    
    ##########
    # Helpers
    ##########

    def get_X_vals(self, df: pd.DataFrame) ->  pd.DataFrame:
        """Get feature values (X values) from a dataframe

        Args:
            df (pd.DataFrame): dataframe of features and labels

        Returns:
            X (pd.DataFrame): DataFrame with features only
        """
        
        X = df.loc[:, ~df.columns.isin([self.target])]
        return X

    def get_Y_vals(self, df: pd.DataFrame) -> np.array:
        """Get target column (Y values)

        Args:
            df (pd.DataFrame): dataframe of features and labels

        Returns:
            Y (np.array): 1d numpy array with target column labels only
        """
        
        y = df.loc[:, [self.target]].values.ravel()
        return y
    
    def onehot_encode(self, 
                      X: pd.DataFrame,
                      onehot_cols: list) -> np.array:
        """Performs one-hot encoding for stated columns. 
        Saves fitted one hot encoder as class attribute fitted_onehot_encoder
        Returns an array of feature values

        Args:
            X (pd.DataFrame): pandas DataFrame with feature columns 
            onehot_cols (list): list of categorical columns for one hot encoding

        Returns:
            np.array: One hot encoded feature values
        """

        onehot_encoder = one_hot.OneHotEncoder(
            cols = onehot_cols,
            handle_unknown = 'error', 
            use_cat_names = True
        )

        enc_df = onehot_encoder.fit_transform(X)
        self.fitted_onehot_encoder = onehot_encoder
        self.train_features = onehot_encoder.get_feature_names()

        return enc_df.values
        
    def encode_labels(self, Y_vals) -> np.array:
        """Encodes target values

        Args:
            Y_vals (np.array): target labels 

        Returns:
            np.array: encoded target labels
        """
        
        label_encoder = LabelEncoder().fit(Y_vals)

        self.fitted_label_encoder = label_encoder
        self.class_labels = label_encoder.classes_.tolist()

        encoded_Y_vals = label_encoder.transform(Y_vals)

        return encoded_Y_vals

    def scale_arrays(self,
                     X_train: np.array, 
                     X_test: np.array, 
                     type: str = None) -> np.array:
        """Scales X_train and X_test arrays. Defaults to MinMaxScaling if type is not stated.

        Args:
            X_train (np.array): feature array for training set 
            X_test (np.array): feature array for test set
            type (str, optional): Whether to standardize features by removing the mean 
                                and scaling to unit variance (standard) or scale using 
                                median and quantile range (robust). If type is not stated, 
                                defaults to MinMaxScaling. 

        Returns:
             X_train_scaled, X_test_scaled [np.array]: scaled X_train and X_test arrays
        """

        if type == 'standard':
            fitted_scaler = StandardScaler().fit(X_train)
        
        if type == 'robust':
            fitted_scaler = RobustScaler().fit(X_train)

        else:
            fitted_scaler = MinMaxScaler().fit(X_train)

        X_train_scaled = fitted_scaler.transform(X_train)
        X_test_scaled = fitted_scaler.transform(X_test)

        self.fitted_scaler = fitted_scaler

        return X_train_scaled, X_test_scaled
    
    def get_cls_report(self, 
                       y_test: np.array, 
                       y_pred: np.array) -> pd.DataFrame:
        """Get classification report in a pandas DataFrame

        Args:
            y_test (np.array): true labels
            y_pred (np.array): predicted labels

        Returns:
            cls_df[pd.DataFrame]: classification report in DataFrame
        """

        cls_report = classification_report(y_test, y_pred, output_dict=True)

        cls_df = pd.DataFrame(cls_report).transpose()
        cls_df = cls_df.drop(index='accuracy') # drop accuracy
        cls_df['support'] = cls_df['support'].astype('int')
        cls_df = cls_df.round(2)

        return cls_df

    def get_confusion_matrix(self, 
                             y_test: np.array, 
                             y_pred: np.array) -> pd.DataFrame:
        """Get confusion matrix in a pandas DataFrame

        Args:
            y_test (np.array): true labels
            y_pred (np.array): predicted labels

        Returns:
            c_matrix_df[pd.DataFrame]: confusion matrix DataFrame
        """
        
        unique_label = np.unique([y_test, y_pred])

        c_matrix_df = pd.DataFrame(
            confusion_matrix(
                y_test, 
                y_pred, 
                labels = unique_label
                ), 
            index = [f'True: {label}' for label in unique_label], 
            columns = [f'Predicted: {label}' for label in unique_label]
        )

        return c_matrix_df

    def get_pred_samples(self, 
                         X_test: np.array, 
                         y_test: np.array, 
                         y_pred: np.array, 
                         n_samples: int) -> pd.DataFrame:
        """Returns a random sample of features, true labels and predicted labels

        Args:
            X_test (np.array): features
            y_test (np.array): true labels
            y_pred (np.array): predicted labels
            n_samples (int): number of rows to obtain for manual inspection of predicted labels (features+labels)

        Returns:
            sample_df[pd.DataFrame]: DataFrame of features, labels and predicted labels
        """

        train_features = self.train_features

        pred_df = pd.DataFrame(
            data = X_test, 
            columns = train_features
        )

        pred_df['true_label'] = y_test
        pred_df['predicted_label'] = y_pred

        if is_classifier(self.trained_estimator):

            # get randomly stratified sample based on true label proportions
            grouped = pred_df.groupby('true_label', group_keys = False)
            sample_based_on_grp_proportion = grouped.apply(
                lambda x: x.sample(
                    int(
                        n_samples*len(x)/len(pred_df)
                    )
                )
            )
            sample_df = sample_based_on_grp_proportion.sample(frac=1) # shuffle
        
        else:

            sample_df = pred_df.sample(n_samples)

        return sample_df
    
    def get_classification_scores(self, 
                                  estimator: object, 
                                  y_test: np.array, 
                                  y_pred: np.array,
                                  proba: np.array) -> pd.DataFrame:
        """Gets performance metrics for classification model evaluation

        Args:
            estimator (object): instantiated model from sklearn framework
            y_test (np.array): true labels
            y_pred (np.array): predicted labels
            proba (np.array): predicted probabilities

        Returns:
            scores_df[pd.DataFrame]: dataframe of classification performance scores 
        """

        acc_score = accuracy_score(y_test, y_pred)
        auc_score = roc_auc_score(y_test, y_pred)
        logloss = log_loss(y_test, proba)

        precision_binary = precision_score(y_test, y_pred, average='binary')
        recall_binary = recall_score(y_test, y_pred, average='binary')

        precision_wgt = precision_score(y_test, y_pred, average = 'weighted')
        recall_wgt = recall_score(y_test, y_pred, average='weighted')

        precision_macro = precision_score(y_test, y_pred, average = 'macro')
        recall_macro = recall_score(y_test, y_pred, average='macro')

        precision_micro = precision_score(y_test, y_pred, average ='micro')
        recall_micro = recall_score(y_test, y_pred, average='micro')
       
        scores = {
            'Accuracy': acc_score,
            'AUC': auc_score,
            'Log loss': logloss,
            'Precision binary': precision_binary,
            'Recall binary': recall_binary,
            'Precision weighted': precision_wgt,
            'Recall weighted': recall_wgt,
            'Precision macro': precision_macro,
            'Recall macro': recall_macro,
            'Precision micro': precision_micro,
            'Recall micro': recall_micro
        }
        scores_df = pd.DataFrame(
            scores, 
            index = [type(estimator).__name__]
        )

        return scores_df

    def get_regression_scores(self, 
                              estimator: object, 
                              y_test: np.array, 
                              y_pred: np.array) -> pd.DataFrame:
        """Gets performance metrics for regression model evaluation

        Args:
            estimator (object): instantiated model from sklearn framework
            y_test (np.array): true labels
            y_pred (np.array): predicted labels

        Returns:
            scores_df[pd.DataFrame]: dataframe of regression performance scores 
        """

        explained_variance = explained_variance_score(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        rmse = mean_squared_error(y_test, y_pred, squared = False)

        scores = {
            'Explained Variance': explained_variance,
            'R2': r2,
            'Mean Absolute Error': mae,
            'Mean Squared Error': mse,
            'Root Mean Squared Error': rmse
        }
        scores_df = pd.DataFrame(
            scores, 
            index = [type(estimator).__name__]
        )

        return scores_df
    
    def get_roc_curve(self,
                      y_test: np.array,
                      proba: np.array, 
                      estimator: object = None) -> pd.DataFrame:
        """Create a dataframe of false postive rates, true positive rates for different
        thresholds using y_test/true labels and predicted probabilities of the positive class. 
        Works for binary classification problems only. 

        Args:
            y_test (np.array): true labels 
            proba (np.array): array of predicted probabilities for each class 
            estimator (object): sklearn estimator object

        Returns:
            roc_curve_df[pd.DataFrame]: dataframe of roc curve values for an estimator
        """
        
        if estimator is None:
            estimator = self.trained_estimator

        # keep probabilities for the positive outcome only - binary case
        pos_prob = proba[:, 1]
        
        fpr, tpr, _ = roc_curve(y_test, pos_prob)
        
        roc_curve_df = pd.DataFrame(
            {
                'fpr': fpr,
                'tpr': tpr,
            }
        )
        roc_curve_df['model'] = type(estimator).__name__
        
        return roc_curve_df

    def get_pr_curve(self,
                     y_test: np.array,
                     proba: np.array, 
                     estimator: object = None) -> pd.DataFrame:
        """Create a dataframe of precision and recall scores at different thresholds.
        Only for binary classification problem (for now).

        Args:
            y_test (np.array): true labels 
            proba (np.array): array of predicted probabilities for each class 
            estimator (object): sklearn estimator object

        Returns:
            pr_curve_df[pd.DataFrame]: dataframe with precision, recall for different thresholds, 
            model type and proportion of the positive class ('proportion_1') as columns
        """

        if estimator is None:
            estimator = self.trained_estimator

        # keep probabilities for the positive outcome only - binary case
        pos_prob = proba[:, 1]
        
        precision, recall, _ = precision_recall_curve(y_test, pos_prob)
        
        pr_curve_df = pd.DataFrame(
            {
                'precision': precision,
                'recall': recall
            }
        )
        pr_curve_df['model'] = type(estimator).__name__
        pr_curve_df['proportion_1'] = len(y_test[y_test==1]) / len(y_test)
        
        return pr_curve_df

    ##################
    # Core functions
    ##################

    def train(self):
        """Creates a training pipeline

        Returns:
            estimator [object]: trained estimator from sklearn framework
            train_test_data [dict]: dictionary of transformed train test data with the following keys:
                                    'X_train': Features in training set 
                                    'X_test': Features in test set 
                                    'y_train': Labels in training set 
                                    'y_test': Labels in test set 
        """
        
        Y_vals = self.get_Y_vals(self.df)
        X_vals = self.get_X_vals(self.df)

        if not is_numeric_dtype(Y_vals):
            Y_vals = self.encode_labels(Y_vals)

        if self.onehot_cols:
            X_vals = self.onehot_encode(X_vals, self.onehot_cols)
        else:
            self.train_features = X_vals.columns.tolist()
            X_vals = X_vals.values

        if is_classifier(self.estimator) \
            or self.model_type == "Auto-Classification":

            X_train, X_test, y_train, y_test = train_test_split(
                X_vals, 
                Y_vals, 
                test_size = self.test_size,
                random_state = self.seed,
                stratify = Y_vals,
                shuffle = True
            )
        else:
            X_train, X_test, y_train, y_test = train_test_split(
                X_vals, 
                Y_vals, 
                test_size = self.test_size,
                random_state = self.seed,
                shuffle = True
            )

        if self.scaler:
            X_train, X_test = self.scale_arrays(
                X_train = X_train, 
                X_test = X_test, 
                type = self.scaler
            )

        if self.model_type == "Auto-Classification" or \
            self.model_type == "Auto-Regression":

            automl_estimator = self.estimator()
            automl_estimator.fit(
                X_train = X_train,
                y_train = y_train,
                **self.automl_settings
            )
            estimator = automl_estimator.model # get only the best model
        
        else:

            estimator = self.estimator(**self.params)
            estimator.fit(X_train, y_train)

        self.trained_estimator = estimator

        # for use in other methods
        train_test_data = {
            'X_train': X_train,
            'X_test': X_test,
            'y_train': y_train,
            'y_test': y_test
        }
        self.train_test_data = train_test_data

        return estimator, train_test_data

    def predict(self, 
                estimator: object = None, 
                X_test: np.array = None) -> np.array:
        """Make predictions from a trained model

        Args:
            estimator (object, optional): trained model to use to predict labels. Defaults to None.
            X_test (np.array, optional): Preprocessed features from test/validation set to use for predictions. 
                                         Defaults to None. 
   
        Returns:
           y_pred[np.array]: array of predicted labels
        """

        if estimator is None:
            estimator = self.trained_estimator
        
        if X_test is None:
            X_test = self.train_test_data.get('X_test')

        y_pred = estimator.predict(X_test)

        return y_pred

    def predict_proba(self,
                     estimator: object = None, 
                     X_test: np.array = None) -> np.array:
        """Predict probabilty of each class from features. For classification
        problems only. 

        Args:
            estimator (object, optional): SKlearn estimator object. Defaults to None.
            X_test (np.array, optional): Preprocessed features from test/validation set to use for predictions. 
                                         Defaults to None. 

        Returns:
            proba[np.array]: array of predicted probabilities for each class. 
                             Each element at (i, j) is the probability for instance i to be in class j
        """

        if estimator is None:
            estimator = self.trained_estimator
        
        if X_test is None:
            X_test = self.train_test_data.get('X_test')
        
        proba = estimator.predict_proba(X_test)

        return proba

    def evaluate(self, 
                 estimator: object = None, 
                 X_test: np.array = None,
                 y_test: np.array = None, 
                 y_pred: np.array = None,
                 proba: np.array = None,
                 n_samples: int = None,
                 report_dir: str = None) -> dict:
        """Evaluate a trained model

        Args:
            estimator (object, optional): Trained model from sklearn framework. Defaults to None.
            X_test (np.array, optional): Preprocesssed features to be used for prediction. Defaults to None.
            y_test (np.array, optional): True labels. Defaults to None.
            y_pred (np.array, optional): Predicted labels. Defaults to None.
            n_samples (int, optional): How many random samples (rows) from X_test to obtain for manual inspection 
                                       of predicted and true labels together in a dataframe. Defaults to None.
            report_dir (str, optional): Absolute path to directory to save reports to. Defaults to None.

        Returns:
            results (dict): Dictionary of DataFrames with the following keys:
                           - scores: model performance scores, 
                           - classification_report: classification report, 
                           - confusion_matrix: confusion matrix, 
                           - roc_curve: roc curve values in a dataframe, 
                           - pr_curve: precision-recall curve values in a dataframe, 
                           - predicted_samples: predicted labels, true labels with features
        """
        if estimator is None:
            estimator = self.trained_estimator

        if X_test is None:
            X_test = self.train_test_data.get('X_test')

        if y_test is None:
             y_test = self.train_test_data.get('y_test')

        if y_pred is None:
            y_pred = estimator.predict(X_test)
        
        if n_samples is None:
            n_samples = self.n_samples

        if is_classifier(estimator):

            if proba is None:
                proba = self.predict_proba(estimator, X_test)

            scores_df = self.get_classification_scores(
                estimator, 
                y_test, 
                y_pred,
                proba
            )

            # roc curve, pr curves only for binary classification for now
            if len(np.unique(y_test)) == 2:
                roc_curve_df = self.get_roc_curve(y_test, proba, estimator)
                pr_curve_df = self.get_pr_curve(y_test, proba, estimator)

            else: 
                roc_curve_df = None
                pr_curve_df = None

            if self.fitted_label_encoder:
                y_test = self.fitted_label_encoder.inverse_transform(y_test)
                y_pred =self.fitted_label_encoder.inverse_transform(y_pred)
            
            cls_df = self.get_cls_report(y_test, y_pred)
            c_matrix_df = self.get_confusion_matrix(y_test, y_pred)

        elif is_regressor(estimator):

            scores_df = self.get_regression_scores(estimator, y_test, y_pred)
            cls_df = None
            c_matrix_df = None
            roc_curve_df = None
            pr_curve_df = None

        # Create sample DataFrame of predicted scores and ground truth
        if self.fitted_scaler:
            X_test = self.fitted_scaler.inverse_transform(X_test)

        sample_df = self.get_pred_samples(
            X_test, 
            y_test, 
            y_pred, 
            n_samples
        )

        results = {
            'scores': scores_df,
            'classification_report': cls_df,
            'confusion_matrix': c_matrix_df,
            'roc_curve': roc_curve_df,
            'pr_curve': pr_curve_df,
            'predicted_samples': sample_df
        }

        # save all reports to csv file (optional)
        if report_dir:

            scores_df.to_csv(
                f'{report_dir}/{self.model_type}_{str(date.today())}_scores.csv'
                )
            cls_df.to_csv(
                f'{report_dir}/{self.model_type}_{str(date.today())}_classification_report.csv'
                )
            c_matrix_df.to_csv(
                f'{report_dir}/{self.model_type}_{str(date.today())}_confusion_matrix.csv'
                )
            sample_df.to_csv(
                f'{report_dir}/{self.model_type}_{str(date.today())}_predicted_samples.csv'
                )

        return results
    
    def save_model(self, path:str = None):
        """Saves a dictionary of the trained estimator, fitted scaler, fitted one hot encoder,
        fitted label encoder and list of training features in pickle format. 
        Dictionary keys: 
        - 'trained_estimator': trained estimator from training pipeline,
        - 'fitted_scaler': fitted scaler used to scale features in training pipeline,
        - 'onehot_encoder': fitted onehot encoder used to encode non-numeric features in training pipeline,
        - 'label_encoder': fitted label encoder used to encode non-numeric labels in training pipeline,
        - 'train_features': list of feature used in training pipeline

        Args:
            path (str, optional): Absolute file path for model to be saved in pkl format. 
                                  path should end with '.pkl'. Defaults to None.
        """
        if path:
            model_path = Path(path)
        
        else:
            model_filename = self.model_type + '_' + str(date.today()) + '.pkl'
            model_dir = Path.cwd() / self.model_dir / self.model_type
            model_dir.mkdir(parents=False, exist_ok=True) # create folder for model
            model_path = model_dir/ model_filename

        model_dict = {
            'trained_estimator': self.trained_estimator,
            'fitted_scaler': self.fitted_scaler,
            'onehot_encoder': self.fitted_onehot_encoder,
            'label_encoder': self.fitted_label_encoder,
            'train_features': self.train_features
        } 
        pickle.dump(model_dict, open(model_path, 'wb'))

        logger.info(
            f"Trained estimator {self.model_type} saved to {model_path}"
            )
    
    def load_model(self, path = None):
        """Loads a dictionary of trained estimator, fitted scaler, fitted encoders,
        and training features from a .pkl file. 
        Sets the values of the dictionary keys as the following class attributes:
        - trained_estimator: trained estimator from sklearn framework.
        - fitted_scaler: scaler used to scale features in the training pipeline.
        - fitted_onehot_encoder : one hot encoder used to encode non-numeric features in the training pipeline. 
        - fitted_label_encoder: label encoder used to encode non-numeric labels in the training pipeline.
        - train_feautures: list of feauture column names used in the training pipeline.

        Args:
            path (str, optional): Absolute file path of .pkl file. Defaults to None.

        Returns:
            model_dict[dict]: Dictionary of the loaded pkl file with the following keys:
                              - trained_estimator,
                              - fitted_scaler,
                              - onehot_encoder,
                              - label_encoder,
                              - train_features
        """
        if path is None:
            path = self.load_path

        model_dict = pickle.load(open(path, 'rb'))

        self.trained_estimator = model_dict.get('trained_estimator')
        self.fitted_scaler = model_dict.get('fitted_scaler')
        self.fitted_onehot_encoder = model_dict.get('onehot_encoder')
        self.fitted_label_encoder = model_dict.get('label_encoder')
        self.train_features = model_dict.get('train_features')

        return model_dict

    def make_inference(self, data: pd.DataFrame) -> np.array:
        """Inference pipeline 

        Args:
            data (pd.DataFrame): cleaned/preprocessed dataframe for prediction

        Returns:
            y_pred(np.array): array of predicted labels
            X_vals(np.array): scaled/tranformed features used for predictions (for evaluation purposes)
        """

        X_vals = self.get_X_vals(data)

        if self.trained_estimator is None:
            
            try:
                _ = self.load_model()

            except TypeError as e:
                logger.error(
                    f"Unable to make inference without a trained model. File path to .pkl file is not specified: {e} \
                        Either use train() or instantiate SupervisedModel class with file path to a trained model."
                )
                return 'ERROR', 'ERROR'

        if self.fitted_onehot_encoder:
            X_vals = self.fitted_onehot_encoder.transform(X_vals)
            self.inference_features = X_vals.columns.tolist()

        else:
            self.inference_features = X_vals.columns.tolist()
            X_vals = X_vals.values

        if self.fitted_scaler:
            X_vals = self.fitted_scaler.transform(X_vals)

        y_pred = self.trained_estimator.predict(X_vals)

        return y_pred, X_vals