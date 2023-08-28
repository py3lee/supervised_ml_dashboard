##########
# Built in
##########
import sys

########
# Libs #
########
from IPython.core.display import display, HTML
import ipywidgets as widgets
import pandas as pd
from sklearn.base import is_classifier

#########################
# Custom - user to import 
#########################
from .dashboard_helpers import (
    display_model_params,
    plot_cmatrix,
    plot_roc,
    plot_pr,
    get_shap_values,
    plot_shap_summary,
    get_shap_feature_importance,
    get_feature_importance,
    plot_feature_importance,
    plot_feat_vs_target,
    widget_option_unique_values,
    get_all_feature_names,
    has_feature_importances_attr
)

sys.path.append('../')
from src.models.supervised_model_class import SupervisedModel

class Dashboard:
    def __init__(
        self, 
        df: pd.DataFrame, 
        target: str,
        n_samples: int = 100,
        cfg: object = None
    ):
        """Creates a Machine Learning Supervised model dashboard that displays the following:
       
        - For classifiers:
            - model performance scores, 
            - Receiver operating characteristic curve,
            - Precision-Recall curve,   
            - Classification Report, 
            - Confusion Matrix, 
            - Model hyperparameters, 
            - Sample of predicted samples with features and true labels,
            - Feature Importance from Tree based models, 
            - SHAP summary plot, 
            - Bivariate plots of target vs features for top N features

        - For regressors:
            - model performance scores, 
            - Model hyperparameters, 
            - Sample of predicted samples with features and true labels,
            - Feature Importance from Tree based models, 
            - SHAP summary plot, 
            - Bivariate plots of target vs features for top N features

        Args:

            df (pd.DataFrame): Dataset of features and labels 

            target (str): Name of target column in df 

            n_samples (int, optional): How many randomly sampled rows in a DataFrame to manually inspect predictions 
            with true labels and features. Defaults to 100.

            cfg (module/object): config file containing required variables for training of the model 
            using the helper SupervisedModel class. Defaults to None.
        """

        self.df = df
        self.target = target
        self.n_samples = n_samples
        self.cfg = cfg

        # variables to be shared among widgets after a main event
        self.estimator = None 
        self.sample_df = None
        self.feature_importance = None
        self.shap_feature_importance = None
        self.n_model_scores = []
        self.n_roc_curve = []
        self.n_pr_curve = []
        
        #################
        # OUTPUT WIDGETS
        #################

        # Performance - All tab
        self.curve_output = widgets.Output()
        self.scores_output = widgets.Output()

        # Selected model tab
        self.model_param_output = widgets.Output()
        self.cls_report_output = widgets.Output()
        self.c_matrix_output = widgets.Output()
        self.model_roc_output = widgets.Output()
        self.model_pr_output = widgets.Output()

        # Predicted samples tab for manual inspection
        self.samples_output = widgets.Output()
        self.samples_tab_widgets_output = widgets.Output()

        # Feature importance tab 
        self.top_n_feat_widget_output = widgets.Output()
        self.feat_plot_output = widgets.Output()
        self.shap_plot_output = widgets.Output()

        # Bivariate plots of target vs features tab 
        self.feat_bivariate_output = widgets.Output()

        ################################
        # WIDGETS FOR USER INTERACTION
        ################################

        # Press enter to load and evaluate a trained model 
        style = {'description_width': 'initial'}
        self.load_model_path_widget = widgets.Text(
                                        value ='',
                                        placeholder ='press <enter> when done',
                                        description = '.pkl path:',
                                        disabled = False,
                                        style = style
                                ) 

        # Choose a model type to train and evaluate 
        self.model_type_widget = widgets.Dropdown(
            options = [
                'None',
                'RFC', 
                'DT', 
                'RFR', 
                'Auto-Classification',
                'Auto-Regression'
                ],
            value = 'None',
            description = 'Select model type',
            style = style
        )

        # Select X_test size (only for evaluating a trained model)
        self.X_test_size_widget = widgets.IntSlider(
            value = len(self.df),
            min = 1,
            max = len(self.df),
            description = 'Validation size',
            step = 10,
            disabled = False,
            style = style
        )

        #####################################    
        # BIND EVENT HANDLERS TO MAIN WIDGETS
        #####################################

        self.load_model_path_widget.on_submit(
            self.load_model_path_eventhandler
        )
        self.model_type_widget.observe(
            self.model_type_eventhandler, 
            names = 'value'
        )
        self.X_test_size_widget.observe(
            self.X_test_size_eventhandler, 
            names = 'value'
        )

        #############################
        # CONTAINERS FOR MAIN WIDGETS
        ##############################

        # layout for containers
        container_layout = widgets.Layout(
                display = 'flex',
                flex_flow = 'columns',
                align_items = 'center',
                width = '100%'
            )  
        item_layout = widgets.Layout(margin='1em 0 1em 0')

        # Main dashboard (above the tabs)
        self.loaded_model_widgets = widgets.VBox(
            [
                widgets.HTML(
                    '<h4 style="text-align:center;">Load a trained model</h4>'
                ),
            self.load_model_path_widget,
            self.X_test_size_widget
            ], 
            layout = container_layout
        )

        self.train_model_widgets = widgets.VBox(
            [
                widgets.HTML(
                    '<h4 style="text-align:center;">Train and evaluate a model</h4>'
                    ),
                self.model_type_widget
                ], 
            layout = container_layout
        )

        self.above_tabs = widgets.HBox(
            [ 
                self.loaded_model_widgets,
                self.train_model_widgets
            ]
        )

        # Performance - All' tab
        self.main_tab = widgets.VBox(
            [
                self.scores_output, 
                self.curve_output
                ],
            layout = item_layout
        )

        # Selected model tab
        self.selected_model_grid = widgets.GridspecLayout(
            3, 
            2, 
            layout = widgets.Layout(justify_content='center')
        )
        self.selected_model_grid[0, 0] = self.model_param_output
        self.selected_model_grid[1, 0] = self.cls_report_output
        self.selected_model_grid[1, 1] = self.c_matrix_output
        self.selected_model_grid[2, 0] = self.model_roc_output
        self.selected_model_grid[2, 1] = self.model_pr_output
                                                    
        # Predicted samples tab
        self.samples_tab = widgets.VBox(
            [
                self.samples_tab_widgets_output, 
                self.samples_output
                ]
        )

        # Feature importance tab
        self.feature_tab = widgets.VBox(
            [
                self.top_n_feat_widget_output, 
                self.feat_plot_output,
                self.shap_plot_output
                ], 
            layout = item_layout
        )

        # Set containers into tabs 
        self.tab = widgets.Tab(
            [
                self.main_tab,
                self.selected_model_grid,
                self.samples_tab,
                self.feature_tab,
                self.feat_bivariate_output
            ], 
            layout = item_layout
        )

        self.tab.set_title(0, 'Performance - All')
        self.tab.set_title(1, 'Selected Model')
        self.tab.set_title(2, 'Predicted samples')
        self.tab.set_title(3, 'Feature Importance')
        self.tab.set_title(4, 'Target vs Features')

        ###########################
        # Overall dashboard layout 
        ###########################

        title_html = """

        <h2 style="text-align:center;">Supervised Model Evaluation</h2>

        """
        description_html = """

        <style>
        p {
            margin-bottom: 1.2em;
        }
        </style>

        <p style="text-align:center;">Evaluate the performance of a machine learning model.<br>
        Select below to use a trained model, or train and evaluate a new model using the scikit-learn framework</p>
        """

        self.app_contents = [
            widgets.HTML(
                title_html, 
                layout = widgets.Layout(
                    margin ='0 0 1em 0', 
                    max_width = '1200px'
                )
            ),
            widgets.HTML(
                description_html, 
                layout = widgets.Layout(
                    margin ='0 0 0 0', 
                    max_width ='1100px'
                )
            ),
            self.above_tabs, 
            self.tab
        ]
                            
        self.app = widgets.VBox(
            self.app_contents, 
            layout = widgets.Layout(
                margin = '30px auto 30px auto', 
                max_width = '1024px'
            )
        )

        display(self.app)

    ##########
    # Helpers
    ##########

    def create_widgets_samples_tab(self, sample_df: pd.DataFrame):
        """Creates interactive widgets for the Predicted Samples tab.
        Widget values are based on the dataframe (samples_df) of predicted samples returned by SupervisedModel class.

        Args:

            sample_df (pd.DataFrame): DataFrame of features, labels and predicted labels
        """
        if self.samples_tab_widgets_output:
            self.samples_tab_widgets_output.clear_output()

        # Filter predicted samples_df by true labels
        self.true_label_widget = widgets.Dropdown(
            options = widget_option_unique_values(
                sample_df['true_label']
            ),
            description = 'True labels'
        )

        # Filter predicted samples_df by predicted labels
        self.pred_label_widget = widgets.Dropdown(
            options = widget_option_unique_values(
                sample_df['predicted_label']
            ),
            description = 'Predicted'
        )

        # Filter predicted samples df by features
        self.filter_sample_feat_widget = widgets.SelectMultiple(
            options = get_all_feature_names(
                sample_df
            ),
            value = get_all_feature_names(
                sample_df
            ),
            description ='Features',
            disabled = False,
            style = {'description_width': 'initial'}
        )

        # limit displayed rows in predicted samples df
        self.limit_sample_rows_widget = widgets.IntSlider(
            value = self.n_samples,
            min = 1,
            max = self.n_samples,
            description ='Sample size',
            step = 1,
            disabled = False,
            style = {'description_width': 'initial'}
        )

        # bind widgets to event handlers
        self.true_label_widget.observe(
            self.true_label_eventhandler, 
            names ='value'
        )
        self.pred_label_widget.observe(
            self.pred_label_eventhandler, 
            names = 'value'
        )
        self.filter_sample_feat_widget.observe(
            self.filter_sample_features_eventhandler, 
            names = 'value'
        )
        self.limit_sample_rows_widget.observe(
            self.limit_sample_rows_eventhandler, 
            names ='value'
        )

        # put samples widgets into one container
        self.samples_widgets = widgets.HBox(
            [
                self.limit_sample_rows_widget, 
                self.filter_sample_feat_widget,
                self.true_label_widget, 
                self.pred_label_widget
                ],
            layout = widgets.Layout(
                margin='1em 0 1em 0',
                max_width ='900px'
            )
        )

        display(self.samples_widgets)
    
    def create_widget_feat_importance_tab(
        self, 
        shap_feature_importance: pd.DataFrame
    ):
        """Creates interactive widget for the Feature Importance tab.
        Widget values are based on the number of features in shap_feature_importance dataframe

        Args:

            shap_feature_importance (pd.DataFrame): DataFrame of global feature importance scores based on SHAP values
        """
        if self.top_n_feat_widget_output:
            self.top_n_feat_widget_output.clear_output()

        # limit number of features to visualise in Features importance tab
        max_features = len(shap_feature_importance)

        self.limit_plot_feat_widget = widgets.IntSlider(
            value = max_features,
            min = 1,
            max = max_features,
            description = 'Top N features',
            step = 1,
            disabled = False,
            continuous_update = False,
            style = {'description_width': 'initial'}
        )

        # Bind widget to event handler
        self.limit_plot_feat_widget.observe(
            self.limit_plot_features_eventhandler, 
            names = 'value'
        )
        
        display(self.limit_plot_feat_widget)

    def filter_predicted_samples(
        self, 
        sample_df: pd.DataFrame, 
        row_limit: int, 
        true_choice: str, 
        pred_choice: str, 
        sample_features: tuple
    ):
        """Updates displayed sample_df in Predicted samples tab based on the following widget values:

        - limit_sample_rows_widget,
        - true_label_widget,
        - pred_label_widget,
        - filter_sample_feat_widget

         Args:

            sample_df (pd.DataFrame): DataFrame of predicted samples generated from SupervisedModel evaluate() method. 

            row_limit (int): number of rows in sample_df to display in dashboard tab

            true_choice (str): class in true labels column in sample_df to filter displayed DataFrame.

            pred_choice (str): class in predicted labels column in sample_df to filer displayed DataFrame.

            sample_features (tuple): tuple of feature names from filter_sample_feat_widget
        """
        self.samples_output.clear_output()
        
        # limit rows to display
        display_df = sample_df.iloc[0 :row_limit, :]

        sample_features = list(sample_features) # convert tuple to list
        features_and_labels = ['true_label','predicted_label'] + sample_features

        # filter features 
        display_df = display_df.loc[:, features_and_labels]

        if (true_choice == 'ALL') & (pred_choice == 'ALL'):
            display_df = display_df

        elif (true_choice == 'ALL'):
            display_df = display_df[
                display_df.predicted_label == pred_choice
            ]

        elif (pred_choice == 'ALL'):
            display_df = display_df[
                display_df.true_label == true_choice
            ]

        else:
            display_df = display_df[
                (display_df.true_label == true_choice) & 
                (display_df.predicted_label == pred_choice)
            ]

        with self.samples_output:
            display(
                HTML(
                    display_df.to_html(index=False)
                )
            )

    def update_features_plot(
        self, 
        plot_nfeatures: int
    ):
        """Function to update impurity-based feature importance plot based on value from limit_plot_feat_widget

        Args:

            plot_nfeatures (int): Top N features to plot, input from limit_plot_feat_widget
        """
        self.feat_plot_output.clear_output()
        with self.feat_plot_output:

            plot_feature_importance(
                estimator = self.estimator, 
                feature_importance = self.feature_importance, 
                plot_nfeatures = plot_nfeatures
            )

    def update_target_vs_feat_plot(self, plot_nfeatures: int):
        """Function to update bivariate plots of target and top N features based on input from limit_plot_feat_widget

        Args:

            plot_nfeatures (int): Top N features to plot, input from limit_plot_feat_widget
        """
        
        self.feat_bivariate_output.clear_output()
        with self.feat_bivariate_output:

            plot_feat_vs_target(
                df = self.df, 
                shap_feature_importance = self.shap_feature_importance, 
                plot_nfeatures = plot_nfeatures, 
                target = self.target,
                estimator = self.estimator
            )    

    def change_dataset_size(
        self,
        df: pd.DataFrame, 
        df_size: int,
        target: str
    ):
        """Obtain a random sample from the dataset. For analyses triggered by 'load a trained model' 
        section of the dashboard only. If the trained model is a classifier, a random stratified sample 
        based on true label proportion will be returned. 

        Args:

            df (pd.DataFrame): entire dataset used to instantiate Dashboard class

            df_size (int): sample size

            target (str): Column name of target variable in df.

        Returns:

            limit_df (pd.DataFrame): Dataframe of sampled rows
        """
        if is_classifier(self.estimator):

            # get randomly stratified sample based on true label proportions
            grouped = df.groupby(target, group_keys = False)
            sample_based_on_grp_proportion = grouped.apply(
                lambda x: x.sample(
                    int(
                        df_size*len(x)/len(df)
                    )
                )
            )
            limit_df = sample_based_on_grp_proportion.sample(frac=1) # shuffle
        
        else:
            limit_df = df.sample(df_size)

        return limit_df

    def display_n_model_scores(self):
        """Function to display the combined scores_df for each estimator after each main widget event 
        ('load a model' or 'train and evaluate a model' events). 
        
        - Triggered by load_model_path_widget or model_type_widgets
        """
        self.scores_output.clear_output()

        with self.scores_output:
            display_score = pd.concat(self.n_model_scores)
            display_score = display_score.round(3)
            
            display(
                    HTML('<h4 style="text-align:center;">Scores obtained on test set</h4>')
                )
            display(display_score)
    
    def display_combined_curves(self):
        """Function to display combined ROC and PR curves in the Performance-All tab after each main widget event
        ('load a model' or 'train and evaluate a model' events)
        
        - Triggered by load_model_path_widget or model_type_widgets"""

        # combine curve dataframes generated by previous model training events
        roc_curve_df = pd.concat(self.n_roc_curve).reset_index(drop=True)
        pr_curve_df = pd.concat(self.n_pr_curve).reset_index(drop=True) 
        
        self.curve_output.clear_output()
        
        with self.curve_output:
            display(
                HTML('<h4 style="text-align:center;">Receiver operating characteristic curve</h4>')
            )
            plot_roc(roc_curve_df)
            
            display(
                HTML('<h4 style="text-align:center;">Precision recall curve</h4>')
            )
            plot_pr(pr_curve_df)

    def clear_output(self):
        """Clears existing display from output widgets"""

        # Performance - All tab
        self.curve_output.clear_output()
        self.scores_output.clear_output()

        # Selected model tab
        self.model_param_output.clear_output()
        self.cls_report_output.clear_output()
        self.c_matrix_output.clear_output()
        self.model_roc_output.clear_output()
        self.model_pr_output.clear_output()

        # Predicted samples tab
        self.samples_output.clear_output()

        # Feature importance + target vs features tabs
        self.feat_plot_output.clear_output()
        self.shap_plot_output.clear_output()
        self.feat_bivariate_output.clear_output()

    #########################################################
    # Main widget display event 1: Train and evaluate a model 
    #########################################################

    def train_and_evaluate(
        self,
        model_type: str
    ):
        """Train and evaluate a model. Display performance matrices, output curves, predicted samples, 
        feature importance and bivariate plots. Event triggered by input from model_type_widget.

        Args:

            model_type (str): type of model to train. input from model_type_widget
        """
        self.clear_output()

        if model_type == 'None':
            print('No model selected')
            return

        if self.cfg is None:
            raise Exception(
                "Config file (cfg) is required for training a model."
            )

        model = SupervisedModel(
            model_type = model_type,
            df = self.df,
            target = self.target,
            cfg = self.cfg
        )
        
        ################################################
        # Get required variables for displaying output
        ################################################
            
        estimator, train_test_data = model.train()
        
        X_test = pd.DataFrame(
            data = train_test_data.get('X_test'), 
            columns = model.train_features
        )
        
        X_train = pd.DataFrame(
            data = train_test_data.get('X_train'), 
            columns = model.train_features
        )
    
        results = model.evaluate(n_samples = self.n_samples)
        
        scores_df = results.get('scores')
        cls_df = results.get('classification_report')
        c_matrix_df = results.get('confusion_matrix')
        roc_curve_df = results.get('roc_curve')
        pr_curve_df = results.get('pr_curve')
        sample_df = results.get('predicted_samples')
        
        if has_feature_importances_attr(estimator):
            feature_importance = get_feature_importance(
                estimator = estimator,
                train_features = model.train_features
            )
        
        shap_values, explainer = get_shap_values(
            estimator = estimator,
            X_train = X_train,
            X_test = X_test,
            n_samples = 100
        ) 

        shap_feature_importance = get_shap_feature_importance(
            estimator = estimator,
            feature_names = model.train_features, 
            shap_values = shap_values
        )
    
        #######################################################
        # Direct outputs to be displayed in the respective tabs
        #######################################################

        # Selected model tab
        with self.model_param_output:

            display(HTML(f"Training set size: {len(X_train)}"))
            display(HTML(f"Test set size: {len(X_test)}"))
            print("\n")
            display_model_params(estimator)

        if is_classifier(estimator):

            with self.c_matrix_output:
                plot_cmatrix(c_matrix_df)

            with self.cls_report_output:
                display(
                    HTML('<h4 style="text-align:center;">Classification Report</h4>')
                )
                display(cls_df)

            with self.model_roc_output:
                display(
                    HTML('<h4 style="text-align:center;">Receiver operating characteristic curve</h4>')
                )
                plot_roc(roc_curve_df)
            
            with self.model_pr_output:
                display(
                    HTML('<h4 style="text-align:center;">Precision recall curve</h4>')
                )
                plot_pr(pr_curve_df)
        
        # Predicted samples tab
        with self.samples_tab_widgets_output:
            self.create_widgets_samples_tab(sample_df)

        with self.samples_output:

            self.filter_predicted_samples(
                sample_df = sample_df, 
                row_limit = self.limit_sample_rows_widget.value, 
                true_choice = self.true_label_widget.value, 
                pred_choice = self.pred_label_widget.value, 
                sample_features =  self.filter_sample_feat_widget.value
            )

        # Feature Importance tab
        with self.top_n_feat_widget_output:
            self.create_widget_feat_importance_tab(shap_feature_importance)

        with self.feat_plot_output:

            if has_feature_importances_attr(estimator):

                plot_feature_importance(
                    estimator = estimator, 
                    feature_importance = feature_importance, 
                    plot_nfeatures = len(model.train_features)
                )

        with self.shap_plot_output:

            plot_shap_summary(
                estimator = estimator,
                shap_values = shap_values, 
                X_test = X_test,
                n_samples = 100
            )

        # Target vs features tab
        with self.feat_bivariate_output:

            plot_feat_vs_target(
                df = self.df, 
                shap_feature_importance = shap_feature_importance, 
                plot_nfeatures = self.limit_plot_feat_widget.value, 
                target = self.target,
                estimator = estimator
            )

        # Update variables (shared across widgets)
        self.estimator = estimator 
        self.sample_df = sample_df
        self.shap_feature_importance = shap_feature_importance

        if has_feature_importances_attr(estimator):
            self.feature_importance = feature_importance

        getattr(self, 'n_model_scores').append(scores_df)
        getattr(self, 'n_roc_curve').append(roc_curve_df)
        getattr(self, 'n_pr_curve').append(pr_curve_df)
    
    #################################################################
    # Main widget display event 2: Load and evaluate a trained model 
    #################################################################

    def load_and_evaluate(self, model_path: str):
        """Load and evaluate a trained model. Display performance matrices, output curves, predicted samples, 
        feature importance and bivariate plots.Event triggered by input from load_model_path_widget.

        Args:

            model_path (str): full file path to trained model pkl file. input from model_type_widget
        """
        self.clear_output()

        df = self.change_dataset_size(
            df = self.df, 
            df_size = self.X_test_size_widget.value,
            target = self.target
        )
        
        if model_path == '': 
            raise Exception(
                'No file path given to load and evaluate a trained model. \
                Please type in the absolute file path of the model to load for evaluation purposes'
            )

        # instantiate SupervisedModel class with a default model  
        model = SupervisedModel(
            model_type = 'DT',
            df = df,
            target = self.target,
            cfg = None,
            load_path = model_path
        )

        _ = model.load_model(model_path) # model_dict variable will be set as class attributes for model obj
        estimator = model.trained_estimator
        
        ################################################
        # Get required variables for displaying output
        ################################################

        y_true = df[self.target]

        if model.fitted_label_encoder:
            y_true = model.fitted_label_encoder.transform(y_true)

        y_pred, X_vals = model.make_inference(data = df)

        if not model.inference_features == model.train_features:
            raise Exception(
                "Current features in df and features used for training the model are not the same.\
                    Please check features in df and those used to train the model before proceeding further."
            )

        X_vals = pd.DataFrame(
            data = X_vals, 
            columns = model.inference_features
        )

        results = model.evaluate(
            X_test = X_vals,
            y_test = y_true,
            y_pred = y_pred,
            n_samples = self.n_samples
        )

        scores_df = results.get('scores')
        cls_df = results.get('classification_report')
        c_matrix_df = results.get('confusion_matrix')
        roc_curve_df = results.get('roc_curve')
        pr_curve_df = results.get('pr_curve')
        sample_df = results.get('predicted_samples')
        
        if has_feature_importances_attr(estimator):
            feature_importance = get_feature_importance(
                estimator = estimator,
                train_features = model.train_features
            )
        
        shap_values, explainer = get_shap_values(
            estimator = estimator,
            X_train = X_vals, # use X_vals to estimate expected values
            X_test = X_vals
        )

        shap_feature_importance = get_shap_feature_importance(
            estimator = estimator,
            feature_names = model.inference_features, 
            shap_values = shap_values
        )
    
        #######################################################
        # Direct outputs to be displayed in the respective tabs
        #######################################################

        # Selected model tab
        with self.model_param_output:
            display(HTML((f"X_test size: {len(X_vals)}")))
            print("\n")
            display(
                HTML('<h4 style="text-align:center;">Model parameters used</h4>')
            )
            display_model_params(estimator)

        if is_classifier(estimator):

            with self.c_matrix_output:
                plot_cmatrix(c_matrix_df)

            with self.cls_report_output:
                display(
                    HTML('<h4 style="text-align:center;">Classification Report</h4>')
                )
                display(cls_df)

            with self.model_roc_output:
                display(
                    HTML('<h4 style="text-align:center;">Receiver operating characteristic curve</h4>')
                )
                plot_roc(roc_curve_df)
            
            with self.model_pr_output:
                display(
                    HTML('<h4 style="text-align:center;">Precision recall curve</h4>')
                )
                plot_pr(pr_curve_df)
            
        # Predicted samples tab
        with self.samples_tab_widgets_output:
            self.create_widgets_samples_tab(sample_df)

        with self.samples_output:

            self.filter_predicted_samples(
                sample_df = sample_df, 
                row_limit = self.limit_sample_rows_widget.value, 
                true_choice = self.true_label_widget.value, 
                pred_choice = self.pred_label_widget.value, 
                sample_features =  self.filter_sample_feat_widget.value
            )

        # Feature Importance tab
        with self.top_n_feat_widget_output:
            self.create_widget_feat_importance_tab(shap_feature_importance)

        with self.feat_plot_output:

            if has_feature_importances_attr(estimator):

                plot_feature_importance(
                    estimator = estimator, 
                    feature_importance = feature_importance, 
                    plot_nfeatures = len(model.inference_features)
                )

        with self.shap_plot_output:

            plot_shap_summary(
                estimator = estimator,
                shap_values = shap_values, 
                X_test = X_vals
            )

        # Target vs features tab
        with self.feat_bivariate_output:

            plot_feat_vs_target(
                df = self.df, 
                shap_feature_importance = shap_feature_importance, 
                plot_nfeatures = self.limit_plot_feat_widget.value, 
                target = self.target,
                estimator = estimator
            )

        # Update variables (shared across widgets)
        self.estimator = estimator 
        self.sample_df = sample_df
        self.shap_feature_importance = shap_feature_importance

        if has_feature_importances_attr(estimator):
            self.feature_importance = feature_importance

        getattr(self, 'n_model_scores').append(scores_df)
        getattr(self, 'n_roc_curve').append(roc_curve_df)
        getattr(self, 'n_pr_curve').append(pr_curve_df)
      
    #########################
    # WIDGET EVENT HANDLING
    #########################
    
    def X_test_size_eventhandler(self, change):
        """Event handler when a change is noted in X_test_size_widget 

        Args:

            change (callable): change in widget state
        """
        self.change_dataset_size(
            df = self.df, 
            df_size = change.new,
            target = self.target
            )

    ##################################
    # Events in predicted samples tab
    ##################################

    def true_label_eventhandler(self, change):
        """Event handler when a change is noted in true_label_widget

        Args:

            change (callable): change in widget state
        """
        self.filter_predicted_samples(
            sample_df = self.sample_df, 
            row_limit = self.limit_sample_rows_widget.value, 
            true_choice = change.new, 
            pred_choice = self.pred_label_widget.value, 
            sample_features = self.filter_sample_feat_widget.value
        )
    
    def pred_label_eventhandler(self, change):
        """Event handler when a change is noted in pred_label_widget

        Args:

            change (callable): change in widget state
        """
        self.filter_predicted_samples(
            sample_df = self.sample_df, 
            row_limit = self.limit_sample_rows_widget.value,
            true_choice = self.true_label_widget.value, 
            pred_choice = change.new,
            sample_features = self.filter_sample_feat_widget.value
        ) 

    def filter_sample_features_eventhandler(self, change):
        """ Event handler when a change is noted in filter_sample_feat_widget

        Args:

            change (callable): change in widget state
        """
        self.filter_predicted_samples(
            sample_df = self.sample_df, 
            row_limit = self.limit_sample_rows_widget.value,
            true_choice = self.true_label_widget.value, 
            pred_choice = self.pred_label_widget.value,
            sample_features = change.new
        )

    def limit_sample_rows_eventhandler(self, change):
        """Event handler when a change is noted in limit_sample_rows_widget

        Args:

            change (callable): change in widget state
        """
        self.filter_predicted_samples(
            sample_df = self.sample_df, 
            row_limit = change.new,
            true_choice = self.true_label_widget.value, 
            pred_choice = self.pred_label_widget.value,
            sample_features = self.filter_sample_feat_widget.value
        )

    #########################################################
    # Events in features importance + target vs features tabs
    #########################################################

    def limit_plot_features_eventhandler(self, change):
        """Event handler when a change is noted in limit_plot_feat_widget (Top N features slider)

        Args:

            change (callable): change in widget state
        """
        if has_feature_importances_attr(self.estimator):
            self.update_features_plot(plot_nfeatures = change.new)
            
        self.update_target_vs_feat_plot(plot_nfeatures = change.new)

    #####################################################
    # Main dashboard event 1: Train and evaluate a model
    #####################################################

    def model_type_eventhandler(self, change):
        """Event handler when a change is noted in model_type_widget.

        Args:

            change (callable): change in widget state
        """
        self.train_and_evaluate(model_type = change.new)
        self.display_n_model_scores()

        if is_classifier(self.estimator):
            self.display_combined_curves()

    ###########################################################
    # Main dashboard event 2: Load and evaluate a trained model
    ###########################################################

    def load_model_path_eventhandler(self, change):
        """Event handler when a change is noted in load_model_path_widget.

        Args:
        
            change (callable): change in widget state
        """
        self.load_and_evaluate(model_path = change.value)
        self.display_n_model_scores()

        if is_classifier(self.estimator):
            self.display_combined_curves()
        