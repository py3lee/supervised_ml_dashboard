import logging
import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype
from pathlib import Path

logger = logging.getLogger(__name__)

class CustomPreprocessor():
    def __init__(
        self, 
        df: pd.DataFrame, 
        cfg: object
    ):
        """CustomPreprocessor customizes checks and cleans VAERS data from merged DataFrame.
        Generates clean dataframe for VAERS dataset for modelling and EDA visualization purposes. Includes:

        - custom preprocessing
        - feature engineering 
        
        Attributes:

            df (pd.DataFrame): raw input dataframe
            cfg (module/object): configuration file specific to the dataset. Includes:
                
                - DATA_DICT (dict): data types for each column
                - VISUALIZE_DATA_DICT (dict) : data types to convert to for visualization purposes
                - DROP_COLS (list): columns to drop to obtain  modelling dataset
                - CHECK_ERROR (dict): columns to check for errors
                - IMPUTE_NA (dict): column to fill NaN with a specified value
                - DROP_NA (list): list of columns of which rows with NA will be dropped.
                - DATE_DIFF (dict): dictionary with new column name as key, list as value to calculate difference 
                  in days. Difference in days calculated using list element 0 - list element 1.
                - CLIP_UPPER (dict): Dictionary of key-value pairs with column names as keys and 
                  upper clip threshold limit as values 
                - REPLACE_VALUES (dict): Dictionary of key-value pairs with column names as keys and 
                  dictionary of replacement values for the column.
                - REPLACE_NONE_SYNONYMS (dict):  Dictionary of key-value pairs with feature names
                  as keys and list of none synonym strings to match as values
                - HISTORY_SEARCH (dict): Dictionary of key-value pairs with new column names
                  as keys and list of substrings to match in HISTORY column as values 
                - SYMP_ERROR (list):  list of substrings to search for in SYMP. A match (True) indicates that the error 
                  term is present in SYMP. The substrings stated in symp_error are medical or administrative errors 
                  which should not have occurred in the first place or no adverse event and are therefore excluded 
                  from modelling.
                - SYMP_SEARCH (dict): Dictionary of key-value pairs with new column names
                  as keys and list of substrings to match in SYMP column as values 
                - TARGET_COMPOSITE (list): list of binary features which indicate 
                  type of serious adverse side effect experienced
                - FILEPATH (dict): details on relative input and output filepaths.
                - output_filepath (str): relative file path to folder to save processed DataFrame as csv
        """
        self.df = df
        
        self.check_error = getattr(cfg, "CHECK_ERROR", None)
        self.data_dict = getattr(cfg, "DATA_DICT", None)
        self.visualize_data_dict = getattr(cfg, "VISUALIZE_DATA_DICT", None)
        self.drop_cols = getattr(cfg, "DROP_COLS", None)
        self.drop_na = getattr(cfg, "DROP_NA", None)
        self.impute_na = getattr(cfg, "IMPUTE_NA", None)
        self.date_diff = getattr(cfg, "DATE_DIFF", None)
        self.clip_upper = getattr(cfg, "CLIP_UPPER", None) 
        self.replace_values = getattr(cfg, "REPLACE_VALUES", None)
        self.replace_none_synonyms = getattr(
            cfg, 
            "REPLACE_NONE_SYNONYMS", 
            None
        )
        self.history_search = getattr(cfg, "HISTORY_SEARCH", None)
        self.symp_error = getattr(cfg, "SYMP_ERROR", None)
        self.symp_search = getattr(cfg, "SYMP_SEARCH", None) 
        self.target_composite_list = getattr(cfg, "TARGET_COMPOSITE", None)
        self.filepath = getattr(cfg, "FILEPATH", None)
        self.clean_filepath = self.filepath.get("clean_filepath")
        self.model_filepath = self.filepath.get("model_filepath")
    
    ###########
    # Helpers
    ###########

    def get_bool_error_rows(
        self,
        df: pd.DataFrame,
        col: str,
        criteria: list
    ) -> pd.Series:
        """Get a boolean series indicating whether the row value of the column (col)
        is less than the sane value as stated in the criteria list.

        Args:
            df (pd.DataFrame): main DataFrame
            col (str): column name to check for erroneous rows
            criteria (list): list with the first (0) element indicating the condition to
            check for (either "<" or ">"), and second element (1) indicating the sane value.

        Returns:
            pd.Series: boolean series indicating whether the row value of the column (col)
            is less than the sane value stated in the criteria list
        """
        condition = criteria[0]
        sane_value = criteria[1]

        if condition == "<":
            row_condition = df[col] < sane_value

        elif condition == ">":
            row_condition = df[col] > sane_value

        return row_condition

    def date_diff_days(
        self, 
        df: pd.DataFrame,
        new_col: str,
        date_col1: str,
        date_col2: str
    ) -> pd.DataFrame:
        """Creates a new column calculated from the difference in days between 2 date columns 
        (date_col1 - date_col2)

        Args:
            df (pd.DataFrame): pandas DataFrame    
            new_col (str): name of new column to create    
            date_col1 (str): name of the first date column to calculate the date difference from     
            date_col2 (str): name of date column to subtract date_col1 with

        Returns:
            df (pd.DataFrame): pandas DataFrame with new column calculated from 
            the difference in days between 2 date columns (date_col1 - date_col2)
        
        """
        df[new_col] = (df[date_col1] - df[date_col2]).dt.days.astype(int)

        return df

    def col_matched_substring(
        self, 
        df: pd.DataFrame,
        col: str,
        searchfor:list,
        new_col: str
    ) -> pd.DataFrame:
        """Creates a new binary column in the dataframe based on matches of a specified substring
        
            - If the row of the column contains any of the substring, value will be 1
            - If the row of the column does not contain any of the substring, value will be 0 

        Args:
            df (pd.DataFrame): pandas DataFrame    
            col (str): name of column to match substrings    
            searchfor (list): list of strings or substrings to search for    
            new_col (str): name of the new column to create in the dataframe
        
        Returns:
            df (pd.DataFrame): pandas dataframe with newly created columns
        
        """
        df_mask = df[col].str.contains('|'.join(searchfor), na=False)

        df.loc[df_mask == True, new_col] = 1
        df.loc[df_mask == False, new_col] = 0

        return df
    
    def replace_less_than(
        self,
        df: pd.DataFrame, 
        col:str, 
        ref_value: str,
        replace_col: str
    ) -> pd.DataFrame:
        """Replace values in a column of a dataframe that are less than the 
        reference value (col_values < ref value)
        
            - corresponding values from replace_col will be used as replacement value

        Args:
            df (pd.DataFrame): pandas DataFrame     
            col (str): column name    
            ref_value (str or pd.Series): reference value to check against.     
            replace_col (str): Column name to replace corresponding values with.
            
        Returns:
            df (pd.DataFrame): pandas DataFrame with column values replaced 
            with another replace_col value if rows are less than the reference 
            value (col_values < ref value)
        
        """
        condition = df[col] < ref_value
        
        logger.info(
            f"Replacing {condition.sum()} rows in {col} with values from {replace_col}"
        )

        df[col] = np.where(condition, df[replace_col], df[col])
        
        return df

    #################
    # Core Functions
    #################

    def drop_error_rows(
        self, 
        df: pd.DataFrame, 
        check_error: dict
    ) -> pd.DataFrame:
        """Drops erroneous rows for a stated column, condition and sane value as specified 
        in the config variable CHECK_ERROR.

        Args:
            df (pd.DataFrame): main DataFrame
            check_error (dict): dictionary with the following key-value pairs:
              - key: column name to check erroneous row values
              - value: list with the first (0) element indicating the reference condition to
                check (either "<" or ">"), and second element (1) indicating the sane value to check against.

        Returns:
            df (pd.DataFrame): main DataFrame with erroneous rows dropped for selected columns
        """
        for key, value in check_error.items():

            row_condition = self.get_bool_error_rows(
                df = df,
                col = key,
                criteria = value
            )
            total_drop = row_condition.sum()
            percent_drop = round(total_drop/len(df)*100, 2)
            logger.info(f"Dropping {total_drop} ({percent_drop}%) for {key} {value}")

            df = df.loc[~row_condition, :]
        
        return df

    def drop_error_symp(
        self,
        df: pd.DataFrame,
        symp_error: list
    ) -> pd.DataFrame:
        """Drops rows if SYMP values match any of the substrings stated in the symp_error list specified in the config.

        Args:
            df (pd.DataFrame): main DataFrame
            symp_error (list): list of substrings to search for in SYMP. A match (True) indicates that the error term is
            present in SYMP. The substrings stated in symp_error are medical or administrative errors which should
            not have occurred in the first place or no adverse event and are therefore excluded from modelling. 
            (example: Product administered to patient of inappropriate age)

        Returns:
            df (pd.DataFrame): main DataFrame with erroneous rows dropped for SYMP column
        """
        symp_error = [string.lower() for string in symp_error]
        error_mask = df['SYMP'].str.contains('|'.join(symp_error), na=False)

        logger.info(
            f"Dropping {error_mask.sum()} ({round(error_mask.sum()/len(df)*100, 2)}%) error rows in SYMP"
        )

        df = df.loc[error_mask == False, :]

        return df
    
    def impute_missing(
        self, 
        df: pd.DataFrame, 
        impute_na: dict
    ) -> pd.DataFrame:
        """Impute missing values with a specified value as stated in the configuration file. 

        Args:
            df (pd.DataFrame): main DataFrame
            impute_na (dict): dictionary with the following key-value pairs:
            - key: column name.
            - value: value to fill NaN values with.

        Returns:
            df (pd.DataFrame): main DataFrame with missing values for specified columns filled
        """
        for key, value in impute_na.items():
            
            total_na = df[key].isnull().sum()
            percent_na = round(total_na/len(df)*100, 2)
            logger.info(
                f"{total_na} ({percent_na}%) in {key} filled with {value}"
            )

            df[key] = df[key].fillna(value)
        
        return df
    
    def drop_missing(
        self, 
        df: pd.DataFrame,
        drop_na: list
    ) -> pd.DataFrame:
        """Drop rows with NA values for the columns specified in the configuration file. 

        Args:
            df (pd.DataFrame): main DataFrame
            drop_na (list): list of columns of which rows with NA will be dropped.

        Returns:
            df (pd.DataFrame): main DataFrame with NA values for specified columns dropped. 
        """
        for col in drop_na:
            total_na = df[col].isnull().sum()
            percent_na = round(total_na/len(df)*100,2)
            logger.info(
                f"Dropping {total_na} ({percent_na}%) in {col}"
            )
        df = df.dropna(subset=drop_na).reset_index(drop=True) 

        return df

    def to_lowercase(
        self, 
        df: pd.DataFrame, 
        data_dict: dict
    ) -> pd.DataFrame:
        """Lowercase string values for string columns as stated in the data dictionary variable in the
        configuration file.

        Args:
            df (pd.DataFrame): main DataFrame
            data_dict (dict): dictionary with the following key-value pairs:
            - key: column name
            - value: data type for the column

        Returns:
            df (pd.DataFrame): main DataFrame with string columns converted to lowercase
        """
        for key, value in data_dict.items():

            if 'string' in value:
                df[key] = df[key].astype('string')
                df[key] = df[key].str.lower()
        
        return df

    def cal_numdays(
        self, 
        df: pd.DataFrame, 
        date_diff: dict
    ) -> pd.DataFrame:
        """Recalculates NUMDAYS based on VAX_DATE and ONSET DATE
        
            - original NUMDAYS variable was completely erroneous due to wrong VAX_DATE and ONSET_DATES
              (i.e. some dates were in 1920)
        
        - Step 1: replace erroneous ONSET_DATE < VAX_DATE with VAX_DATE
        - Step 2: calculate difference in days betweeen both date columns

        Args:
            df (pd.DataFrame): pandas DataFrame     
            date_diff (dict): dictionary with new column name as key,
                list as value. Difference in days calculated 
                using list element 0 - list element 1
        
        Returns:
            df (pd.DataFrame): df with NUMDAYS_CAL variable
        
        """
        df = self.replace_less_than(
            df = df,
            col = 'ONSET_DATE',
            ref_value = df['VAX_DATE'],
            replace_col = 'VAX_DATE'
        )
        
        # recalculate erroneous NUMDAYS as new column NUMDAYS_CAL
        for key, value in date_diff.items():

            df = self.date_diff_days(
                df = df,
                new_col = key,
                date_col1 =  value[0],
                date_col2 = value[1]
            )

        return df

    def replace_string_values(
        self, 
        df: pd.DataFrame, 
        replace_values: dict
    ) -> pd.DataFrame:
        """Replaces string values in columns based on specified config variables 
        
            - key: column name 
            - value: dictionary to map strings to replacement values 

        Args:
            df (pd.DataFrame): pandas DataFrame    
            replace_values (dict): Dictionary of key-value pairs with column names
                as keys and dictionary of replacement values for the column 
        
        Returns:
            df (pd.DataFrame): pandas DataFrame with specified string values replaced.
        
        """
        for key, value in replace_values.items():

             df[key] = df[key].replace(
                 to_replace = value, 
                 value = None, 
                 inplace = False, 
                 limit = None,
                 regex = False,
                 method = None
            )
        
        return df

    def clip_upper_limit(
        self, 
        df: pd.DataFrame, 
        clip_upper: dict
    ) -> pd.DataFrame:
        """Clips the upper limit of columns based on specified config variables 
        
            - key: column name 
            - value: upper clipping threshold 

        Args:
            df (pd.DataFrame): pandas DataFrame    
            clip_upper (dict): Dictionary of key-value pairs with column names
                as keys and upper clip threshold limit as values 

        Returns:
            df (pd.DataFrame): pandas DataFrame with specified columns clipped
        
        """
        for key, value in clip_upper.items():
            
            df[key] = df[key].clip(upper = value)
        
        return df

    def create_feature_history(
        self, 
        df: pd.DataFrame, 
        history_search: dict
    ) -> pd.DataFrame:
        """Create new binary features from HISTORY column based on substring match from a config 
        
            - 1 = match any substring, 0 = no matched substrings
            - key: new binary feature name 
            - value: list of substrings to match 

        Args:
            df (pd.DataFrame): pandas DataFrame    
            history_search (dict): Dictionary of key-value pairs with new column names
                as keys, values are a list of substrings to match in HISTORY column
        
        Returns:
            df (pd.DataFrame): pandas DataFrame with new features as additional columns
        
        """
        for key, value in history_search.items():
            
            df = self.col_matched_substring(
                df = df, 
                col = 'HISTORY', 
                searchfor = value, 
                new_col = key
            )

        return df  

    def create_feature_symp(
        self, 
        df: pd.DataFrame, 
        symp_search: dict
    ) -> pd.DataFrame:
        """Create new binary features from SYMP column based on substring match from a config 
        
            - 1 = match any substring, 0 = no matched substrings
            - key: new binary feature name 
            - value: list of substrings to match 

        Args:
            df (pd.DataFrame): pandas DataFrame    
            symp_search (dict): Dictionary of key-value pairs with new column names
                as keys, and values are list of substrings to match in SYMP column
        
        Returns:
            df (pd.DataFrame): pandas DataFrame with new features as additional columns
        
        """
        for key, value in symp_search.items():
            
            df = self.col_matched_substring(
                df = df,
                col = 'SYMP',
                searchfor = value,
                new_col = key
            )

        return df 

    def create_feature_dum(
        self, 
        df: pd.DataFrame, 
        replace_none_synonyms: dict
    ) -> pd.DataFrame:
        """Create new binary features from unstructured text features 
        
            - 1 = at least 1 medical condition (i.e. medical history or allergies), 
            - 0 = no medical condition.
            - key: name of feature with unstructured text 
            - value: list of none synonyms (and variations which represent no medical condition) in the column
        
        - Step 1: Replace none synonyms with np.nan
        - Step 2: Replace np.nan with 0 (includes existing NaN), and replace all character strings with 1
        
        Args:
            df (pd.DataFrame): pandas DataFrame    
            replace_none_synonyms (dict): Dictionary of key-value pairs with feature names
                as keys and list of none synonym strings to match as values 

        Returns:
            df (pd.DataFrame): pandas DataFrame with new features as additional columns
        
        """
        for key, value in replace_none_synonyms.items():

            # Replace none synonyms with np.nan for HISTORY and ALLERGIES 
            df[key] = df[key].replace(
                to_replace = value, 
                value = np.nan, 
                inplace = False, 
                limit = None,
                regex = False,
                method = None
            )

            new_col = key + '_DUM'

            # Replace NaN as 0, all other character strings as 1
            df[new_col] = df[key].replace(
                to_replace = (r'.*', np.nan), 
                value = (1, 0), 
                inplace = False,
                limit = None,
                regex = True,
                method = None
            )

        return df   

    def create_feature_nur_home(
        self, 
        df: pd.DataFrame
    ) -> pd.DataFrame:
        """Create new binary feature NUR_HOME via 2 steps:
       
            - Replace 'SEN' category in V_ADMINBY to 1,
            - Replace all other categories (all other word characters) with 0

        Args:
            df (pd.DataFrame): pandas DataFrame

        Returns:
            df (pd.DataFrame): pandas DataFrame with new feature NUR_HOME created
        
        """
        df['NUR_HOME'] = df['V_ADMINBY'].\
            replace('SEN', 1).\
                replace(
                    r'\W*', 
                    0, 
                    regex = True
                )

        return df
    
    def create_target_composite(
        self, 
        df: pd.DataFrame, 
        target_composite_list: list
    ) -> pd.DataFrame:
        """Create the binary composite target column TARGET_SERIOUS_ADVERSE 
        from on a list of features in the config file
        
            - 1: At least 1 serious adverse side effect experienced,
            - 0: no serious adverse side effect experienced

        Args:
            df (pd.DataFrame): pandas DataFrame    
            target_composite_list (list): list of binary features which indicate 
                type of serious adverse side effect experienced

        Returns:
            df (pd.DataFrame): pandas DataFrame with composite TARGET_SERIOUS_ADVERSE
        
        """
        df['composite'] = 0 # dummy numeric series for addition in the for loop
        
        for col in target_composite_list:

            if not is_numeric_dtype(df[col]):

                replace_str = {'Y': 1, 'N': 0}
                df[col] = df[col].replace(replace_str)
            
            df['composite'] += df[col]
        
        df['TARGET_SERIOUS_ADVERSE'] = np.where(
            df['composite'] == 0,
            'No', 
            'Yes'
        )
        
        return df

    def run(self):
        """Runs all required methods to clean the merged dataset and engineer features required for visualization
        and modelling phase.

        Returns: 
            df_model (pd.DataFrame): processed pandas DataFrame for modelling 
            df_visualize (pd.DataFrame): processed pandas DataFrame for further EDA analyses
        
        """
        df = self.to_lowercase(df = self.df, data_dict = self.data_dict)

        df = self.drop_error_rows(df, check_error = self.check_error)
        df = self.drop_error_symp(df, symp_error = self.symp_error)

        df = self.drop_missing(df, drop_na = self.drop_na)
        df = self.impute_missing(df, self.impute_na)

        df = self.cal_numdays(df, date_diff = self.date_diff)

        # feature engineering
        df = self.create_feature_dum(
            df, 
            replace_none_synonyms = self.replace_none_synonyms
        )
        
        df = self.create_feature_history(
            df, 
            history_search = self.history_search
        )   

        df = self.create_feature_symp(
            df, 
            symp_search = self.symp_search
        )

        df = self.create_feature_nur_home(df)

        df = self.create_target_composite(
            df, 
            target_composite_list = self.target_composite_list
        )

        # csv for visualization and further EDA
        df_visualize = df
        full_clean_filepath = Path(__file__).parents[2] / self.clean_filepath
        df_visualize.to_csv(
            full_clean_filepath, 
            index=False
        )
        logger.info(f"Visualization csv saved to: {full_clean_filepath}")

        # further modelling prep
        df = self.replace_string_values(
            df, 
            replace_values = self.replace_values
        )
        df = self.clip_upper_limit(df, clip_upper = self.clip_upper)

        # csv for modelling
        df_model = df.drop(columns = self.drop_cols)
        full_model_filepath = Path(__file__).parents[2] / self.model_filepath
        df_model.to_csv(
            full_model_filepath, 
            index=False
        )
        logger.info(f"Modelling csv saved to: {full_model_filepath}")
 
        return df_visualize, df_model