##########
# Built-in
##########
import glob
import logging
from pathlib import Path

########
# Libs #
########
import pandas as pd
from pandas.api.types import is_numeric_dtype

logger = logging.getLogger(__name__)

class CustomIngestor():
    def __init__(self, cfg):
        """Custom ingestor for VAERS dataset. 
        Takes in 3 raw csv files, converts them to 1 merged DataFrame for initial EDA and subsequent preprocessing.
        Gets the following variables from a configuration file, and sets them as class attrbutes: 

        Args:
            cfg (module/object): configuration file. Sets the following configuration variables as class attributes:
            
            - FILEPATH (dict): details on relative input and output filepaths.
            - folder_path (str): relative file path to folder containing the 3 raw csv files
            - merged_filepath (str): relative file path to save merged DataFrame as csv file
            - SYMP_COLS_DICT (dict): a dictionary with the following key-value pairs:
                - key: name of newly aggregated column
                - values: list of column names to use for aggregation
            - SYMP_CSV_COLS (list): list of column names to subset Symptoms DataFrame 
            - VAX_CSV_COLS (list): list of column names to subset Vaccine DataFrame
            - MAX_DATE (str): upper limit of the date range for RECVDATE. Used to filter the dataset. 
              Only rows with RECVDATE less than max_date will remain
        """

        # attributes from config
        self.filepath = getattr(cfg, "FILEPATH", None)
        self.folder_path = self.filepath.get("folder_path")
        self.merged_filepath = self.filepath.get("merged_filepath")
        self.symp_cols_dict = getattr(cfg, "SYMP_COLS_DICT", None)
        self.symp_csv_cols = getattr(cfg, "SYMP_CSV_COLS", None)
        self.vax_csv_cols = getattr(cfg, "VAX_CSV_COLS", None)
        self.max_date = getattr(cfg, "MAX_DATE", None)

    ##########
    # Helpers
    ##########

    def get_df_list(self, folder_path: str) -> list:
        """Get a list of csv filepaths to load as DataFrames

        Args:
            folder_path (str): relative file path to raw data folder containing csv files

        Returns:
            df_list (list): a list of absolute filepaths to load as DataFrames
        """
        full_folder_path = Path(__file__).parents[2] / folder_path

        df_list = [
            file 
            for file in glob.glob(
                f'{full_folder_path}/*.csv'
                )
        ]
        return df_list

    def read_csvs(self, df_list: list) -> pd.DataFrame:
        """Read a list of csv filepaths as pandas DataFrames

        Args:
            df_list (list): a list of absolute filepaths to load as DataFrames

        Returns:
            main_df (pd.DataFrame): main raw DataFrame for VAERS dataset, from 2021VAERSDATA.csv
            symp_df (pd.DataFrame): raw DataFrame for SYMPTOMS csv file, from 2021VAERSSYMPTOMS.csv
            vax_df (pd.DataFrame): raw DataFrame for vaccine csv file, from 2021VAERSVAX.csv
        """
        for file_path in df_list:

            if 'DATA' in Path(file_path).name:
                main_df = pd.read_csv(
                    file_path,
                    na_values = [
                        "n/a", "na", "-", "<NA>", "Na", "None", "none"
                        ],
                    low_memory = False,
                    encoding = "ISO-8859-1"
                )

            elif 'SYMPTOMS' in Path(file_path).name:
                symp_df = pd.read_csv(file_path)
            
            elif 'VAX' in Path(file_path).name:
                vax_df = pd.read_csv(
                    file_path,
                    encoding = "ISO-8859-1", 
                    engine='python'
                )
            
            else:
                logger.info('Unknown file in directory path - please check')

        return main_df, symp_df, vax_df


    def object_to_str(self, df: pd.DataFrame) -> pd.DataFrame:
        """Convert object (mixed dtypes) column dtypes to string dtype. 
        Intermediate step in aggregating SYMPTOMS columns.

        Args:
            df (pd.DataFrame): raw DataFrame

        Returns:
            df (pd.DataFrame): DataFrame with object columns converted to string types
        """
        for col in df.columns.tolist():

            if not is_numeric_dtype(df[col]):
                df[col] = df[col].astype('string')
                
        return df

    def na_as_blank(self, df: pd.DataFrame) -> pd.DataFrame:
        """Fills NaN values in string columns with a blank string '' to allow joining of columns via string methods.
        This is because string methods will raise an error if values are NaN.
        Intermediate step in aggregating SYMPTOMS columns.

        Args:
            df (pd.DataFrame): DataFrame

        Returns:
             df (pd.DataFrame): DataFrame with NaN values replaced with blank strings''
        """
        for col in df.columns.tolist():
            
            if df[col].isnull().sum() > 0:
                df[col] = df[col].fillna(value='')
                
        return df
    
    def remove_commas(self, df: pd.DataFrame) -> pd.DataFrame:
        """Removes one or more strings with value ',,'.
        These ',,' values arise due to Na/empty cells after joining multiple SYMPTOM columns. 
        Intermediate step in aggregating SYMPTOMS columns. 

        Args:
            df (pd.DataFrame): DataFrame

        Returns:
            df (pd.DataFrame): DataFrame with one or more',,' strings removed.
        """
        df['SYMP'] = df['SYMP'].replace(
                to_replace = r',,+', 
                value = "",
                regex = True,
                method = None
            )
        return df

    def one_id_per_row(
        self, 
        df: pd.DataFrame, 
        column: str
    ) -> pd.DataFrame:
        """Aggregate multiple rows of the same VAERS_ID with different values in 'column' into 1 row by joining 
        multiple rows of 'column' string values into 1 cell per VAERS_ID

        Args:
            df (pd.DataFrame): DataFrame
            column (str) : name of column to aggregate multiple rows of string values into 1 row per VAERS_ID

        Returns:
            df (pd.DataFrame): DataFrame with one VAERS_ID per row
        """
        df = df.groupby(['VAERS_ID'])[column]\
             .apply(','.join)\
                 .reset_index()

        return df 

    def create_count_sympt(self, df: pd.DataFrame):
        """Creates a new feature COUNT_SYMPT in Symptoms DataFrame by counting the number of elements in SYMP

        Args:
            df (pd.DataFrame): Symptoms DataFrame

        Returns:
            df (pd.DataFrame): DataFrame with new column 'COUNT_SYMPT' indicating count of elements in SYMP column
        """
        split_symp = df['SYMP'].str.split(',')
        df['COUNT_SYMPT'] = split_symp.apply(len)

        return df

    ################
    # Core functions
    ################

    def aggregate_symps(
        self, 
        df: pd.DataFrame, 
        symp_cols_dict: dict
    ) -> pd.DataFrame:
        """Aggregates the 5 SYMPTOM columns into 1 column, 'SYMP'

        Args:
            df (pd.DataFrame): raw DataFrame
            symp_cols_dict (dict): a dictionary with the following key-value pairs:
            - key: name of newly aggregated column
            - values: list of column names to use for aggregation

        Returns:
            df (pd.DataFrame): DataFrame with an additional new column created from aggregated symptom columns
        """
        df = self.object_to_str(df)
        df = self.na_as_blank(df)

        for key, value in symp_cols_dict.items():
            df[key] = df[value].agg(','.join, axis=1)

        df = self.remove_commas(df)
        df = self.one_id_per_row(df, column = 'SYMP')

        logger.debug(
            f"Number of duplicates in symp_df: \
            {df.duplicated(subset =['VAERS_ID']).sum()}"
        )

        logger.debug(f"{df.shape}")
        logger.debug(f"{df.loc[0, 'SYMP']}")
        logger.debug(f"{df.loc[10, 'SYMP']}")

        return df

    def convert_dates(self, df: pd.DataFrame) -> pd.DataFrame:
        """Convert datetime columns to datatime format 

        Args:
            df (pd.DataFrame): main DataFrame

        Returns:
            df (pd.DataFrame): main DataFrame with datetime columns converted to datetime format
        """
        date_cols = [
            col
            for col in df.columns.tolist() 
            if 'DATE' in col
        ]
        for date_col in date_cols:
            df[date_col] = pd.to_datetime(df[date_col])
        
        return df

    def limit_max_date(
        self, 
        df: pd.DataFrame, 
        max_date: str
    ) -> pd.DataFrame:
        """Subset the main DataFrame by filtering rows less than the max_date specified in the config file. 

        Args:
            df (pd.DataFrame): main DataFrame
            max_date (str): upper limit of the date range for RECVDATE. Used to filter the dataset. 
            Only rows with RECVDATE less than max_date will remain

        Returns:
            df (pd.DataFrame): main DataFrame filtered according to the max_date set for RECVDATE
        """
        df = df.loc[
            df['RECVDATE'] < max_date, 
            :]
        logger.debug(f"Max date: {df['RECVDATE'].max()}")

        return df

    def get_pfizer_moderna(
        self, 
        df: pd.DataFrame, 
        vax_csv_cols: list
    ) -> pd.DataFrame:
        """Subset the DataFrame by:
        1. Filtering only VAX_MANU values containing the following vaccines:
        - MODERNA
        - PFIZER\BIONTECH
        2. Selecting only relevant columns (from config file) in vaccine dataframe to be merged with main DataFrame.

        Args:
            df (pd.DataFrame): Vaccine DataFrame, from 2021VAERSVAX.csv
            vax_csv_cols (list): Column names to be merged into the main DataFrame.

        Returns:
            df (pd.DataFrame): subsetted Vaccine DataFrame, to be merged into the main DataFrame
        """
        row_condition = df['VAX_MANU'].isin(
            ["MODERNA" , "PFIZER\BIONTECH"]
        )

        df = df.loc[
            row_condition,
            vax_csv_cols
        ]
        df = df.drop_duplicates(subset=['VAERS_ID']) # make VAERS_ID unique for each row.duplicate VAERS_ID rows due to non-COVID vaccines.
        logger.debug(df.VAX_MANU.value_counts())

        logger.debug(
            f"Number of duplicates in vax_df: \
            {df.duplicated(subset=['VAERS_ID']).sum()}"
        )

        return df

    def merge_dfs(
        self, 
        main_df: pd.DataFrame, 
        vax_df: pd.DataFrame, 
        symp_df: pd.DataFrame
    ):
        """Merge the 3 DataFrames (2021VAERSDATA.csv, 2021VAERSSYMPTOMS.csv, 2021VAERSVAX.csv) into 1 main DataFrame
        for EDA analyses and subsequent further preprocessing. 

        Args:
            main_df (pd.DataFrame): main DataFrame, from 2021VAERSDATA.csv
            vax_df (pd.DataFrame): Vaccine DataFrame, from 2021VAERSVAX.csv
            symp_df (pd.DataFrame): Symptoms DataFrame, from 2021VAERSSYMPTOMS.csv

        Returns:
            merged_df (pd.DataFrame): merged main dataframe
        """
        main_vax = pd.merge(
            main_df, 
            vax_df,
            how = 'inner',
            on = 'VAERS_ID'
        )

        merged_df = pd.merge(
            main_vax,
            symp_df,
            how = 'inner',
            on = 'VAERS_ID'
        )
        return merged_df
    
    ###############
    # main function
    ###############

    def run(self):
        """Runs the functions for CustomIngestor

        Returns:
            merged_df (pd.DataFrame): merged main DataFrame for EDA analyses and subsequent further preprocessing. 
        """

        df_list = self.get_df_list(self.folder_path)
        logger.debug(df_list)

        main_df, symp_df, vax_df = self.read_csvs(df_list)
        logger.debug(f"main_df: {main_df.info()}")
        logger.debug(f"symp_df: {symp_df.info()}")
        logger.debug(f"vax_df: {vax_df.info()}")

        # SYMPTOMS CSV - aggregate symptoms
        symp_df = self.aggregate_symps(
            symp_df, 
            self.symp_cols_dict
        )
        symp_df = self.create_count_sympt(symp_df)

        # VAX CSV - select only COVID19 vaccines
        vax_df = self.get_pfizer_moderna(
            vax_df, 
            self.vax_csv_cols
        )

        # MAIN DATA
        main_df = self.convert_dates(main_df)
        main_df = self.limit_max_date(main_df, self.max_date)
        logger.debug(
            f"Number of duplicates in main_df: \
                {main_df.duplicated(subset = 'VAERS_ID').sum()}"
        )

        merged_df = self.merge_dfs(
            main_df, 
            vax_df,
            symp_df
        )
        logger.debug(merged_df.info())

        full_merged_filepath = Path(__file__).parents[2] / self.merged_filepath
        merged_df.to_csv(
            full_merged_filepath, 
            index=False
        )
        logger.info(f"Merged file saved to: {full_merged_filepath}")
        
        return merged_df


