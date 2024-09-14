import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import missingno as msno




class DataUtils:
    '''
    This class is used to clean, visualize and identify outliers, calculate PCA and perform KMeans clustering from the dataset
    '''
    def __init__(self,data: pd.DataFrame):
        self.data = data

    def data_info(self):
        '''
        Provides information about the data: 
            * provides the percentage of missing values
            * The number of missing values for each column
            * the data types of the missing values
        '''
        
        missing_values = self.data.isna().sum()
        missing_percent = self.data.isna().mean() * 100 
        data_types = self.data.dtypes

        

        info_df = pd.DataFrame({
            "Missing values" : missing_values, 
            "Missing Percentage" : missing_percent, 
            "Dtypes" : data_types
        })

        info_df = info_df[missing_percent > 0]
        info_df = info_df.sort_values(by='Missing Percentage', ascending=False)

        max_na_col = list(info_df.loc[info_df['Missing values'] == info_df['Missing values'].max()].index)
        more_than_half_na = list(info_df.loc[info_df['Missing Percentage'] > 50].index)

        print(f"**The data contains `{self.data.shape[0]}` rows and `{self.data.shape[1]}` columns.**\n\n"
            f"**The data has `{info_df.shape[0]}` missing columns.**\n\n"
            f"**The column with the maximum number of missing values is `{max_na_col}`.**\n\n"
            f"**Columns with more than 50% missing values are:**")

        # Print the list of columns with more than 50% missing values
        for column in more_than_half_na:
            print(f"- `{column}`")
            
        
        return info_df
    

    def visualize_missing_values(self):
        '''
        This method generates a heatmap to visually represent the missing values in the dataset.
        '''

        missing_cols = self.data.columns[self.data.isna().any()]

        missing_data = self.data[missing_cols]

        msno.bar(missing_data)


    def visualize_outliers(self):
        '''
        This funcions helps in visualizing outliers using boxplot
        '''

        numerical_cols = self.data.select_dtypes(include=['float64', 'int64']).columns
        num_cols = len(numerical_cols)
        nrows = num_cols // 5 + num_cols % 5

        fig, axes = plt.subplots(nrows=nrows, ncols=5, figsize=(20,12))
        axes = axes.flatten()

        for i, col in enumerate(numerical_cols):
            sns.boxplot(y=self.data[col], ax=axes[i])
            axes[i].set_title(col)

        plt.tight_layout()
        plt.show()


    def num_univariant_visualization(self):
        '''
        This funcions helps in visualizing histograms for numeric columns
        '''

        numerical_cols = self.data.select_dtypes(include=['float64', 'int64']).columns
        num_cols = len(numerical_cols)
        nrows = num_cols // 5 + num_cols % 5

        fig, axes = plt.subplots(nrows=nrows, ncols=5, figsize=(20,12))
        axes = axes.flatten()

        for i, col in enumerate(numerical_cols):
            sns.histplot(self.data[col], ax=axes[i], bins=10, kde=True)  
            axes[i].set_title(col)

        for j in range(i + 1, len(axes)):
            axes[j].set_visible(False)        

        plt.tight_layout()
        plt.show()

    def cat_univariant_visualization(self):
        '''
        This funcions helps in visualizing histograms for categorical columns
        '''

        categorical_cols = self.data.select_dtypes(include=['object']).columns
        cat_cols = len(categorical_cols)
        nrows = cat_cols // 5 + cat_cols % 5

        fig, axes = plt.subplots(nrows=nrows, ncols=5, figsize=(20,12))
        axes = axes.flatten()


        for i, col in enumerate(categorical_cols):
            self.data[col].value_counts().plot(kind='bar', edgecolor='black', ax=axes[i])
            plt.title(f'Bar Chart of {col}')
            plt.xlabel(col)
            plt.ylabel('Frequency')
               

        plt.tight_layout()
        plt.show()   

    def outlier_remover(self, columns: list) -> pd.DataFrame:
        '''
        This funtion removes all the outliers in a data using IQR technique

        Parameters: 
            columns(list): A list of columns that we don't need to remove the outlier, like unique identifiers

        Returns:
            pd.DataFrame: A dataframe without outliers
        '''

        numeric_col = self.data.select_dtypes(include='float64').columns
        fix_cols = [col for col in numeric_col if col not in columns]

        Q1 = self.data[fix_cols].quantile(0.25)
        Q3 = self.data[fix_cols].quantile(0.75)

        IQ = Q3 - Q1

        lower_bound = Q1 - 1.5 * IQ
        upper_bound = Q3 + 1.5 * IQ

        self.data[fix_cols] = self.data[fix_cols].clip(lower=lower_bound, upper=upper_bound, axis=1)

        return self.data  