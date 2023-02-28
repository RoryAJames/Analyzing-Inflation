import pandas as pd
import numpy as np

class OutlierRemover:
    def __init__(self, df):
        self.df = df

    def remove_outliers_iqr(self, column_name):
        Q1 = self.df[column_name].quantile(0.25)
        Q3 = self.df[column_name].quantile(0.75)
        IQR = Q3 - Q1
        lower_limit = Q1 - 1.5*IQR
        upper_limit = Q3 + 1.5*IQR
        self.df = self.df[(self.df[column_name] >= lower_limit) & (self.df[column_name] <= upper_limit)]
        return self.df

    def remove_outliers_3std(self, column_name):
        mean = np.mean(self.df[column_name])
        std_dev = np.std(self.df[column_name])
        lower_limit = mean - 3*std_dev
        upper_limit = mean + 3*std_dev
        self.df = self.df[(self.df[column_name] >= lower_limit) & (self.df[column_name] <= upper_limit)]
        return self.df
