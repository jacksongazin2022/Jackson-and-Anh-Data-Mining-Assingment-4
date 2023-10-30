import gzip
import os
import shutil
import pandas as pd
from pyarrow.parquet import read_table
from scipy.sparse import coo_matrix
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, MinMaxScaler, normalize
from sklearn.pipeline import Pipeline
import numpy as np
from sklearn.preprocessing import StandardScaler, MaxAbsScaler
from sklearn.decomposition import TruncatedSVD
from sklearn.base import BaseEstimator, TransformerMixin
from memory_profiler import memory_usage
import matplotlib.pyplot as plt
import seaborn as sns
from kneed import KneeLocator
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.metrics import adjusted_rand_score
from sklearn.model_selection import GridSearchCV
from yellowbrick.cluster import KElbowVisualizer, SilhouetteVisualizer
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.model_selection import GridSearchCV
from scipy import stats
from sklearn.cluster import HDBSCAN
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import silhouette_score
from scipy import stats
from sklearn.neighbors import LocalOutlierFactor
from sklearn.model_selection import train_test_split

# You can also define custom functions, classes, and other code in this module.

def load_data(data_path, row_info_path, column_info_path):
    # Load non_zero parquet data
    table = read_table(data_path)
    nonzero_data = table.to_pandas()
    
    # Adjust column indices to be 0-based
    nonzero_data['col_indices'] = nonzero_data['col_indices'] - 1
    
    # Load row and column index info
    rows = pd.read_csv(row_info_path)
    row_names = rows.iloc[:, 1].to_list()
    
    columns = pd.read_csv(column_info_path)
    column_names = columns.iloc[:, 1].to_list()
    
    # Convert the sparse matrix to a dense DataFrame
    sparse_matrix = coo_matrix(
        (nonzero_data['nonzero_elements'], (nonzero_data['row_indices'], nonzero_data['col_indices'])),
        shape=(len(row_names), len(column_names))
    )
    
    
   

    print('Returning sparse_matrix, column_names, row_names, and row_indices')
    
    row_indices = np.arange(sparse_matrix.shape[0])
    
    return sparse_matrix, column_names, row_names, row_indices


class SparseTrainTestSplit(BaseEstimator, TransformerMixin):
    def __init__(self, test_size=0.2, random_state=None, row_indices = None):
        self.test_size = test_size
        self.random_state = random_state
        self.row_indices = row_indices

    def fit(self, X, y=None):
        # Not needed for this class
        return self
    def transform(self, X):
        # Split the data using train_test_split
        sparse_train, sparse_test, train_indices, test_indices = train_test_split(
            X, self.row_indices, test_size=self.test_size, random_state=self.random_state
        )

        # Convert the split data back to coo_matrix
        sparse_train = coo_matrix(sparse_train)
        sparse_test = coo_matrix(sparse_test)

        return sparse_train, sparse_test, train_indices, test_indices

class DataClean:
    def __init__(self, remove_by_column=True):
        self.remove_by_column = remove_by_column
        self.output_folder = "../output"
        
        # Create the output folder if it doesn't exist
        if not os.path.exists(self.output_folder):
            os.makedirs(self.output_folder)

    def fit(self, X, y=None):
        # Check if there are NaN values in the data
        self.has_nan_values = np.isnan(X.data).any()

        with open(os.path.join(self.output_folder, "na_info.txt"), "w") as f:
            if self.has_nan_values:
                # Find rows and columns with NaN values
                self.rows_with_nan = self.find_rows_with_nan(X)
                self.columns_with_nan = self.find_columns_with_nan(X)

                f.write("Rows with NaN Values:\n")
                f.write(", ".join(map(str, self.rows_with_nan)))
                f.write("\nTotal Rows with NaN Values: {}\n".format(len(self.rows_with_nan)))

                f.write("\nColumns with NaN Values:\n")
                f.write(", ".join(map(str, self.columns_with_nan)))
                f.write("\nTotal Columns with NaN Values: {}\n".format(len(self.columns_with_nan)))
            else:
                f.write("No NaN Values Found in the Data.\n")

        return self

    def transform(self, X):
        if self.has_nan_values:
            if self.remove_by_column:
                X = self._remove_nan_by_column(X)
            else:
                X = self._remove_nan_by_row(X)

        return X

    def _remove_nan_by_column(self, X):
        nan_mask = np.isnan(X.data)
        valid_columns = np.unique(X.col[~nan_mask])
        X = X.tocsc()[:, valid_columns].tocoo()
        return X

    def _remove_nan_by_row(self, X):
        if not isinstance(X, coo_matrix):
            raise ValueError("Input must be a sparse COO matrix.")
    
        non_nan_rows = ~np.isnan(X.sum(axis=1).A.ravel())
        X = X.tocsr()[non_nan_rows].tocoo()
        return X

    def find_rows_with_nan(self, X):
        nan_mask = np.isnan(X.data)
        rows_with_nan = np.unique(X.row[nan_mask])
        return rows_with_nan

    def find_columns_with_nan(self, X):
        nan_mask = np.isnan(X.data)
        columns_with_nan = np.unique(X.col[nan_mask])
        return columns_with_nan
class Reindex:
    def __init__(self, columns=None, names=None, output_folder="../Output"):
        self.columns = columns
        self.names = names
        self.output_folder = output_folder

    def fit(self, X, y=None):
        return self

    def transform(self, X, indices, data_type):
        row_names_indices = [self.names[i] for i in indices]
        df_with_indices = pd.DataFrame(X, columns=self.columns, index=row_names_indices)
        
        # Generate the output filename based on the data type (e.g., "train" or "test")
        output_filename = f"pca_{data_type}_df.csv"
        
        # Save the DataFrame to a CSV file in the specified output folder
        output_path = os.path.join(self.output_folder, output_filename)
        df_with_indices.to_csv(output_path)
        
        return df_with_indices
class DataEDAPCA:
    def __init__(self, columns, z_threshold=10):
        self.columns = columns
        self.z_threshold = z_threshold
        self.removed_indices = None
        self.updated_df = None
        self.outlier_df_cleaned = None

    def plot_box_and_scatter(self, X):
        # Create a boxplot of the specified columns
        X[self.columns].boxplot()
        plt.title(f"Boxplot of {self.columns[0]} and {self.columns[1]}")
        plt.show()

        # Create a scatterplot of the specified columns
        plt.scatter(X[self.columns[0]], X[self.columns[1]])
        plt.xlabel(self.columns[0])
        plt.ylabel(self.columns[1])
        plt.title(f"Scatterplot of {self.columns[0]} and {self.columns[1]}")
        plt.show()
    def remove_outliers(self, X):
        # Calculate Z-scores for the specified columns
        z_scores = stats.zscore(X[self.columns])

        # Create a DataFrame for Z-scores
        z_scores_df = pd.DataFrame(z_scores, columns=self.columns, index=X.index)

        print("DataFrame of Z-scores sent to output folder")
        z_scores_output_path = '../Output/z_scores.csv'
        z_scores_df.to_csv(z_scores_output_path)

        print("Z-threshold:", self.z_threshold)
        # Find rows where the value of any column is greater than the absolute value of the threshold
        outlier_mask = (np.abs(z_scores_df) > self.z_threshold).any(axis=1)

        # Create a new DataFrame containing outliers
        outlier_df = X[outlier_mask]
        
        total_rows = outlier_df.shape[0]
        
      

        print("Df of Outliers sent to output folder")
        outlier_output_path = '../Output/outlier_indices.csv'
        outlier_df.to_csv(outlier_output_path)

        # Drop the rows with high Z-scores from the original DataFrame X
        print(f'Removing {total_rows} from our dataframe as they exceed our threshold of abs{self.z_threshold}')
        X.drop(outlier_df.index, inplace=True)
        self.removed_indices = outlier_df.index.tolist()
        self.updated_df = X
        self.outlier_df_cleaned = outlier_df

    def fit_transform(self, X, another_df=None):
        # Step 1: Display the initial boxplot and scatterplot
        print('Box Plot and Scatterplot of Data Set with Outliers')
        self.plot_box_and_scatter(X)

        # Step 2: Notify rows with high Z-scores and remove them
        self.remove_outliers(X)
       

        # Step 3: If another_df is provided, subset it using the same indices
        removed_metadata = None
        if another_df is not None:
            removed_metadata = another_df[another_df['name'].isin(self.removed_indices)]

        # Step 4: Display the boxplot and scatterplot of the updated data
        print('Box Plot and Scatterplot of Data Set without Outliers')
        self.plot_box_and_scatter(X)
        print('Returning updated dataframe, metadata of rows we removed, and PC of rows we removed')
        return self.updated_df, removed_metadata, self.outlier_df_cleaned



