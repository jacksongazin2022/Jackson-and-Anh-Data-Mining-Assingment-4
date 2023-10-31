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
from sklearn.preprocessing import MaxAbsScaler
from sklearn.decomposition import TruncatedSVD
from sklearn.base import BaseEstimator, TransformerMixin
from memory_profiler import memory_usage
import matplotlib.pyplot as plt
import seaborn as sns
from kneed import KneeLocator
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, adjusted_rand_score
from sklearn.model_selection import GridSearchCV
from yellowbrick.cluster import KElbowVisualizer, SilhouetteVisualizer
from sklearn.cluster import HDBSCAN
from scipy import stats
from sklearn.neighbors import LocalOutlierFactor
from sklearn.model_selection import train_test_split
import pickle
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.cluster import KMeans
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import adjusted_rand_score
from sklearn.metrics import make_scorer
from sklearn.metrics import silhouette_samples


# You can also define custom functions, classes, and other code in this module.

import pandas as pd
import numpy as np
from scipy.sparse import coo_matrix

def load_data(data_path, row_info_path, column_info_path, transpose=False):
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
    if transpose:
        transposed_matrix = sparse_matrix.transpose()
        row_indices = np.arange(transposed_matrix.shape[0])
        print('Returning Transposed matrix, row_names of the transposed matrix, col_names of the transposed matrix, and row_indices of transposed matrix')
        return transposed_matrix, column_names, row_names, row_indices
    else:
        row_indices = np.arange(sparse_matrix.shape[0])
        print('Returning sparse_matrix, column_names, row_names, and row_indices')
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
    def __init__(self, trans=False, remove_by_column=True):
        self.remove_by_column = remove_by_column
        self.output_folder = "../output"
        self.trans = trans
        
        # Create the output folder if it doesn't exist
        if not os.path.exists(self.output_folder):
            os.makedirs(self.output_folder)

    def fit(self, X, y=None):
        # Check if there are NaN values in the data
        self.has_nan_values = np.isnan(X.data).any()
        if not self.trans:
            file_name = "na_info.txt"
        else:
            file_name = "na_info.txt_trans"

        with open(os.path.join(self.output_folder, file_name), "w") as f:
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
    def __init__(self, trans = False, columns=None, names=None, output_folder="../Output"):
        self.columns = columns
        self.names = names
        self.output_folder = output_folder
        self.trans = trans

    def fit(self, X, y=None):
        return self
    def save_to_csv(self, dataframe, filename):
        if self.trans:
            filename = filename + "_trans.csv"  # Add "_trans" to the filename if data is transposed
        else:
            filename = filename + ".csv"
        dataframe.to_csv(filename)

    def transform(self, X, indices, data_type):
        row_names_indices = [self.names[i] for i in indices]
        df_with_indices = pd.DataFrame(X, columns=self.columns, index=row_names_indices)
        
        # Generate the output filename based on the data type (e.g., "train" or "test")
        output_filename = f"../Output/pca_{data_type}_df"
        self.save_to_csv(df_with_indices, output_filename)
        
        return df_with_indices
class DataEDAPCA:
    def __init__(self, columns, trans = False, z_threshold=10, graphs = False):
        self.columns = columns
        self.z_threshold = z_threshold
        self.removed_indices = None
        self.updated_df = None
        self.outlier_df_cleaned = None
        self.trans = trans
        self.graphs = graphs

    def plot_box_and_scatter(self, X):
        # Create a boxplot of the specified columns
        if not self.graphs:
            return
        X[self.columns].boxplot()
        plt.title(f"Boxplot of {self.columns[0]} and {self.columns[1]}")
        plt.show()

        # Create a scatterplot of the specified columns
        plt.scatter(X[self.columns[0]], X[self.columns[1]])
        plt.xlabel(self.columns[0])
        plt.ylabel(self.columns[1])
        plt.title(f"Scatterplot of {self.columns[0]} and {self.columns[1]}")
        plt.show()
    def save_to_csv(self, dataframe, filename):
        if self.trans:
            filename = filename + "_trans.csv"  # Add "_trans" to the filename if data is transposed
        else:
            filename = filename + ".csv"
        dataframe.to_csv(filename)  # Add "_trans" to the filename if data is transposed
    def remove_outliers(self, X):
        # Calculate Z-scores for the specified columns
        z_scores = stats.zscore(X[self.columns])

        # Create a DataFrame for Z-scores
        z_scores_df = pd.DataFrame(z_scores, columns=self.columns, index=X.index)

        print("DataFrame of Z-scores sent to output folder")
        self.save_to_csv(z_scores_df, '../Output/z_scores')
        print("Z-threshold:", self.z_threshold)
        # Find rows where the value of any column is greater than the absolute value of the threshold
        outlier_mask = (np.abs(z_scores_df) > self.z_threshold).any(axis=1)

        # Create a new DataFrame containing outliers
        outlier_df = X[outlier_mask]
        self.outlier_df_cleaned = outlier_df
        
        total_rows = outlier_df.shape[0]
        print("Df of Outliers sent to output folder")
        self.save_to_csv(self.outlier_df_cleaned, '../Output/outlier_indices')
        
            

        # Drop the rows with high Z-scores from the original DataFrame X
        print(f'Removing {total_rows} from our dataframe as they exceed our threshold of abs{self.z_threshold}')
        X.drop(outlier_df.index, inplace=True)
        self.removed_indices = outlier_df.index.tolist()
        self.updated_df = X
    def fit_transform(self, X, another_df=None):
        # Step 1: Display the initial boxplot and scatterplot
        print('Box Plot and Scatterplot of Data Set with Outliers')
        self.plot_box_and_scatter(X)

        # Step 2: Notify rows with high Z-scores and remove them
        self.remove_outliers(X)
        print('Sending updated Df to Output')
        ## Sending updated df to Output
        self.save_to_csv(X,'../Output/pca_train_df_without_outliers')
       

        # Step 3: If another_df is provided, subset it using the same indices
        removed_metadata = None
        if another_df is not None:
            removed_metadata = another_df[another_df['name'].isin(self.removed_indices)]

        # Step 4: Display the boxplot and scatterplot of the updated data
        print('Box Plot and Scatterplot of Data Set without Outliers')
        self.plot_box_and_scatter(X)
        print('Returning updated dataframe, metadata of rows we removed, and PC of rows we removed')
        return self.updated_df, removed_metadata, self.outlier_df_cleaned



class Optimize_and_Compare_Hdbscan(BaseEstimator, TransformerMixin):
    def __init__(self, hdbscan_params, alpha=0.05, random_state=42):
        self.hdbscan_params = hdbscan_params
        self.alpha = alpha
        self.grid_search_scores = {}  # Store the silhouette scores for grid search
        self.default_estimator_scores = {}  # Store information for the default estimator
        self.choice = "Default Hdbscan Estimator"
        self.random_state = random_state
    def silhouette_scorer(self, estimator, X):  # Define it as an instance method
        labels = estimator.fit_predict(X)
        if len(set(labels)) == 1:
            return 0  # Silhouette score is undefined for a single cluster
        return silhouette_score(X, labels)
    
    def fit(self, X, y=None):
        # Perform Grid Search
        grid = GridSearchCV(HDBSCAN(min_cluster_size=20), self.hdbscan_params, cv=3, scoring=self.silhouette_scorer, refit=True)
        grid.fit(X)

        # Save best estimator
        grid_search_estimator = grid.best_estimator_

        # Store the results of grid search
        self.grid_search_scores = grid.cv_results_
        grid_search_silhouette_score = silhouette_score(X, grid_search_estimator.labels_)

        # Fit the default Kmeans
        default_hdbscan = HDBSCAN(min_cluster_size=20).fit(X)
        

        # Store the results
        self.default_estimator_scores['parameters'] = default_hdbscan.get_params()
        self.default_estimator_scores['silhouette_score'] = silhouette_score(X, default_hdbscan.labels_)

        # Compare silhouette scores and choose the best estimator
        if grid_search_silhouette_score > self.default_estimator_scores['silhouette_score']:
            t_stat, p_value = stats.ttest_ind(default_hdbscan.labels_, grid_search_estimator.labels_)

            # Set the default choice to "Grid Search Estimator"

            # Output informative print statements
            print("Default Hdbscan Silhouette Score:", self.default_estimator_scores['silhouette_score'])
            print("Grid Search Estimator Silhouette Score:", grid_search_silhouette_score)
            if p_value < self.alpha:
                self.choice = "Grid Search Estimator"
                print("The difference between the two groups is statistically significant.")
                print(f"Using {self.choice} as it performs significantly better using a threshold of alpha = .05 .")
            else:
                print("The difference between the two groups is not statistically significant.")
                print(f"Using {self.choice} as there is no significant improvement using a threshold of alpha = .05.")
        else:
            print("Grid Search Estimator Silhouette Score:", grid_search_silhouette_score)
            print("Default KMeans Silhouette Score:", self.default_estimator_scores['silhouette_score'])
            print("Default Parameter has a higher Silhouette Score.")
            print("Using Default Parameter as it performs better based on Silhouette Score.")

        self.best_estimator = grid_search_estimator if self.choice == "Grid Search Estimator" else default_hdbscan
        
        
        return self

    def transform(self, X, y=None):
        return self.best_estimator

class OptimizeAndCompareKMeans(BaseEstimator, TransformerMixin):
    def __init__(self, kmeans_params, alpha=0.05, random_state=42):
        self.kmeans_params = kmeans_params
        self.alpha = alpha
        self.grid_search_scores = {}  # Store the silhouette scores for grid search
        self.default_estimator_scores = {}  # Store information for the default estimator
        self.choice = "Default KMeans Estimator"
        self.random_state = random_state
    def silhouette_scorer(self, estimator, X):  # Define it as an instance method
        labels = estimator.fit_predict(X)
        if len(set(labels)) == 1:
            return 0  # Silhouette score is undefined for a single cluster
        return silhouette_score(X, labels)
    
    def fit(self, X, y=None):
        # Perform Grid Search
        grid = GridSearchCV(KMeans(random_state=self.random_state), self.kmeans_params, cv=3, scoring=self.silhouette_scorer, refit=True)
        grid.fit(X)

        # Save best estimator
        grid_search_estimator = grid.best_estimator_

        # Store the results of grid search
        self.grid_search_scores = grid.cv_results_
        grid_search_silhouette_score = silhouette_score(X, grid_search_estimator.labels_)

        # Fit the default Kmeans
        default_kmeans = KMeans(n_clusters=2, random_state=0, n_init="auto").fit(X)

        # Store the results
        self.default_estimator_scores['parameters'] = default_kmeans.get_params()
        self.default_estimator_scores['silhouette_score'] = silhouette_score(X, default_kmeans.labels_)

        # Compare silhouette scores and choose the best estimator
        if grid_search_silhouette_score > self.default_estimator_scores['silhouette_score']:
            t_stat, p_value = stats.ttest_ind(default_kmeans.labels_, grid_search_estimator.labels_)

            # Set the default choice to "Grid Search Estimator"

            # Output informative print statements
            print("Default KMeans Silhouette Score:", self.default_estimator_scores['silhouette_score'])
            print("Grid Search Estimator Silhouette Score:", grid_search_silhouette_score)
            if p_value < self.alpha:
                self.choice = "Grid Search Estimator"
                print("The difference between the two groups is statistically significant.")
                print(f"Using {self.choice} as it performs significantly better using a threshold of alpha = .05 .")
            else:
                print("The difference between the two groups is not statistically significant.")
                print(f"Using {self.choice} as there is no significant improvement using a threshold of alpha = .05.")
        else:
            print("Grid Search Estimator Silhouette Score:", grid_search_silhouette_score)
            print("Default KMeans Silhouette Score:", self.default_estimator_scores['silhouette_score'])
            print("Default Parameter has a higher Silhouette Score.")
            print("Using Default Parameter as it performs better based on Silhouette Score.")

        self.best_estimator = grid_search_estimator if self.choice == "Grid Search Estimator" else default_kmeans
        return self

    def transform(self, X, y=None):
        return self.best_estimator
    def save_to_csv(self, dataframe, filename):
        if self.trans:
            filename = filename + "_trans.csv"  # Add "_trans" to the filename if data is transposed
        else:
            filename = filename + ".csv"
        dataframe.to_csv(filename)
def create_labels_and_scoring_df(estimator, output_file_location, pca_train_df, pca_test_df, trans = False):
    # Extract cluster labels for training and test data
    train_cluster_labels = estimator.labels_
    test_cluster_labels = estimator.predict(pca_test_df)

    # Calculate silhouette scores for training and test data points
    silhouette_train_samples = silhouette_samples(pca_train_df, train_cluster_labels)
    silhouette_test_samples = silhouette_samples(pca_test_df, test_cluster_labels)

    # Create DataFrames for training and test data
    train_df = pd.DataFrame({
        'Index': pca_train_df.index,
        'Data_Type': 'Train',
        'Cluster_Label': train_cluster_labels,
        'Silhouette_Score': silhouette_train_samples
    })

    test_df = pd.DataFrame({
        'Index': pca_test_df.index,
        'Data_Type': 'Test',
        'Cluster_Label': test_cluster_labels,
        'Silhouette_Score': silhouette_test_samples
    })

    # Concatenate the DataFrames for training and test data
    result_df = pd.concat([train_df, test_df])
    
    print(f'Sending Result file to {output_file_location}')

    # Save the concatenated DataFrame to a single CSV file
    save_to_csv(result_df, output_file_location, trans)

    return result_df


