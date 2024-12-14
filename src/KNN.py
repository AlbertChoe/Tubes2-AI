import numpy as np
import pandas as pd
from typing import Union

class KNN:
    def __init__(self, k: int = 3):
        self.k = k
    
    def fit(self, X_train: pd.DataFrame, y_train: pd.Series) -> None:
        """Fit the model to the training data."""
        self.X_train = X_train.values
        self.y_train = y_train.values
    
    def predict(self, X_test: pd.DataFrame) -> np.ndarray:
        """Predict the class labels for the test data."""
        X_test = X_test.values
        return np.array([self._predict(x) for x in X_test])
    
    def _predict(self, x: np.ndarray) -> Union[int, float]:
        """Predict the class label for a single test point."""
        distances = [self._euclidean_distance(x, x_train) for x_train in self.X_train]
        
        k_indices = np.argsort(distances)[:self.k]
    
        k_nearest_labels = [self.y_train[i] for i in k_indices]
        
        most_common = self._most_common_label(k_nearest_labels)
        return most_common
    
    def _euclidean_distance(self, x1: np.ndarray, x2: np.ndarray) -> float:
        """Calculate the Euclidean distance between two points."""
        return np.sqrt(np.sum((x1 - x2) ** 2))
    
    def _most_common_label(self, labels: np.ndarray) -> Union[int, float]:
        """Return the most common label among the neighbors."""
        unique, counts = np.unique(labels, return_counts=True)
        most_common = unique[np.argmax(counts)]
        return most_common

from scipy.spatial.distance import cdist
from scipy.sparse import csr_matrix

class OptimizedKNN:
    def __init__(self, k: int = 3, distance_metric: str = 'euclidean'):
        """
        Initialize KNN with more flexible distance metric options.
        
        Parameters:
        -----------
        k : int, optional (default=3)
            Number of nearest neighbors to consider
        distance_metric : str, optional (default='euclidean')
            Distance metric to use for neighbor calculation
        """
        self.k = k
        self.distance_metric = distance_metric
    
    def _ensure_ndarray(self, X):
        """
        Ensure the input data is converted to a NumPy ndarray.
        
        Parameters:
        -----------
        X : pd.DataFrame, csr_matrix, or ndarray
            Input data to convert.
        
        Returns:
        --------
        np.ndarray
            Converted data as an ndarray.
        """
        if isinstance(X, pd.DataFrame):
            return X.values
        elif isinstance(X, csr_matrix):
            return X.toarray()
        elif isinstance(X, np.ndarray):
            return X
        if isinstance(X, pd.Series):
            return X.values
        else:
            raise TypeError("Input data must be a DataFrame, csr_matrix, or ndarray.")
    
    def fit(self, X_train, y_train) -> None:
        """
        Store training data after ensuring it's an ndarray.
        
        Parameters:
        -----------
        X_train : pd.DataFrame, csr_matrix, or ndarray
            Feature matrix for training
        y_train : pd.Series, or ndarray
            Target labels for training
        """
        self.X_train = self._ensure_ndarray(X_train)
        self.y_train = np.array(y_train)
    
    def predict(self, X_test) -> np.ndarray:
        """
        Predict labels for test points in a memory-efficient manner.
        
        Parameters:
        -----------
        X_test : pd.DataFrame, csr_matrix, or ndarray
            Feature matrix for testing
        
        Returns:
        --------
        np.ndarray
            Predicted labels for test points
        """
        X_test = self._ensure_ndarray(X_test)
        predictions = []
        
        for test_point in X_test:
            distances = cdist([test_point], self.X_train, metric=self.distance_metric)
            
            k_indices = np.argsort(distances[0])[:self.k]
            
            k_nearest_labels = self.y_train[k_indices]

            unique_labels, counts = np.unique(k_nearest_labels, return_counts=True)
            most_common_label = unique_labels[np.argmax(counts)]
            
            predictions.append(most_common_label)
        
        return np.array(predictions)