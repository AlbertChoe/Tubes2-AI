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
from scipy.stats import mode

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
    
    def fit(self, X_train: pd.DataFrame, y_train: pd.Series) -> None:
        """
        Store training data.
        
        Parameters:
        -----------
        X_train : pd.DataFrame
            Feature matrix for training
        y_train : pd.Series
            Target labels for training
        """
        self.X_train = X_train.values
        self.y_train = y_train.values
    
    def predict(self, X_test: pd.DataFrame) -> np.ndarray:
        """
        Vectorized prediction for multiple test points.
        
        Parameters:
        -----------
        X_test : pd.DataFrame
            Feature matrix for testing
        
        Returns:
        --------
        np.ndarray
            Predicted labels for test points
        """
        distances = cdist(X_test.values, self.X_train, metric=self.distance_metric)
        
        k_indices = distances.argsort(axis=1)[:, :self.k]
        
        k_nearest_labels = self.y_train[k_indices]
        
        predictions = mode(k_nearest_labels, axis=1).mode.flatten()
        
        return predictions