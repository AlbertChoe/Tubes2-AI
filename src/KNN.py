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
        print(type(self.X_train))
        print(self.X_train)
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

from scipy.spatial.distance import cdist
from scipy.stats import mode

class OptimizedKNN2:
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
        Predict labels for test points in a memory-efficient manner.
        
        Parameters:
        -----------
        X_test : pd.DataFrame
            Feature matrix for testing
        
        Returns:
        --------
        np.ndarray
            Predicted labels for test points
        """
        predictions = []
        
        for test_point in X_test.values:

            distances = cdist([test_point], self.X_train, metric=self.distance_metric)
            
            k_indices = np.argsort(distances[0])[:self.k]

            k_nearest_labels = self.y_train[k_indices]
            
            most_common_label = mode(k_nearest_labels).mode
            
            if most_common_label.ndim == 0:
                predictions.append(most_common_label)
            else:
                predictions.append(most_common_label[0])
        
        return np.array(predictions)
    
import heapq
import psutil

class MemoryEfficientKNN:
    def __init__(self, k: int = 3, memory_limit: float = 0.7):
        """
        Memory-efficient K-Nearest Neighbors implementation.
        
        Parameters:
        -----------
        k : int, optional (default=3)
            Number of nearest neighbors to consider
        memory_limit : float, optional (default=0.7)
            Fraction of available memory to use (helps prevent allocation errors)
        """
        self.k = k
        self.memory_limit = memory_limit
        
    def fit(self, X_train: pd.DataFrame, y_train: pd.Series) -> None:
        """
        Store training data efficiently.
        
        Parameters:
        -----------
        X_train : pd.DataFrame
            Feature matrix for training
        y_train : pd.Series
            Target labels for training
        """
        self.X_train = X_train.values.astype(np.float32)
        self.y_train = y_train.values
        
    def predict(self, X_test: pd.DataFrame) -> np.ndarray:
        """
        Predict labels with memory-conscious approach.
        
        Parameters:
        -----------
        X_test : pd.DataFrame
            Feature matrix for testing
        
        Returns:
        --------
        np.ndarray
            Predicted labels for test points
        """
        X_test_values = X_test.values.astype(np.float32)
        
        predictions = []
        
        batch_size = self._calculate_batch_size(X_test_values)
        
        for i in range(0, len(X_test_values), batch_size):
            batch = X_test_values[i:i+batch_size]
            batch_predictions = self._predict_batch(batch)
            predictions.extend(batch_predictions)
        
        return np.array(predictions)
    
    def _predict_batch(self, batch: np.ndarray) -> list:
        """
        Predict labels for a batch of test points.
        
        Parameters:
        -----------
        batch : np.ndarray
            Batch of test points
        
        Returns:
        --------
        list
            Predicted labels for the batch
        """
        batch_predictions = []
        
        for x in batch:
            nearest = []
            
            for j, x_train in enumerate(self.X_train):
                distance = np.sqrt(np.sum((x - x_train) ** 2))
                if len(nearest) < self.k:
                    heapq.heappush(nearest, (-distance, j))
                else:
                    heapq.heappushpop(nearest, (-distance, j))
            k_nearest_labels = [self.y_train[idx] for (_, idx) in nearest]
            batch_predictions.append(self._most_common_label(k_nearest_labels))
        
        return batch_predictions
    
    def _calculate_batch_size(self, X_test: np.ndarray) -> int:
        """
        Dynamically calculate batch size based on available memory.
        
        Parameters:
        -----------
        X_test : np.ndarray
            Test data matrix
        
        Returns:
        --------
        int
            Calculated batch size
        """
        try:
            element_size = X_test.dtype.itemsize
            total_elements = X_test.size
            estimated_memory = total_elements * element_size
            
            available_memory = psutil.virtual_memory().available
            
            batch_size = int((available_memory * self.memory_limit) / 
                             (estimated_memory / len(X_test) * element_size))
            
            return max(min(batch_size, len(X_test)), 1)
        
        except Exception:
            return max(len(X_test) // 10, 1)
    
    def _most_common_label(self, labels: list) -> Union[int, float]:
        """
        Return the most common label among the neighbors.
        
        Parameters:
        -----------
        labels : list
            List of neighbor labels
        
        Returns:
        --------
        Union[int, float]
            Most frequent label
        """
        unique, counts = np.unique(labels, return_counts=True)
        return unique[np.argmax(counts)]

    def score(self, X_test: pd.DataFrame, y_true: pd.Series) -> float:
        """
        Calculate the accuracy of predictions.
        
        Parameters:
        -----------
        X_test : pd.DataFrame
            Test feature matrix
        y_true : pd.Series
            True labels
        
        Returns:
        --------
        float
            Accuracy score
        """
        predictions = self.predict(X_test)
        return np.mean(predictions == y_true.values)