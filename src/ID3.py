import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from typing import Optional, Union, List

class DecisionNode:
    def __init__(
        self, 
        feature: Optional[int] = None, 
        threshold: Optional[float] = None, 
        left: Optional[Union['DecisionNode', int]] = None, 
        right: Optional[Union['DecisionNode', int]] = None, 
        *, 
        value: Optional[int] = None
    ):
        """
        Initialize a decision tree node.
        
        Args:
            feature (int, optional): Index of the feature used for splitting.
            threshold (float, optional): Threshold value for splitting.
            left (DecisionNode or int, optional): Left child node or leaf value.
            right (DecisionNode or int, optional): Right child node or leaf value.
            value (int, optional): Leaf node value.
        """
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value

class DecisionTreeID3:
    def __init__(
        self, 
        max_depth: Optional[int] = None, 
        min_samples_split: int = 10, 
        min_information_gain: float = 1e-5
    ):
        """
        Initialize the Decision Tree ID3 classifier.
        
        Args:
            max_depth (int, optional): Maximum depth of the tree.
            min_samples_split (int): Minimum number of samples to split an internal node.
            min_information_gain (float): Minimum information gain for a split.
        """
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_information_gain = min_information_gain
        self.tree = None
        self.label_encoder = LabelEncoder()
        self.features: Optional[np.ndarray] = None

    def fit(self, X_train: pd.DataFrame, y_train: pd.Series) -> 'DecisionTreeID3':
        """
        Fit the decision tree classifier.
        
        Args:
            X_train (pd.DataFrame): Training feature matrix.
            y_train (pd.Series): Training target vector.
        
        Returns:
            DecisionTreeID3: Fitted decision tree.
        """
        self.features = np.array(X_train.columns)
        
        X = X_train.values
        y = self.label_encoder.fit_transform(y_train)
        
        self.tree = self._build_tree(X, y)
        return self

    @staticmethod
    def _entropy(y: np.ndarray) -> float:
        """
        Calculate entropy of a target vector.
        
        Args:
            y (np.ndarray): Target vector.
        
        Returns:
            float: Entropy value.
        """
        counts = np.bincount(y)
        probabilities = counts / counts.sum()

        return -np.sum(probabilities * np.log2(probabilities + 1e-9))

    def _best_split(self, X: np.ndarray, y: np.ndarray) -> tuple:
        """
        Find the best feature and threshold for splitting.
        
        Args:
            X (np.ndarray): Feature matrix.
            y (np.ndarray): Target vector.
        
        Returns:
            tuple: Best feature index, best threshold, and best information gain.
        """
        parent_entropy = self._entropy(y)
        _, n_features = X.shape
        best_feature, best_threshold, best_info_gain = None, None, -1

        for feature_idx in range(n_features):
            column = X[:, feature_idx]
            sorted_indices = np.argsort(column)
            sorted_column = column[sorted_indices]
            sorted_labels = y[sorted_indices]

            unique_values, unique_indices = np.unique(sorted_column, return_index=True)
            if len(unique_values) <= 1:
                continue

            thresholds = (sorted_column[unique_indices[:-1]] + sorted_column[unique_indices[1:]]) / 2

            max_info_gain_idx, max_info_gain = self._calculate_info_gain(
                sorted_labels, unique_indices, thresholds, parent_entropy
            )

            if max_info_gain > best_info_gain:
                best_info_gain = max_info_gain
                best_feature = feature_idx
                best_threshold = thresholds[max_info_gain_idx]

        return best_feature, best_threshold, best_info_gain

    def _calculate_info_gain(
        self, 
        sorted_labels: np.ndarray, 
        unique_indices: np.ndarray, 
        thresholds: np.ndarray, 
        parent_entropy: float
    ) -> tuple:
        """
        Calculate information gain for potential splits.
        
        Args:
            sorted_labels (np.ndarray): Sorted target labels.
            unique_indices (np.ndarray): Indices of unique values.
            thresholds (np.ndarray): Potential split thresholds.
            parent_entropy (float): Entropy of the parent node.
        
        Returns:
            tuple: Index of best split and corresponding information gain.
        """
        n_samples = len(sorted_labels)
        n_classes = len(np.unique(sorted_labels))
        
        # Prepare arrays for entropy calculation
        entropies_left = np.zeros(len(thresholds))
        entropies_right = np.zeros(len(thresholds))
        
        for cls in range(n_classes):
            cls_mask = sorted_labels == cls
            cls_cumsum = np.cumsum(cls_mask)

            left_counts = cls_cumsum[unique_indices[1:] - 1]
            right_counts = cls_cumsum[-1] - left_counts
            
            split_sizes_left = unique_indices[1:] 
            split_sizes_right = n_samples - split_sizes_left

            probs_left = np.divide(left_counts, split_sizes_left, 
                                   out=np.zeros_like(left_counts, dtype=float), 
                                   where=split_sizes_left != 0)
            probs_right = np.divide(right_counts, split_sizes_right, 
                                    out=np.zeros_like(right_counts, dtype=float), 
                                    where=split_sizes_right != 0)

            entropies_left -= probs_left * np.log2(probs_left + 1e-9)
            entropies_right -= probs_right * np.log2(probs_right + 1e-9)
        
        weighted_entropy = (
            (split_sizes_left / n_samples) * entropies_left + 
            (split_sizes_right / n_samples) * entropies_right
        )
        info_gains = parent_entropy - weighted_entropy
        
        max_info_gain_idx = np.argmax(info_gains)
        return max_info_gain_idx, info_gains[max_info_gain_idx]

    def _build_tree(self, X: np.ndarray, y: np.ndarray, depth: int = 0) -> Union[DecisionNode, int]:
        """
        Recursively build the decision tree.
        
        Args:
            X (np.ndarray): Feature matrix.
            y (np.ndarray): Target vector.
            depth (int): Current tree depth.
        
        Returns:
            Union[DecisionNode, int]: Decision tree node or leaf value.
        """
        num_samples, _ = X.shape
        num_labels = len(np.unique(y))

        if (num_labels == 1 or
            num_samples < self.min_samples_split or
            (self.max_depth is not None and depth >= self.max_depth)):
            return np.bincount(y).argmax()

        best_feature, best_threshold, best_info_gain = self._best_split(X, y)

        if (best_info_gain < self.min_information_gain or 
            best_feature is None):
            return np.bincount(y).argmax()

        left_mask = X[:, best_feature] < best_threshold
        right_mask = ~left_mask

        if not np.any(left_mask) or not np.any(right_mask):
            return np.bincount(y).argmax()
        
        left_subtree = self._build_tree(X[left_mask], y[left_mask], depth + 1)
        right_subtree = self._build_tree(X[right_mask], y[right_mask], depth + 1)

        return DecisionNode(
            feature=best_feature, 
            threshold=best_threshold, 
            left=left_subtree, 
            right=right_subtree
        )

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predict target values for input features.
        
        Args:
            X (pd.DataFrame): Input feature matrix.
        
        Returns:
            np.ndarray: Predicted target values.
        """
        X_array = X.values
        
        predictions = np.array([self._predict_sample(instance) for instance in X_array])
        
        return self.label_encoder.inverse_transform(predictions)

    def _predict_sample(self, instance: np.ndarray) -> int:
        """
        Predict the target for a single sample by traversing the tree.
        
        Args:
            instance (np.ndarray): Single feature vector.
        
        Returns:
            int: Predicted class.
        """
        node = self.tree
        while isinstance(node, DecisionNode):
            if instance[node.feature] < node.threshold:
                node = node.left
            else:
                node = node.right
        return node