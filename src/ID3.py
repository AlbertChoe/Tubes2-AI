import numpy as np
import pandas as pd
from typing import Any, Dict

class ID3:
    def __init__(self):
        self.tree: Dict[Any, Any] = None

    def fit(self, X_train: pd.DataFrame, y_train: pd.Series) -> None:
        self.tree = self._id3(X_train, y_train, list(X_train.columns))

    def predict(self, X_test: pd.DataFrame) -> list:
        return [self._predict(x) for _, x in X_test.iterrows()]

    def _id3(self, X: pd.DataFrame, y: pd.Series, features: list) -> Any:
        if len(set(y)) == 1:
            return y.iloc[0]
        if len(features) == 0:
            return self._most_common_label(y)
        
        best_feature = self._best_feature(X, y, features)
        tree: Dict[Any, Any] = {best_feature: {}}
        
        feature_values = set(X[best_feature])
        for value in feature_values:
            subset_X = X[X[best_feature] == value]
            subset_y = y[X[best_feature] == value]
            subset_features = [f for f in features if f != best_feature]
            tree[best_feature][value] = self._id3(subset_X, subset_y, subset_features)
        
        return tree

    def _best_feature(self, X: pd.DataFrame, y: pd.Series, features: list) -> str:
        best_gain = -float('inf')
        best_feature: str = None
        
        for feature in features:
            gain = self._information_gain(X, y, feature)
            if gain > best_gain:
                best_gain = gain
                best_feature = feature
        
        return best_feature

    def _information_gain(self, X: pd.DataFrame, y: pd.Series, feature: str) -> float:
        entropy_before = self._entropy(y)
        feature_values = set(X[feature])
        subsets = [y[X[feature] == value] for value in feature_values]
        
        entropy_after = sum((len(subset) / len(y)) * self._entropy(subset) for subset in subsets)
        
        return entropy_before - entropy_after

    def _entropy(self, y: pd.Series) -> float:
        n = len(y)
        value_counts = y.value_counts()
        entropy = 0
        
        for count in value_counts:
            probability = count / n
            entropy -= probability * np.log2(probability) if probability > 0 else 0
        
        return entropy

    def _most_common_label(self, y: pd.Series) -> Any:
        return y.mode().iloc[0]

    def _predict(self, x: pd.Series) -> Any:
        tree = self.tree
        while isinstance(tree, dict):
            feature = list(tree.keys())[0]
            value = x[feature]
            tree = tree[feature].get(value, None)
            if tree is None:
                return None
        return tree