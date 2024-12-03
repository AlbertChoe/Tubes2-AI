import numpy as np
import pandas as pd
from typing import Dict, Any

class NaiveBayes:
    def fit(self, X_train: pd.DataFrame, y_train: pd.Series) -> None:
        """Fit the model to the training data."""
        self.X_train = X_train
        self.y_train = y_train
        self.class_probs = self._calculate_class_probabilities(y_train)
        self.feature_probs = self._calculate_feature_probabilities(X_train, y_train)
    
    def predict(self, X_test: pd.DataFrame) -> list:
        """Predict the class labels for the test data."""
        return [self._predict(x) for _, x in X_test.iterrows()]
    
    def _predict(self, x: pd.Series) -> Any:
        """Predict the class label for a single test point."""
        posteriors: Dict[Any, float] = {}
        
        for c in self.class_probs:
            prior = np.log(self.class_probs[c])
            likelihood = 0
            for feature in x.index:
                prob = self.feature_probs[c].get(feature, {}).get(x[feature], 0)
                likelihood += np.log(prob) if prob > 0 else -np.inf
            posteriors[c] = prior + likelihood
        
        return max(posteriors, key=posteriors.get)

    def _calculate_class_probabilities(self, y_train: pd.Series) -> Dict[Any, float]:
        """Calculate the prior probabilities of each class."""
        total_count = len(y_train)
        class_counts = y_train.value_counts()
        return (class_counts / total_count).to_dict()
    
    def _calculate_feature_probabilities(self, X_train: pd.DataFrame, y_train: pd.Series) -> Dict[Any, Dict[str, Dict[Any, float]]]:
        """Calculate the probabilities of features given each class."""
        feature_probs: Dict[Any, Dict[str, Dict[Any, float]]] = {}
        
        for c in y_train.unique():
            feature_probs[c] = {}
            subset = X_train[y_train == c]
            for feature in X_train.columns:
                feature_probs[c][feature] = subset[feature].value_counts() / len(subset)
        
        return feature_probs