import numpy as np
import pandas as pd
from typing import Dict, Any

class NaiveBayes:
    def fit(self, X_train: pd.DataFrame, y_train: pd.Series) -> None:
        """Fit the model to the training data using Gaussian Naive Bayes."""
        self.X_train = X_train
        self.y_train = y_train
        self.class_probs = self._calculate_class_probabilities(y_train)
        self.feature_params = self._calculate_feature_parameters(X_train, y_train)
    
    def predict(self, X_test: pd.DataFrame) -> list:
        """Predict the class labels for the test data using Gaussian Naive Bayes."""
        if isinstance(X_test, np.ndarray):
            X_test = pd.DataFrame(X_test)
        return [self._predict(x) for _, x in X_test.iterrows()]
    
    def _predict(self, x: pd.Series) -> Any:
        """Predict the class label for a single test point using Gaussian Naive Bayes."""
        posteriors: Dict[Any, float] = {}
        
        for c in self.class_probs:
            prior = np.log(self.class_probs[c])
            likelihood = 0
            for feature in x.index:
                mean, var = self.feature_params[c].get(feature, (0, 1))
                prob = self._gaussian_probability(x[feature], mean, var)
                likelihood += np.log(prob) if prob > 0 else -np.inf
            posteriors[c] = prior + likelihood
        
        return max(posteriors, key=posteriors.get)

    def _calculate_class_probabilities(self, y_train: pd.Series) -> Dict[Any, float]:
        """Calculate the prior probabilities of each class."""
        total_count = len(y_train)
        class_counts = y_train.value_counts()
        return (class_counts / total_count).to_dict()
    
    def _calculate_feature_parameters(self, X_train: pd.DataFrame, y_train: pd.Series) -> Dict[Any, Dict[str, tuple]]:
        """Calculate the mean and variance of features for each class."""
        feature_params: Dict[Any, Dict[str, tuple]] = {}
        
        for c in y_train.unique():
            feature_params[c] = {}
            subset = X_train[y_train == c]
            for feature in X_train.columns:
                mean = subset[feature].mean()
                var = subset[feature].var()
                feature_params[c][feature] = (mean, var)
        
        return feature_params
    
    def _gaussian_probability(self, x: float, mean: float, var: float) -> float:
        """Calculate the Gaussian probability density function for a given value."""
        if var == 0:
            return 1.0 if x == mean else 0.0
        exponent = np.exp(- (x - mean)**2 / (2 * var))
        return (1 / np.sqrt(2 * np.pi * var)) * exponent