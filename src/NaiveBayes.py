import numpy as np
import pandas as pd
from typing import Dict, Any

class NaiveBayes:
    def __init__(self, alpha: float = 1e-10):
        """
        Initialize Gaussian Naive Bayes Classifier
        
        Parameters:
        -----------
        alpha : float, default=1e-10
            Smoothing parameter to handle zero variance
        """
        self.alpha = alpha

    def fit(self, X_train: pd.DataFrame, y_train: pd.Series):
        """
        Fit the Gaussian Naive Bayes model to the training data.
        
        Parameters:
        -----------
        X_train : pd.DataFrame
            Training feature matrix
        y_train : pd.Series
            Training target vector
        
        Returns:
        --------
        self : GaussianNaiveBayes
            Fitted classifier
        """
        X_train = pd.DataFrame(X_train)
        y_train = pd.Series(y_train)
        
        self.X_train = X_train
        self.y_train = y_train
        
        self.classes_ = y_train.unique()
        self.class_probs = self._calculate_class_probabilities(y_train)
        
        self.feature_params = self._calculate_feature_parameters(X_train, y_train)
        
        return self

    def predict(self, X_test: pd.DataFrame) -> np.ndarray:
        """
        Predict class labels for test data.
        
        Parameters:
        -----------
        X_test : pd.DataFrame
            Test feature matrix
        
        Returns:
        --------
        np.ndarray
            Predicted class labels
        """
        X_test = pd.DataFrame(X_test)
        
        return np.array([self._predict(x) for _, x in X_test.iterrows()])

    def _predict(self, x: pd.Series) -> Any:
        """
        Predict class label for a single sample.
        
        Parameters:
        -----------
        x : pd.Series
            Single sample features
        
        Returns:
        --------
        Any
            Predicted class label
        """
        posteriors: Dict[Any, float] = {}
        
        for c in self.classes_:
            prior = np.log(self.class_probs[c])

            likelihood = 0
            for feature in x.index:
                mean, var = self.feature_params[c].get(feature, (0, 1))
                prob = self._gaussian_probability(x[feature], mean, var)
                likelihood += np.log(prob) if prob > 0 else -np.inf
 
            posteriors[c] = prior + likelihood

        return max(posteriors, key=posteriors.get)

    def _calculate_class_probabilities(self, y_train: pd.Series) -> Dict[Any, float]:
        """
        Calculate prior probabilities of each class.
        
        Parameters:
        -----------
        y_train : pd.Series
            Training target vector
        
        Returns:
        --------
        Dict[Any, float]
            Dictionary of class probabilities
        """
        class_counts = y_train.value_counts()
        return (class_counts / len(y_train)).to_dict()

    def _calculate_feature_parameters(self, X_train: pd.DataFrame, y_train: pd.Series) -> Dict[Any, Dict[str, tuple]]:
        """
        Calculate mean and variance of features for each class with smoothing.
        
        Parameters:
        -----------
        X_train : pd.DataFrame
            Training feature matrix
        y_train : pd.Series
            Training target vector
        
        Returns:
        --------
        Dict[Any, Dict[str, tuple]]
            Nested dictionary of feature parameters for each class
        """
        feature_params: Dict[Any, Dict[str, tuple]] = {}
        
        for c in self.classes_:
            feature_params[c] = {}

            subset = X_train[y_train == c]
            
            for feature in X_train.columns:

                mean = subset[feature].mean()
                var = subset[feature].var()
                
                var = var + self.alpha if var > 0 else self.alpha
                
                feature_params[c][feature] = (mean, var)
        
        return feature_params

    def _gaussian_probability(self, x: float, mean: float, var: float) -> float:
        """
        Calculate Gaussian probability density function.
        
        Parameters:
        -----------
        x : float
            Input value
        mean : float
            Mean of the distribution
        var : float
            Variance of the distribution
        
        Returns:
        --------
        float
            Probability density
        """
        exponent = -((x - mean) ** 2) / (2 * var)
        return np.exp(exponent) / np.sqrt(2 * np.pi * var)

    def score(self, X_test: pd.DataFrame, y_test: pd.Series) -> float:
        """
        Calculate the accuracy of the model.
        
        Parameters:
        -----------
        X_test : pd.DataFrame
            Test feature matrix
        y_test : pd.Series
            True labels for test data
        
        Returns:
        --------
        float
            Accuracy score
        """
        predictions = self.predict(X_test)
        return np.mean(predictions == y_test)