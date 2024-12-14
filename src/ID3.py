import numpy as np

class Node:
    def __init__(self, feature=None, value=None, results=None, true_branch=None, false_branch=None):
        """
        Represents a single node in the decision tree.
        """
        self.feature = feature
        self.value = value
        self.results = results
        self.true_branch = true_branch
        self.false_branch = false_branch

class ID3:
    def __init__(self):
        self.root = None

    def _entropy(self, y):
        """
        Calculate entropy for the given labels.
        """
        counts = np.bincount(y) if isinstance(y[0], int) else {label: list(y).count(label) for label in set(y)}
        probabilities = [count / len(y) for count in counts.values()] if not isinstance(y[0], int) else counts / len(y)
        return -np.sum([p * np.log2(p) for p in probabilities if p > 0])

    def _split_data(self, X, y, feature, value):
        """
        Split the dataset based on the feature and its value.
        Handles both continuous and categorical features.
        """
        X = X.to_numpy() if hasattr(X, "to_numpy") else X
        y = y.to_numpy() if hasattr(y, "to_numpy") else y

        if isinstance(value, (int, float)):
            true_indices = np.where(X[:, feature] <= value)[0]
            false_indices = np.where(X[:, feature] > value)[0]
        else:
            true_indices = np.where(X[:, feature] == value)[0]
            false_indices = np.where(X[:, feature] != value)[0]

        true_X, true_y = X[true_indices], y[true_indices]
        false_X, false_y = X[false_indices], y[false_indices]
        return true_X, true_y, false_X, false_y

    def _build_tree(self, X, y):
        """
        Build the decision tree recursively.
        """
        X = X.to_numpy() if hasattr(X, "to_numpy") else X
        y = y.to_numpy() if hasattr(y, "to_numpy") else y

        if len(set(y)) == 1:
            return Node(results=y[0])

        best_gain = 0
        best_criteria = None
        best_sets = None
        n_features = X.shape[1]

        current_entropy = self._entropy(y)

        for feature in range(n_features):
            feature_values = set(X[:, feature]) if not isinstance(X[0, feature], (int, float)) else set(np.unique(X[:, feature]))
            for value in feature_values:
                true_X, true_y, false_X, false_y = self._split_data(X, y, feature, value)

                true_entropy = self._entropy(true_y)
                false_entropy = self._entropy(false_y)
                p = len(true_y) / len(y)
                gain = current_entropy - p * true_entropy - (1 - p) * false_entropy
                
                if gain > best_gain:
                    best_gain = gain
                    best_criteria = (feature, value)
                    best_sets = (true_X, true_y, false_X, false_y)

        if best_gain > 0:
            true_branch = self._build_tree(best_sets[0], best_sets[1])
            false_branch = self._build_tree(best_sets[2], best_sets[3])
            return Node(feature=best_criteria[0], value=best_criteria[1], true_branch=true_branch, false_branch=false_branch)

        return Node(results=y[0])

    def fit(self, X, y):
        """
        Fit the decision tree to the training data.
        """
        self.root = self._build_tree(X, y)

    def predict_sample(self, node, sample):
        """
        Predict the label for a single sample.
        """
        if node.results is not None:
            return node.results
        else:
            branch = node.false_branch
            if sample[node.feature] <= node.value if isinstance(node.value, (int, float)) else sample[node.feature] == node.value:
                branch = node.true_branch
            return self.predict_sample(branch, sample)

    def predict(self, X):
        """
        Predict the labels for multiple samples.
        """
        X = X.to_numpy() if hasattr(X, "to_numpy") else X
        return np.array([self.predict_sample(self.root, sample) for sample in X])