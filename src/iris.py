import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

# Load the Iris dataset
iris = load_iris()
X = iris.data  # Features: sepal length, sepal width, petal length, petal width
y = iris.target  # Labels: 0=Setosa, 1=Versicolor, 2=Virginica

print(X)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Define the ID3 class as previously provided
class Node:
    def __init__(self, feature=None, value=None, results=None, true_branch=None, false_branch=None):
        self.feature = feature
        self.value = value
        self.results = results
        self.true_branch = true_branch
        self.false_branch = false_branch

class ID3:
    def __init__(self):
        self.root = None

    def _entropy(self, y):
        counts = np.bincount(y) if isinstance(y[0], int) else {label: list(y).count(label) for label in set(y)}
        probabilities = [count / len(y) for count in counts.values()] if not isinstance(y[0], int) else counts / len(y)
        return -np.sum([p * np.log2(p) for p in probabilities if p > 0])

    def _split_data(self, X, y, feature, value):
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

    def _best_split(self, X, y):
        best_gain = 0
        best_criteria = None
        best_sets = None
        n_features = X.shape[1]
        current_entropy = self._entropy(y)

        for feature in range(n_features):
            feature_values = set(X[:, feature]) if not isinstance(X[0, feature], (int, float)) else np.unique(X[:, feature])

            for value in feature_values:
                true_X, true_y, false_X, false_y = self._split_data(X, y, feature, value)

                if len(true_X) == 0 or len(false_X) == 0:
                    continue

                true_entropy = self._entropy(true_y)
                false_entropy = self._entropy(false_y)
                p = len(true_y) / len(y)
                gain = current_entropy - p * true_entropy - (1 - p) * false_entropy

                if gain > best_gain:
                    best_gain = gain
                    best_criteria = (feature, value)
                    best_sets = (true_X, true_y, false_X, false_y)

        return best_gain, best_criteria, best_sets

    def _build_tree(self, X, y):
        X = X.to_numpy() if hasattr(X, "to_numpy") else X
        y = y.to_numpy() if hasattr(y, "to_numpy") else y
        if len(set(y)) == 1:
            return Node(results=y[0])

        best_gain, best_criteria, best_sets = self._best_split(X, y)

        if best_gain == 0:
            majority_class = np.bincount(y).argmax()
            return Node(results=majority_class)

        true_branch = self._build_tree(best_sets[0], best_sets[1])
        false_branch = self._build_tree(best_sets[2], best_sets[3])

        return Node(feature=best_criteria[0], value=best_criteria[1], true_branch=true_branch, false_branch=false_branch)

    def fit(self, X, y):
        self.root = self._build_tree(X, y)

    def predict_sample(self, node, sample):
        if node.results is not None:
            return node.results
        else:
            branch = node.false_branch
            if sample[node.feature] <= node.value if isinstance(node.value, (int, float)) else sample[node.feature] == node.value:
                branch = node.true_branch
            return self.predict_sample(branch, sample)

    def predict(self, X):
        X = X.to_numpy() if hasattr(X, "to_numpy") else X
        return np.array([self.predict_sample(self.root, sample) for sample in X])

# Initialize and train the ID3 model
model = ID3()
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
accuracy = np.mean(y_pred == y_test)
print(f"Accuracy: {accuracy * 100:.2f}%")