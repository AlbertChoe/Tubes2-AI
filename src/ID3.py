import numpy as np
from sklearn.preprocessing import LabelEncoder

class DecisionNode:
    def __init__(self, feature=None, threshold=None, left=None, right=None, *, value=None):
        self.feature = feature          
        self.threshold = threshold      
        self.left = left                
        self.right = right              
        self.value = value              

class DecisionTreeID3:
    def __init__(self, max_depth=None, min_samples_split=10, min_information_gain=1e-5):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_information_gain = min_information_gain
        self.tree = None
        self.label_encoder = LabelEncoder()

    def fit(self, X_train, y_train):

        self.features = np.array(X_train.columns)
        X = X_train.values
        y = self.label_encoder.fit_transform(y_train)
        self.tree = self._build_tree(X, y)

    def _entropy(self, y):
        counts = np.bincount(y)
        probabilities = counts / counts.sum()
        return -np.sum(probabilities * np.log2(probabilities + 1e-9))

    def _best_split(self, X, y):
        best_feature = None
        best_threshold = None
        best_info_gain = -1
        parent_entropy = self._entropy(y)
        n_samples, n_features = X.shape

        classes = np.unique(y)
        n_classes = len(classes)

        for feature_idx in range(n_features):
            X_column = X[:, feature_idx]
            sorted_indices = np.argsort(X_column)
            X_sorted = X_column[sorted_indices]
            y_sorted = y[sorted_indices]

            unique_values, unique_indices = np.unique(X_sorted, return_index=True)
            unique_values = unique_values[1:] 
            unique_indices = unique_indices[1:]  

            if len(unique_values) == 0:
                continue

            thresholds = (X_sorted[unique_indices - 1] + X_sorted[unique_indices]) / 2

            num_left = unique_indices
            num_right = n_samples - num_left

            entropies_left = np.zeros(len(thresholds))
            entropies_right = np.zeros(len(thresholds))

            for cls in classes:
                cls_mask = y_sorted == cls
                cls_cumsum = np.cumsum(cls_mask).astype(int)
                cls_total = cls_cumsum[-1]

                cls_left = cls_cumsum[unique_indices - 1]
                cls_right = cls_total - cls_left

                probs_left = np.divide(cls_left, num_left, out=np.zeros_like(cls_left, dtype=float), where=num_left != 0)
                probs_right = np.divide(cls_right, num_right, out=np.zeros_like(cls_right, dtype=float), where=num_right != 0)

                entropies_left -= probs_left * np.log2(probs_left + 1e-9)
                entropies_right -= probs_right * np.log2(probs_right + 1e-9)

            weighted_entropy = (num_left / n_samples) * entropies_left + (num_right / n_samples) * entropies_right
            info_gains = parent_entropy - weighted_entropy

            max_info_gain_idx = np.argmax(info_gains)
            max_info_gain = info_gains[max_info_gain_idx]

            if max_info_gain > best_info_gain:
                best_info_gain = max_info_gain
                best_feature = feature_idx
                best_threshold = thresholds[max_info_gain_idx]

        return best_feature, best_threshold, best_info_gain

    def _build_tree(self, X, y, depth=0):
        num_samples, num_features = X.shape
        num_labels = len(np.unique(y))

        if num_labels == 1:
            return y[0]
        if num_samples < self.min_samples_split:
            return np.bincount(y).argmax()
        if self.max_depth is not None and depth >= self.max_depth:
            return np.bincount(y).argmax()

        best_feature, best_threshold, best_info_gain = self._best_split(X, y)

        if best_info_gain < self.min_information_gain or best_feature is None:
            return np.bincount(y).argmax()

        left_indices = X[:, best_feature] < best_threshold
        right_indices = X[:, best_feature] >= best_threshold

        if np.sum(left_indices) == 0 or np.sum(right_indices) == 0:
            return np.bincount(y).argmax()

        left_X, left_y = X[left_indices], y[left_indices]
        right_X, right_y = X[right_indices], y[right_indices]

        left_subtree = self._build_tree(left_X, left_y, depth + 1)
        right_subtree = self._build_tree(right_X, right_y, depth + 1)

        return DecisionNode(feature=best_feature, threshold=best_threshold, left=left_subtree, right=right_subtree)

    def predict(self, X):
        X = X.values  
        predictions = np.array([self._predict_sample(instance) for instance in X])
        return self.label_encoder.inverse_transform(predictions)

    def _predict_sample(self, instance):
        node = self.tree
        while isinstance(node, DecisionNode):
            if instance[node.feature] < node.threshold:
                node = node.left
            else:
                node = node.right
        return node