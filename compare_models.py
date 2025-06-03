import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from collections import Counter

# --- Neural Network Functions ---
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def cross_entropy_loss(y_true, y_pred):
    eps = 1e-12
    y_pred = np.clip(y_pred, eps, 1 - eps)
    return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))

def predict_nn(X, weights, bias):
    return sigmoid(np.dot(X, weights) + bias)

def train_nn(X_train, y_train, X_test, y_test, learning_rate=0.1, epochs=1000):
    input_size = X_train.shape[1]
    a = np.sqrt(input_size)
    weights = np.random.uniform(-1 / a, 1 / a, size=input_size)
    bias = 0.0

    train_loss, test_loss, train_acc, test_acc = [], [], [], []

    for _ in range(epochs):
        y_pred_train = predict_nn(X_train, weights, bias)
        error = y_pred_train - y_train
        dw = np.dot(X_train.T, error) / len(y_train)
        db = np.mean(error)
        weights -= learning_rate * dw
        bias -= learning_rate * db

        y_pred_test = predict_nn(X_test, weights, bias)
        train_loss.append(cross_entropy_loss(y_train, y_pred_train))
        test_loss.append(cross_entropy_loss(y_test, y_pred_test))
        train_acc.append(np.mean((y_pred_train > 0.5) == y_train))
        test_acc.append(np.mean((y_pred_test > 0.5) == y_test))

    return weights, bias, train_loss, test_loss, train_acc, test_acc


# --- Decision Tree Functions ---
def entropy(y):
    counts = Counter(y)
    total = len(y)
    return -sum((c / total) * np.log2(c / total) for c in counts.values())

def information_gain(X_col, y, thresh):
    left = y[X_col <= thresh]
    right = y[X_col > thresh]
    if len(left) == 0 or len(right) == 0:
        return 0
    p = len(y)
    return entropy(y) - (len(left)/p)*entropy(left) - (len(right)/p)*entropy(right)

class Node:
    def __init__(self, feature=None, threshold=None, left=None, right=None, value=None):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value

    def is_leaf(self):
        return self.value is not None

class ID3DecisionTree:
    def __init__(self, max_depth=None):
        self.max_depth = max_depth

    def fit(self, X, y):
        self.root = self._build_tree(X, y)

    def _build_tree(self, X, y, depth=0):
        if len(set(y)) == 1:
            return Node(value=y[0])
        if self.max_depth is not None and depth >= self.max_depth:
            return Node(value=Counter(y).most_common(1)[0][0])

        best_gain, best_feat, best_thresh = 0, None, None
        for feature in range(X.shape[1]):
            thresholds = np.percentile(X[:, feature], np.linspace(10, 90, 9))
            for threshold in thresholds:
                gain = information_gain(X[:, feature], y, threshold)
                if gain > best_gain:
                    best_gain, best_feat, best_thresh = gain, feature, threshold

        if best_gain == 0:
            return Node(value=Counter(y).most_common(1)[0][0])

        left_mask = X[:, best_feat] <= best_thresh
        right_mask = ~left_mask
        left = self._build_tree(X[left_mask], y[left_mask], depth + 1)
        right = self._build_tree(X[right_mask], y[right_mask], depth + 1)
        return Node(best_feat, best_thresh, left, right)

    def predict(self, X):
        return np.array([self._predict_sample(x, self.root) for x in X])

    def _predict_sample(self, x, node):
        while not node.is_leaf():
            node = node.left if x[node.feature] <= node.threshold else node.right
        return node.value


# --- Main Comparison ---
data = load_breast_cancer()
X, y = data.data, data.target
X = StandardScaler().fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Neural Network
weights, bias, train_loss, test_loss, train_acc, test_acc = train_nn(X_train, y_train, X_test, y_test)

# Decision Tree
tree = ID3DecisionTree(max_depth=8)
tree.fit(X_train, y_train)
tree_train_acc = np.mean(tree.predict(X_train) == y_train)
tree_test_acc = np.mean(tree.predict(X_test) == y_test)

# Results
print(f"Neural Network Accuracy, Train: {train_acc[-1]:.4f}")
print(f"Neural Network Accuracy, Test:  {test_acc[-1]:.4f}")
print(f"Decision Tree Accuracy, Train:  {tree_train_acc:.4f}")
print(f"Decision Tree Accuracy, Test:   {tree_test_acc:.4f}")

# Visualization
plt.figure(figsize=(10, 6))
plt.plot(train_acc, label="NN Train Accuracy")
plt.plot(test_acc, label="NN Test Accuracy")
plt.axhline(tree_train_acc, color='green', linestyle='--', label=f"Tree Train Acc: {tree_train_acc:.2f}")
plt.axhline(tree_test_acc, color='red', linestyle='--', label=f"Tree Test Acc: {tree_test_acc:.2f}")
plt.title("Accuracy Comparison Between Neural Network and Decision Tree")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.grid(True)
plt.show()
