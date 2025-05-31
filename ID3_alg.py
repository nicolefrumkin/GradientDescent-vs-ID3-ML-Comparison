# Full Name: Nitzan Monfred | ID: 316056126
# Full Name: Nicole Frumkin | ID: 211615372
# Full Name: Alona Gertskin | ID: 207787540

import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from collections import Counter
from graphviz import Digraph

def entropy(y):
    counts = Counter(y)
    total = len(y)
    return -sum((count/total) * np.log2(count/total) for count in counts.values())

def information_gain(X_column, y, threshold):
    left_mask = X_column <= threshold
    right_mask = X_column > threshold

    if len(y[left_mask]) == 0 or len(y[right_mask]) == 0:
        return 0

    parent_entropy = entropy(y)
    n = len(y)
    left_entropy = entropy(y[left_mask])
    right_entropy = entropy(y[right_mask])

    child_entropy = (len(y[left_mask]) / n) * left_entropy + (len(y[right_mask]) / n) * right_entropy
    return parent_entropy - child_entropy

class Node:
    def __init__(self, feature=None, threshold=None, left=None, right=None, *, value=None):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value  # Leaf node prediction

    def is_leaf(self):
        return self.value is not None

class ID3DecisionTree:
    def __init__(self, max_depth=None):
        self.max_depth = max_depth
        self.root = None

    def fit(self, X, y):
        self.root = self._build_tree(X, y)

    def _build_tree(self, X, y, depth=0):
        if len(set(y)) == 1:
            return Node(value=y[0])
        if self.max_depth and depth >= self.max_depth:
            most_common = Counter(y).most_common(1)[0][0]
            return Node(value=most_common)

        best_gain = 0
        best_feat = None
        best_thresh = None

        for feature in range(X.shape[1]):
            thresholds = np.unique(X[:, feature])
            for threshold in thresholds:
                gain = information_gain(X[:, feature], y, threshold)
                if gain > best_gain:
                    best_gain = gain
                    best_feat = feature
                    best_thresh = threshold

        if best_gain == 0:
            most_common = Counter(y).most_common(1)[0][0]
            return Node(value=most_common)

        left_mask = X[:, best_feat] <= best_thresh
        right_mask = ~left_mask

        left_subtree = self._build_tree(X[left_mask], y[left_mask], depth + 1)
        right_subtree = self._build_tree(X[right_mask], y[right_mask], depth + 1)

        return Node(feature=best_feat, threshold=best_thresh, left=left_subtree, right=right_subtree)

    def predict(self, X):
        return np.array([self._predict_sample(x, self.root) for x in X])

    def _predict_sample(self, x, node):
        while not node.is_leaf():
            if x[node.feature] <= node.threshold:
                node = node.left
            else:
                node = node.right
        return node.value

    def print_tree(self, node=None, depth=0):
        if node is None:
            node = self.root
        indent = "  " * depth
        if node.is_leaf():
            print(f"{indent}Predict: {node.value}")
        else:
            print(f"{indent}Feature {node.feature} <= {node.threshold}")
            self.print_tree(node.left, depth + 1)
            print(f"{indent}Feature {node.feature} > {node.threshold}")
            self.print_tree(node.right, depth + 1)


def visualize_tree(tree: ID3DecisionTree, feature_names=None):
    dot = Digraph()
    node_id = 0

    def add_nodes_edges(node, parent=None, edge_label=None):
        nonlocal node_id
        curr_id = str(node_id)
        node_id += 1

        if node.is_leaf():
            label = f"{node.value}"
        else:
            feat_name = feature_names[node.feature] if feature_names is not None else f"X{node.feature}"
            label = f"{feat_name} ≤ {node.threshold:.2f}"

        dot.node(
            curr_id,
            label,
            fontname="Arial",
            fontsize="17",
            style="rounded",
            shape="box",
        )

        if parent is not None:
            dot.edge(
                parent,
                curr_id,
                label=edge_label,
                fontname="Arial",
                fontsize="16"
            )

        if not node.is_leaf():
            add_nodes_edges(node.left, curr_id, "True")
            add_nodes_edges(node.right, curr_id, "False")

    add_nodes_edges(tree.root)
    return dot

# For comparison with the neural network
def evaluate_model():
    data = load_breast_cancer()
    X, y = data.data, data.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    tree = ID3DecisionTree(max_depth=5)
    tree.fit(X_train, y_train)
    predictions = tree.predict(X_test)
    accuracy = np.mean(predictions == y_test)

    print(f"\nID3 Test Accuracy: {accuracy * 100:.2f}%")
    print("\nDecision Tree Structure:")
    # tree.print_tree()

    # Visualize the tree
    dot = visualize_tree(tree, feature_names=data.feature_names)
    dot.render("id3_tree", format="png", cleanup=True)
    print("Decision tree saved as id3_tree.png")

if __name__ == "__main__":
    evaluate_model()
