# Full Name: Nitzan Monfred | ID: 316056126
# Full Name: Nicole Frumkin | ID: 211615372
# Full Name: Alona Gertskin | ID: 207787540

import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from collections import Counter
from graphviz import Digraph


def entropy(y):
    """
    Calculate entropy: val(q) = -q*log₂(q) - (1-q)*log₂(1-q)
    Where q = |{(x,y) ∈ S:y = 1}| / |S|
    """
    counts = Counter(y)  # Count positive/negative examples
    total = len(y)       # |S|
    return -sum((count/total) * np.log2(count/total) for count in counts.values())


def information_gain(X_column, y, threshold):
    """
    Calculate information gain for a given feature and threshold
    Information Gain = parent_entropy - weighted_child_entropy
    """
    left_mask = X_column <= threshold
    right_mask = X_column > threshold

    # Skip if split doesn't separate data
    if len(y[left_mask]) == 0 or len(y[right_mask]) == 0:
        return 0

    parent_entropy = entropy(y)                     # val(q)
    n = len(y)
    left_entropy = entropy(y[left_mask])            # val(qp,0) 
    right_entropy = entropy(y[right_mask])          # val(qp,1)
    
    # Weighted entropy: (1-fp)*val(qp,0) + fp*val(qp,1)
    child_entropy = (len(y[left_mask]) / n) * left_entropy + \
                    (len(y[right_mask]) / n) * right_entropy
    
    return parent_entropy - child_entropy           # Information gain


class Node:
    """Decision tree node"""
    def __init__(self, feature=None, threshold=None, left=None, right=None, *, value=None):
        self.feature = feature      # Feature index for splitting
        self.threshold = threshold  # Threshold value for splitting
        self.left = left           # Left child (feature <= threshold)
        self.right = right         # Right child (feature > threshold)
        self.value = value         # Leaf node prediction

    def is_leaf(self):
        """Check if node is a leaf"""
        return self.value is not None


class ID3DecisionTree:
    """ID3 Decision Tree implementation"""
    
    def __init__(self, max_depth=None):
        self.max_depth = max_depth
        self.root = None

    def fit(self, X, y):
        """Train the decision tree"""
        self.root = self._build_tree(X, y)

    def _build_tree(self, X, y, depth=0):
        """
        Recursively build the decision tree using ID3 algorithm
        """
        # Base case 1: Pure node (all examples have same label)
        if len(set(y)) == 1:
            return Node(value=y[0])
        
        # Base case 2: Maximum depth reached
        if self.max_depth and depth >= self.max_depth:
            most_common = Counter(y).most_common(1)[0][0]
            return Node(value=most_common)

        # Find best feature and threshold to split on
        best_gain = 0
        best_feat = None
        best_thresh = None
        
        # Choose p̂ = arg min over p∈P of weighted entropy
        for feature in range(X.shape[1]):                    # Iterate over predicates P
            thresholds = np.unique(X[:, feature])            # Different threshold values
            
            # Optimize for continuous features with many unique values
            if len(thresholds) > 10:
                thresholds = np.percentile(X[:, feature], np.linspace(10, 90, 9))
                
            for threshold in thresholds:
                gain = information_gain(X[:, feature], y, threshold) 
                if gain > best_gain:                         # Maximize gain = minimize weighted entropy
                    best_gain = gain
                    best_feat = feature
                    best_thresh = threshold

        # Base case 3: No information gain
        if best_gain == 0:
            most_common = Counter(y).most_common(1)[0][0]
            return Node(value=most_common)

        # Split dataset based on best feature and threshold
        left_mask = X[:, best_feat] <= best_thresh           # p(x) = 1 condition
        right_mask = ~left_mask                              # p(x) = 0 condition
        
        # Recursively build subtrees
        # T0 = ID3(S\Sp, P\{p̂}), T1 = ID3(Sp, P\{p̂})
        left_subtree = self._build_tree(X[left_mask], y[left_mask], depth + 1)
        right_subtree = self._build_tree(X[right_mask], y[right_mask], depth + 1)

        return Node(feature=best_feat, threshold=best_thresh, 
                   left=left_subtree, right=right_subtree)

    def predict(self, X):
        """Make predictions for multiple samples"""
        return np.array([self._predict_sample(x, self.root) for x in X])

    def _predict_sample(self, x, node):
        """Make prediction for a single sample"""
        while not node.is_leaf():
            if x[node.feature] <= node.threshold:
                node = node.left
            else:
                node = node.right
        return node.value

    def print_tree(self, node=None, depth=0):
        """Print tree structure in text format"""
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


def evaluate_model():
    """Evaluate ID3 decision tree on breast cancer dataset"""
    # Load and split dataset
    data = load_breast_cancer()
    X, y = data.data, data.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train decision tree
    tree = ID3DecisionTree(max_depth=8)
    tree.fit(X_train, y_train)
    
    # Evaluate performance
    train_predictions = tree.predict(X_train)
    test_predictions = tree.predict(X_test)
    
    train_accuracy = np.mean(train_predictions == y_train)
    test_accuracy = np.mean(test_predictions == y_test)
    
    # Display results
    print(f"Train Accuracy: {train_accuracy * 100:.2f}%")
    print(f"Test Accuracy: {test_accuracy * 100:.2f}%")

    # Visualize the tree
    try:
        dot = visualize_tree(tree, feature_names=data.feature_names)
        dot.render("id3_tree", format="png", cleanup=True)
        print("Decision tree saved as id3_tree.png")
    except:
        print("Graphviz visualization failed - check installation")
        print("\nTree structure:")
        tree.print_tree()


def visualize_tree(tree: ID3DecisionTree, feature_names=None):
    """Create visual representation of the decision tree using Graphviz"""
    dot = Digraph()
    node_id = 0

    def add_nodes_edges(node, parent=None, edge_label=None):
        nonlocal node_id
        curr_id = str(node_id)
        node_id += 1

        # Create node label
        if node.is_leaf():
            label = f"{node.value}"
        else:
            feat_name = feature_names[node.feature] if feature_names is not None else f"X{node.feature}"
            label = f"{feat_name} ≤ {node.threshold:.2f}"

        # Add node to graph
        dot.node(
            curr_id,
            label,
            fontname="Arial",
            fontsize="17",
            style="rounded",
            shape="box",
        )

        # Add edge from parent
        if parent is not None:
            dot.edge(
                parent,
                curr_id,
                label=edge_label,
                fontname="Arial",
                fontsize="16"
            )

        # Recursively add children
        if not node.is_leaf():
            add_nodes_edges(node.left, curr_id, "True")
            add_nodes_edges(node.right, curr_id, "False")

    add_nodes_edges(tree.root)
    return dot


if __name__ == "__main__":
    evaluate_model()