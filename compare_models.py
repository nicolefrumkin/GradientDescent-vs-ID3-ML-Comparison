# Full Name: Nitzan Monfred | ID: 316056126
# Full Name: Nicole Frumkin | ID: 211615372
# Full Name: Alona Gertskin | ID: 207787540

import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from ID3_alg import ID3DecisionTree  
from neural_network import train as nn_train, predict as nn_predict, sigmoid  

# 1. Load and preprocess the dataset
data = load_breast_cancer()
X, y = data.data, data.target
X = StandardScaler().fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 2. Train Neural Network
print("Training Neural Network...")
nn_weights, nn_bias, _, _, nn_acc = nn_train(X, y, learning_rate=0.1, epochs=1000, stopping_accuracy=None)
y_pred_nn = sigmoid(np.dot(X_test, nn_weights) + nn_bias)
y_pred_nn_class = (y_pred_nn > 0.5).astype(int)
nn_test_accuracy = np.mean(y_pred_nn_class == y_test)

# 3. Train ID3 Decision Tree
print("Training ID3 Decision Tree...")
tree = ID3DecisionTree(max_depth=5)
tree.fit(X_train, y_train)
y_pred_tree = tree.predict(X_test)
tree_test_accuracy = np.mean(y_pred_tree == y_test)

# 4. Print Results
print("\n--- COMPARISON RESULTS ---")
print(f"Neural Network Accuracy: {nn_test_accuracy * 100:.2f}%")
print(f"ID3 Decision Tree Accuracy: {tree_test_accuracy * 100:.2f}%")

