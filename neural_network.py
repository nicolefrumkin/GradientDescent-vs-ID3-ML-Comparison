import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_classification
from sklearn.preprocessing import StandardScaler

# Sigmoid function and its derivative
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# Cross-entropy loss
def cross_entropy_loss(y_true, y_pred):
    eps = 1e-12
    y_pred = np.clip(y_pred, eps, 1 - eps)
    return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))

# Predict function
def predict(X, weights, bias):
    z = np.dot(X, weights) + bias
    return sigmoid(z)

# Gradient Descent function
def train(X, y, learning_rate=0.1, epochs=1000, stopping_accuracy=None):
    n_samples, n_features = X.shape
    weights = np.random.randn(n_features) * 0.01  # Normal init
    bias = 0.0

    train_loss_curve = []
    test_loss_curve = []

    # Split train/test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    for epoch in range(epochs):
        y_pred_train = predict(X_train, weights, bias)

        # Compute gradients
        error = y_pred_train - y_train
        dw = np.dot(X_train.T, error) / len(y_train)
        db = np.mean(error)

        # Update weights and bias
        weights -= learning_rate * dw
        bias -= learning_rate * db

        # Track loss
        loss_train = cross_entropy_loss(y_train, y_pred_train)
        y_pred_test = predict(X_test, weights, bias)
        loss_test = cross_entropy_loss(y_test, y_pred_test)
        train_loss_curve.append(loss_train)
        test_loss_curve.append(loss_test)

        # Stopping condition
        acc = np.mean((y_pred_test > 0.5) == y_test)
        if stopping_accuracy and acc >= stopping_accuracy:
            print(f"Stopped at epoch {epoch}, reached test accuracy {acc:.2f}")
            break

    return weights, bias, train_loss_curve, test_loss_curve, acc

# Main
if __name__ == "__main__":
    # Generate synthetic binary classification dataset
    X, y = make_classification(n_samples=1000*2, n_features=10, n_classes=2, random_state=1)
    X = StandardScaler().fit_transform(X)

    weights, bias, train_loss, test_loss, final_acc = train(X, y, learning_rate=0.1, epochs=500, stopping_accuracy=0.92)

    # Plotting
    plt.plot(train_loss, label='Train Loss')
    plt.plot(test_loss, label='Test Loss')
    plt.xlabel("Epochs")
    plt.ylabel("Cross-Entropy Loss")
    plt.title("Loss Curves")
    plt.legend()
    plt.grid(True)
    plt.show()

    # Output results
    print("\nFinal Weights:\n", weights)
    print("Final Bias:\n", bias)
    print("Final Accuracy:", final_acc)
