# Full Name: Nitzan Monfred | ID: 316056126
# Full Name: Nicole Frumkin | ID: 211615372
# Full Name: Alona Gertskin | ID: 207787540

import numpy as np
import math
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_classification
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_breast_cancer

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# Cross-entropy loss: -ylog(yp)-(1-y)log(1-yp)
def cross_entropy_loss(y_true, y_pred):
    eps = 1e-12
    y_pred = np.clip(y_pred,eps,1-eps) # Avoid log(0)
    # taking mean of the loss across all sampless
    return -np.mean(y_true*np.log(y_pred)+(1-y_true)*np.log(1-y_pred))

# Predict function: z=Xw+b
def predict(X, weights, bias):
    z = np.dot(X, weights)+bias
    return sigmoid(z)

# X - feature matrix
# y - label vector
# learning_rate - step size for weight updates
# epochs - number of iterations for training
# stopping_accuracy - if provided, training stops when test accuracy reaches this value
def train(X, y, learning_rate=0.1, epochs=1000, stopping_accuracy=None):
    input_size = X.shape[1]  # Number of features
    a = np.sqrt(input_size)
    weights = np.random.uniform(-1/a, 1/a, size=input_size)# initialize random weight
    bias = 0.0

    # store the loss for each epoch
    train_loss_curve = []
    test_loss_curve = []

    # Split train/test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    for epoch in range(epochs):
        y_pred_train = predict(X_train, weights, bias) # forward pass

        # Compute gradients
        error = y_pred_train - y_train
        dw = np.dot(X_train.T, error) / len(y_train) # gradient of loss w.r.t weights
        db = np.mean(error) # gradient of loss w.r.t bias

        # Update weights and bias in direction that reduces the loss
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

def visualize_nn(n_features, weights=None, bias=None, train_loss=None, test_loss=None):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))
    fig.subplots_adjust(hspace=0.1)
    
    # --- Neural Network Plot ---
    ax1.axis('off')
    input_y = np.linspace(0, 1, n_features)
    input_x = np.zeros(n_features)
    output_x, output_y = 0.5, 0.5

    # Connections + weights
    for i in range(n_features):
        ax1.plot([input_x[i], output_x], [input_y[i], output_y], 'k')
        if weights is not None:
            dx = output_x - input_x[i]
            dy = output_y - input_y[i]
            mid_x = (input_x[i] + output_x) / 2
            mid_y = (input_y[i] + output_y) / 2
            angle = math.degrees(math.atan2(dy, dx))

            ax1.text(
                mid_x,
                mid_y,
                f"{weights[i]:.2f}",
                fontsize=8,
                color='red',
                ha='center',
                va='center',
                rotation=angle,
                    bbox=dict(facecolor='white', edgecolor='none', boxstyle='round,pad=0.2')
            )

    # Input neurons
    for i in range(n_features):
        ax1.add_patch(plt.Circle((input_x[i], input_y[i]), 0.04, color='skyblue', ec='black', zorder=2))
        ax1.set_aspect('equal')
        ax1.text(input_x[i], input_y[i], f"x{i}", ha='center', va='center', fontsize=9)

    # Output neuron
    ax1.add_patch(plt.Circle((output_x, output_y), 0.05, color='lightgreen', ec='black', zorder=2))
    ax1.set_aspect('equal')
    ax1.text(output_x, output_y, "ŷ", ha='center', va='center', fontsize=10)

    # Bias
    if bias is not None:
        ax1.text(output_x + 0.1, output_y - 0.15, f"bias: {bias:.2f}", ha='center', fontsize=9, color='blue')

    ax1.set_title("Single-Layer Neural Network", fontsize=14)

    # --- Loss Curve Plot ---
    if train_loss is not None and test_loss is not None:
        ax2.plot(train_loss, label='Train Loss', color='blue')
        ax2.plot(test_loss, label='Test Loss', color='orange')
        ax2.set_title("Loss Curve", fontsize=14)
        ax2.set_xlabel("Epochs")
        ax2.set_ylabel("Loss")
        ax2.legend()
        ax2.grid(True)

    plt.show()


# Main
if __name__ == "__main__":
    # Generate synthetic binary classification dataset
    X, y = load_breast_cancer(return_X_y=True)
    X = StandardScaler().fit_transform(X)

    weights, bias, train_loss, test_loss, final_acc = train(X, y, learning_rate=0.1, epochs=1000)
    visualize_nn(
        n_features=X.shape[1],
        weights=weights,
        bias=bias,
        train_loss=train_loss,
        test_loss=test_loss
    )

    # Output results
    np.set_printoptions(precision=3, suppress=True)
    print("\nFinal Weights:\n", weights)
    print("Final Bias:\n", f"{bias:.3f}")
    print("Final Accuracy:", f"{final_acc:.3f}")        
