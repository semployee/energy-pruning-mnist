import numpy as np
from sklearn.datasets import fetch_openml

# =============================================
# DATA
# =============================================
print("📥 MNIST laden...")
X, y = fetch_openml('mnist_784', version=1, return_X_y=True, as_frame=False, parser='pandas')
X = X.astype('float32') / 255.0
y = y.astype('int')

X_train, X_test = X[:60000], X[60000:]
y_train, y_test = y[:60000], y[60000:]

def one_hot(y, n=10):
    return np.eye(n)[y]

y_train_oh = one_hot(y_train)

# =============================================
# NETWERK (met softmax)
# =============================================
input_size = 784
hidden_size = 256
output_size = 10
learning_rate = 0.5

np.random.seed(42)
W1 = np.random.randn(input_size, hidden_size) * 0.01
b1 = np.zeros(hidden_size)
W2 = np.random.randn(hidden_size, output_size) * 0.01
b2 = np.zeros(output_size)

def sigmoid(x):
    return 1/(1+np.exp(-x))

def softmax(x):
    e_x = np.exp(x - np.max(x, axis=1, keepdims=True))
    return e_x / np.sum(e_x, axis=1, keepdims=True)

def forward(X, W1, b1, W2, b2):
    h = sigmoid(np.dot(X, W1) + b1)
    o = softmax(np.dot(h, W2) + b2)
    return h, o

def accuracy(X, y, W1, b1, W2, b2):
    _, o = forward(X, W1, b1, W2, b2)
    return np.mean(np.argmax(o, axis=1) == y)

# =============================================
# TRAINING (zonder pruning, zonder energiestraf)
# =============================================
epochs = 30
batch_size = 256

print("🔋 Training...")
for epoch in range(epochs):
    # Shuffle
    idx = np.random.permutation(len(X_train))
    X_train_shuffled = X_train[idx]
    y_train_shuffled = y_train_oh[idx]

    for i in range(0, len(X_train), batch_size):
        Xb = X_train_shuffled[i:i+batch_size]
        yb = y_train_shuffled[i:i+batch_size]

        h, o = forward(Xb, W1, b1, W2, b2)
        error = o - yb
        d_o = error  # voor softmax is de afgeleide (o - y)
        d_h = np.dot(d_o, W2.T) * h * (1 - h)

        W2 -= learning_rate * np.dot(h.T, d_o) / batch_size
        b2 -= learning_rate * np.sum(d_o, axis=0) / batch_size
        W1 -= learning_rate * np.dot(Xb.T, d_h) / batch_size
        b1 -= learning_rate * np.sum(d_h, axis=0) / batch_size

    acc = accuracy(X_test, y_test, W1, b1, W2, b2)
    print(f"Epoch {epoch+1:2d} | Test acc: {acc:.4f}")

print("✅ Training voltooid!")
print(f"Eind nauwkeurigheid: {accuracy(X_test, y_test, W1, b1, W2, b2):.4f}")
