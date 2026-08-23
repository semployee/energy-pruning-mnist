# (c) 2026 [Jouw Naam]
# Energy-Aware Pruning – MNIST met volledige kubische energiefunctie
# Nu consistent met de README: E(w) = k1*|w| + k3*|w|^3
# Pruning op basis van E(w) < Kc, en de energie wordt tijdens training gestraft.

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
import time

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
# NETWERK
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
# KUBISCHE ENERGIEFUNCTIE (consistent met README)
# =============================================
k1 = 1.0
k3 = 0.1

def energy_cost(w):
    """E(w) = k1 * |w| + k3 * |w|^3"""
    abs_w = np.abs(w)
    return k1 * abs_w + k3 * (abs_w ** 3)

def energy_derivative(w):
    """dE/dw = k1 * sign(w) + 3*k3 * w^2 * sign(w)"""
    sign_w = np.sign(w)
    return k1 * sign_w + 3.0 * k3 * (w ** 2) * sign_w

def total_energy(W1, W2):
    """Som van E(w) over alle gewichten"""
    return np.sum(energy_cost(W1)) + np.sum(energy_cost(W2))

# =============================================
# TRAINING MET KUBISCHE ENERGIESTRAF & PRUNING
# =============================================
epochs = 30
batch_size = 256
energy_lambda = 0.0002   # strafsterkte λ
Kc = 0.005               # drempel voor pruning: prune if E(w) < Kc

print("🔋 Training met kubische energie-straf en pruning...")

history = {'acc': [], 'pruned': [], 'energy': []}

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
        d_o = error  # softmax afgeleide
        d_h = np.dot(d_o, W2.T) * h * (1 - h)

        # Bereken de afgeleide van de energiefunctie voor W1 en W2
        dE_dW1 = energy_derivative(W1)
        dE_dW2 = energy_derivative(W2)

        # Gradientupdate met kubische energiestraf
        W2 -= learning_rate * (np.dot(h.T, d_o) / batch_size + energy_lambda * dE_dW2)
        b2 -= learning_rate * np.sum(d_o, axis=0) / batch_size
        W1 -= learning_rate * (np.dot(Xb.T, d_h) / batch_size + energy_lambda * dE_dW1)
        b1 -= learning_rate * np.sum(d_h, axis=0) / batch_size

    # Pruning: verwijder gewichten met E(w) < Kc
    mask1 = energy_cost(W1) < Kc
    W1[mask1] = 0.0
    mask2 = energy_cost(W2) < Kc
    W2[mask2] = 0.0

    # Metingen
    acc = accuracy(X_test, y_test, W1, b1, W2, b2)
    pruned = np.sum(W1 == 0) + np.sum(W2 == 0)
    energy = total_energy(W1, W2)  # totale energie volgens E(w)

    history['acc'].append(acc)
    history['pruned'].append(pruned)
    history['energy'].append(energy)

    print(f"Epoch {epoch+1:2d} | Acc: {acc:.4f} | Pruned: {pruned} | Energy: {energy:.4f}")

print("✅ Training voltooid!")

# =============================================
# SPARSE IMPLEMENTATIE (alleen niet-nul gewichten)
# =============================================
print("\n⚡ Sparse implementatie...")

sparse_W1 = [(i, j, W1[i, j]) for i in range(W1.shape[0]) for j in range(W1.shape[1]) if W1[i, j] != 0]
sparse_W2 = [(i, j, W2[i, j]) for i in range(W2.shape[0]) for j in range(W2.shape[1]) if W2[i, j] != 0]

print(f"🔹 Sparse W1: {len(sparse_W1)}/{W1.size} verbindingen")
print(f"🔹 Sparse W2: {len(sparse_W2)}/{W2.size} verbindingen")

def softmax_1d(x):
    e_x = np.exp(x - np.max(x))
    return e_x / np.sum(e_x)

def sparse_predict_1d(x, sparse_W1, b1, sparse_W2, b2):
    h = np.zeros(len(b1))
    for i, j, val in sparse_W1:
        h[j] += x[i] * val
    h = sigmoid(h + b1)

    o = np.zeros(len(b2))
    for i, j, val in sparse_W2:
        o[j] += h[i] * val
    o = softmax_1d(o + b2)
    return o

print("\n🔬 Testen van sparse implementatie op 1000 test samples...")
correct = 0
start = time.time()
for i in range(1000):
    pred = sparse_predict_1d(X_test[i], sparse_W1, b1, sparse_W2, b2)
    if np.argmax(pred) == y_test[i]:
        correct += 1
duration = time.time() - start
print(f"✅ Sparse acc: {correct/10:.1f}% (1000 samples)")
print(f"⏱️  Tijd voor 1000 samples: {duration*1000:.2f} ms")

# =============================================
# BESPARING (FLOPs schatting)
# =============================================
original_ops = W1.size + W2.size
sparse_ops = len(sparse_W1) + len(sparse_W2)
print(f"\n📊 Energiebesparing rapport:")
print(f"   Origineel: {original_ops} bewerkingen per voorspelling")
print(f"   Sparse:    {sparse_ops} bewerkingen per voorspelling")
print(f"   Besparing: {((1 - sparse_ops/original_ops)*100):.1f}% minder rekenwerk!")

# =============================================
# PLOT
# =============================================
fig, axs = plt.subplots(3, 1, figsize=(10, 10))
axs[0].plot(history['acc'], label='Nauwkeurigheid')
axs[0].set_ylabel('Acc')
axs[0].legend()
axs[0].grid(True)

axs[1].plot(history['pruned'], label='Aantal geprunt', color='orange')
axs[1].set_ylabel('Aantal nullen')
axs[1].legend()
axs[1].grid(True)

axs[2].plot(history['energy'], label='Energie (E_tot)', color='green')
axs[2].set_ylabel('Energie')
axs[2].set_xlabel('Epoch')
axs[2].legend()
axs[2].grid(True)

plt.tight_layout()
plt.savefig('mnist_pruning_result.png', dpi=150)
plt.show()
print("\n✅ Klaar! De grafiek is opgeslagen als 'mnist_pruning_result.png'.")
