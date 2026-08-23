# energy-pruning-mnist
Train a neural network with energy penalties and prune 97% of weights with minimal accuracy loss.
# Energy-Aware Pruning for Neural Networks

This repository demonstrates a method to train neural networks with an **energy penalty** and **pruning**, resulting in models that retain high accuracy while using dramatically fewer operations.

## Key Results (on MNIST)

- Baseline accuracy (full model): **97.2%**
- Pruned accuracy: **95.1%**
- Remaining connections: **5,496** (out of 203,264)
- Energy savings: **97.3% fewer FLOPs**

## Mathematical Foundation

### 1. Energy Cost of a Weight

We define the energy cost of a single weight \( w \) as:

\[
E(w) = \kappa_1 \cdot |w| + \kappa_3 \cdot |w|^3
\]

- \(\kappa_1\): linear cost coefficient (maintenance cost)  
- \(\kappa_3\): cubic cost coefficient (penalty for large weights)

This function expresses that:
- Small weights cost little (linear term).
- Large weights cost disproportionately more (cubic term).
- Zero weights cost nothing.

### 2. Total Network Energy

The total energy of the network is the sum over all weights:

\[
E_{\text{tot}} = \sum_{i,j} E(W_{i,j})
\]

### 3. Training with Energy Penalty

During training, we add this energy as a penalty to the cross‑entropy loss:

\[
\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{CE}} + \lambda \cdot E_{\text{tot}}
\]

The gradient update for a weight becomes:

\[
w \leftarrow w - \eta \left( \frac{\partial \mathcal{L}_{\text{CE}}}{\partial w} + \lambda \cdot \frac{\partial E(w)}{\partial w} \right)
\]

with

\[
\frac{\partial E(w)}{\partial w} = \kappa_1 \cdot \text{sign}(w) + 3\kappa_3 \cdot w^2 \cdot \text{sign}(w)
\]

This drives small weights towards zero while the network still learns the task.

### 4. Pruning Condition

After each training epoch, we apply a hard threshold:

\[
w \leftarrow 0 \quad \text{if} \quad E(w) < K_c
\]

Only weights with \(E(w) \ge K_c\) survive.

### 5. Sparse Inference

After training, only the surviving weights are used for prediction:

\[
\text{FLOPs} = |\mathcal{S}_1| + |\mathcal{S}_2|
\]

where \(\mathcal{S}_1\) and \(\mathcal{S}_2\) are the sets of active weights in layers 1 and 2.

### Parameters Used in This Work

| Parameter | Meaning | Value |
|-----------|---------|-------|
| \(\kappa_1\) | Linear energy cost | 1.0 |
| \(\kappa_3\) | Cubic energy cost | 0.1 |
| \(K_c\) | Pruning threshold | 0.005 |
| \(\lambda\) | Penalty strength | \(2 \times 10^{-4}\) |

---

## Where This Appears in the Code

The implementation in `mnist_pruning_energy.py` follows the math:

| Concept | Code Location |
|---------|---------------|
| Energy function \(E(w)\) | `energy_cost(w, k1=1.0, k3=0.1)` |
| Total energy \(E_{\text{tot}}\) | `energy = np.sum(np.abs(W1)) + np.sum(np.abs(W2))` |
| Energy penalty in loss | `energy_lambda * np.sign(W)` in weight update |
| Pruning condition | `W1[np.abs(W1) < Kc] = 0.0` and same for `W2` |
| Sparse inference | `sparse_predict_1d()` uses only non‑zero weights from `sparse_W1` and `sparse_W2` |
| FLOPs calculation | `original_ops = W1.size + W2.size` and `sparse_ops = len(sparse_W1) + len(sparse_W2)` |

---

## Scripts

- **`mnist_baseline.py`** – trains a standard network without pruning or energy penalty (baseline accuracy ~97%).
- **`mnist_pruning_energy.py`** – trains with energy penalty and pruning, then measures sparse inference speed‑up.

## Requirements

- Python 3.x
- `numpy`, `scikit-learn`, `matplotlib`

Install with:

```bash
pip install numpy scikit-learn matplotlib
```

## Usage

Run the pruning version:

```bash
python mnist_pruning_energy.py
```
The script downloads MNIST, trains the model, applies pruning, shows accuracy and speed‑up, and saves a plot as `mnist_pruning_result.png`.

## File explanation
mnist_pruning3.py is the baseline model
mnist_pruning3_energy.py is the model after pruning (updated by copilot)
mnist_pruning3_energy2.py is the model after pruning (updated by me as suggested by copilot)

Both show a significant increase in energy savings while maintaining functionality therefore accuracy

## License

All rights reserved. You may view the code for personal or educational purposes; using and redistribution requires permission.

## Author

**R.T.Somer** – inspired by natural systems and energy efficiency.  
GitHub: [https://github.com/semployee]
