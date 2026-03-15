# Clifford Attention: Geometric Product Attention for Transformers

A PyTorch implementation of a drop-in Transformer attention layer using the **Clifford geometric product** instead of the standard dot product. Built to test whether richer algebraic structure improves attention on geometric tasks.

---

## Motivation

Standard scaled dot-product attention computes token similarity as:

$$\text{score}(Q, K) = \frac{QK^T}{\sqrt{d_k}}$$

The dot product $u \cdot v$ is **symmetric**: $u \cdot v = v \cdot u$. This means attention scores are invariant to the order of $Q$ and $K$, and cannot encode directional or orientational relationships between tokens.

The **Clifford geometric product** decomposes as:

$$uv = \underbrace{u \cdot v}_{\text{scalar (symmetric)}} + \underbrace{u \wedge v}_{\text{bivector (antisymmetric)}}$$

The bivector $u \wedge v = -v \wedge u$ encodes the **oriented plane** spanned by $u$ and $v$ — information that the dot product discards entirely. This project asks: does preserving this structure improve learning on geometric tasks?

---

## Algebra

We work in $\mathcal{Cl}(3,0)$, the Clifford algebra over $\mathbb{R}^3$ with positive-definite signature. The basis is:

$$\{1,\ e_1, e_2, e_3,\ e_{12}, e_{13}, e_{23},\ e_{123}\} \quad (2^3 = 8 \text{ components})$$

Key relations:

$$e_i e_i = +1, \qquad e_i e_j = -e_j e_i \text{ for } i \neq j$$

Each token embedding is reshaped into $n$ **multivectors** of dimension 8. The attention score between two tokens is the scalar part of their geometric product, summed over all multivectors per head:

$$\text{score}(Q, K) = \sum_{i=1}^{n} \langle Q_i K_i^{\dagger} \rangle_0$$

where $K^{\dagger}$ is the **reverse** of $K$ (grade-2 and grade-3 components negated), and $\langle \cdot \rangle_0$ denotes the scalar part.

---

## Implementation

### Requirements

- `d_model` divisible by `num_heads x 8`
- Verified Cl(3,0) product table (hardcoded from `clifford` library output)

### Files
```
clifford_layer.py    # CliffordAttention layer + geometric product
dataset.py           # Synthetic geometric datasets
models.py            # Standard vs Clifford transformer models
train.py             # Training loop
benchmark.py         # Speed comparison
verify_table.py      # Cl(3,0) product table verification
plot_results.py      # Result curves
```

### Drop-in Usage
```python
from clifford_layer import CliffordAttention

# Replaces nn.MultiheadAttention
# Requires d_model % (num_heads * 8) == 0
attn = CliffordAttention(d_model=64, num_heads=1)
out = attn(x)  # x: (B, T, d_model)
```

---

## Experiments

Both models use **identical parameter counts** (~42k) and architectures. The only difference is the attention mechanism. Each input vector is passed as a **separate token** to force attention to carry the geometric signal.

### Task 1 - Rotation Axis Prediction

**Input:** Two 3D unit vectors $u, v$ as separate tokens
**Target:** $u \times v$ (cross product)
**Loss:** MSE

**Why dot-product attention structurally fails:**
$u \times v = \star(u \wedge v)$ is the Hodge dual of the bivector. Since dot-product attention computes only $u \cdot v$ (symmetric), it cannot represent the antisymmetric cross product. Clifford attention retains $u \wedge v$ directly.

| Epoch | Standard MSE | Clifford MSE |
|-------|-------------|--------------|
| 1     | 0.00136     | 0.00125      |
| 10    | 0.00055     | 0.00044      |
| 20    | 0.00039     | 0.00046      |
| 30    | 0.00038     | 0.00035      |
| 40    | 0.00035     | 0.00034      |

Clifford attention converges to consistently lower MSE from epoch 5 onward.

### Task 2 - Orientation Classification (CW vs CCW)

**Input:** Three 3D points $A, B, C$ as separate tokens
**Target:** Sign of det([B-A, C-A]) - binary classification
**Loss:** Cross-entropy

Both models reach ~99% accuracy at this scale. Clifford shows marginally lower loss in later epochs.

---

## Benchmark
```
Config: B=4, T=64, d_model=256, heads=4
--------------------------------------------------
Standard Attention (MHA)       0.160 ms/iter
Clifford Attention             2.144 ms/iter
Slowdown: ~13.4x
```

Clifford attention is slower due to no native CUDA kernel.

---

## Setup
```bash
conda activate your_env
pip install clifford matplotlib
python verify_table.py
python train.py
python plot_results.py
```

---

## Limitations

- No custom CUDA kernel -- 13x slower than standard MHA
- Orientation win not decisive at this scale
- Synthetic tasks only -- no real dataset evaluation
- Single random seed

---

## References

- Brandstetter et al. (2022). Clifford Neural Layers for PDE Dynamics. ICLR 2023.
- Ruhe et al. (2023). Clifford Group Equivariant Neural Networks. NeurIPS 2023.
- Vaswani et al. (2017). Attention Is All You Need. NeurIPS 2017.

---

## Author

Freshman ECE Honors student, UT Austin.
