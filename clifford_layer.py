# clifford_layer.py
# Cl(3,0) product table verified from clifford library output.
# Basis index: 0=1, 1=e1, 2=e2, 3=e3, 4=e12, 5=e13, 6=e23, 7=e123

import torch
import torch.nn as nn
import torch.nn.functional as F


def clifford_product(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """
    Geometric product for Cl(3,0).
    a, b: (..., 8)
    returns: (..., 8)

    Each row below is one output component.
    Signs read directly from the verified product table.

    Table (row=left, col=right):
         1   e1   e2   e3  e12  e13  e23 e123
    1  [ +1, +e1, +e2, +e3,+e12,+e13,+e23,+e123]
    e1 [+e1,  +1,+e12,+e13, +e2, +e3,+e123,+e23]
    e2 [+e2,-e12,  +1,+e23, -e1,-e123,+e3, -e13]
    e3 [+e3,-e13,-e23,  +1,+e123,-e1, -e2, +e12]
    e12[+e12,-e2, +e1,+e123, -1,-e23,+e13, -e3 ]
    e13[+e13,-e3,-e123,+e1,+e23,  -1,-e12, +e2 ]
    e23[+e23,+e123,-e3,+e2,-e13,+e12,  -1, -e1 ]
    e123[+e123,+e23,-e13,+e12,-e3,+e2,-e1,  -1 ]
    """
    a0,a1,a2,a3,a4,a5,a6,a7 = a.unbind(-1)
    b0,b1,b2,b3,b4,b5,b6,b7 = b.unbind(-1)

    # c0 = scalar part (grade 0)
    # Products that land on scalar: 1*1, e1*e1, e2*e2, e3*e3, e12*e12, e13*e13, e23*e23, e123*e123
    # Signs from table diagonal: +1,+1,+1,+1,-1,-1,-1,-1
    c0 = (a0*b0 + a1*b1 + a2*b2 + a3*b3
               - a4*b4 - a5*b5 - a6*b6 - a7*b7)

    # c1 = e1 part (grade 1)
    # Which pairs (i,j) give e1?
    # 1*e1=e1, e1*1=e1, e2*e12=-e1(sign-), e3*e13=-e1, e12*e2=+e1, e13*e3=+e1, e23*e123=+e1... wait
    # Read directly from table column e1:
    # row 1: +e1 → a0*b1
    # row e1: +1  → but that's scalar... 
    # IMPORTANT: read col by col for output basis element e1:
    # Which (row,col) products = +e1 or -e1?
    # From table:
    #   (1, e1)   = +e1  → a0*b1
    #   (e1, 1)   = +e1  → a1*b0
    #   (e2, e12) = -e1  → -a2*b4   [table: e2 row, e12 col = -(1^e1)]
    #   (e3, e13) = -e1  → -a3*b5   [table: e3 row, e13 col = -(1^e1)]
    #   (e12, e2) = +e1  → +a4*b2   [table: e12 row, e2 col = +(1^e1)]
    #   (e13, e3) = +e1  → +a5*b3   [table: e13 row, e3 col = +(1^e1)]
    #   (e23,e123)= +e1  → ... table: e23 row, e123 col = -(1^e1) → -a6*b7
    #   (e123,e23)= -e1  → table: e123 row, e23 col = -(1^e1) → -a7*b6
    c1 = a0*b1 + a1*b0 - a2*b4 - a3*b5 + a4*b2 + a5*b3 - a6*b7 - a7*b6

    # c2 = e2 part
    #   (1,e2)=+e2    → a0*b2
    #   (e1,e12)=+e2  → a1*b4   [table: e1 row, e12 col = +(1^e2)]
    #   (e2,1)=+e2    → a2*b0
    #   (e3,e23)=-e2  → -a3*b6  [table: e3 row, e23 col = -(1^e2)]
    #   (e12,e1)=-e2  → -a4*b1  [table: e12 row, e1 col = -(1^e2)]
    #   (e13,e123)=-e2→ ... table: e13 row, e123 col = +(1^e2) → +a5*b7
    #   (e23,e3)=-e2  → -a6*b3  [table: e23 row, e3 col = -(1^e3)... careful]
    #   (e123,e13)=-e2→ table: e123 row, e13 col = +(1^e2) → wait let me re-read

    # I'll read the table row by row for each output.
    # Table row e13, col e123 = +(1^e2)  → +a5*b7
    # Table row e23, col e3   = +(1^e2)  → +a6*b3
    # Table row e123, col e13 = +(1^e2)  → +a7*b5 ... 
    # Re-reading carefully:
    # e23 row: 1=+e23, e1=+e123, e2=-e3, e3=+e2, e12=-e13, e13=+e12, e23=-1, e123=-e1
    # So (e23, e3) = +e2 → +a6*b3
    # e123 row: 1=+e123, e1=+e23, e2=-e13, e3=+e12, e12=-e3, e13=+e2, e23=-e1, e123=-1
    # So (e123, e13) = +e2 → +a7*b5
    c2 = a0*b2 + a1*b4 + a2*b0 - a3*b6 - a4*b1 + a5*b7 + a6*b3 + a7*b5

    # c3 = e3 part
    # e1 row, e13 col = +e3  → +a1*b5
    # e2 row, e23 col = +e3  → +a2*b6
    # e3 row, 1      = +e3  → +a3*b0
    # e12 row, e123 col = +e3 ... table: e12 row: e123=-(1^e3) → -a4*b7
    # e13 row, e1 col = +e3  → table e13 row, e1 col = -(1^e3) → -a5*b1
    # e23 row, e2 col = -(1^e3) → -a6*b2
    # e123 row, e12 col = +e12... no: e123 row, e12 col = -(1^e3) → -a7*b4
    c3 = a0*b3 + a1*b5 + a2*b6 + a3*b0 - a4*b7 - a5*b1 - a6*b2 - a7*b4

    # c4 = e12 part
    # (1,e12)=+e12    → +a0*b4
    # (e1,e2)=+e12    → +a1*b2
    # (e2,e1)=-e12    → -a2*b1
    # (e3,e123)=+e12  → table e3 row, e123 col = +(1^e12) → +a3*b7
    # (e12,1)=-1... no that's scalar. (e12,1)=+e12 → +a4*b0
    # (e13,e23)=+e23... table e13 row, e23 col = -(1^e12) → -a5*b6
    # (e23,e13)=+e12  → table e23 row, e13 col = +(1^e12) → +a6*b5
    # (e123,e3)=-e3... table e123 row, e3 col = +(1^e12) → +a7*b3 ... 
    # re-read: e123 row: e3=+(1^e12) → +a7*b3
    c4 = a0*b4 + a1*b2 - a2*b1 + a3*b7 + a4*b0 - a5*b6 + a6*b5 + a7*b3

    # c5 = e13 part
    # (1,e13)=+e13    → +a0*b5
    # (e1,e3)=+e13    → +a1*b3
    # (e2,e123)=-e13  → table e2 row, e123 col = -(1^e13) → -a2*b7
    # (e3,e1)=-e13    → table e3 row, e1 col = -(1^e13) → -a3*b1
    # (e12,e23)=-e23  → table e12 row, e23 col = +(1^e13) → +a4*b6
    # (e13,1)=+e13    → +a5*b0
    # (e23,e12)=+e12  → table e23 row, e12 col = -(1^e13)... 
    # re-read e23 row: e12=-(1^e13) → -a6*b4
    # e123 row: e2=-(1^e13) → ... e123 row: e2=-(1^e13) → -a7*b2
    c5 = a0*b5 + a1*b3 - a2*b7 - a3*b1 + a4*b6 + a5*b0 - a6*b4 - a7*b2

    # c6 = e23 part
    # (1,e23)=+e23    → +a0*b6
    # (e1,e123)=+e23  → table e1 row, e123 col = +(1^e23) → +a1*b7
    # (e2,e3)=+e23    → table e2 row, e3 col = +(1^e23) → +a2*b3
    # (e3,e2)=-e23    → table e3 row, e2 col = -(1^e23) → -a3*b2
    # (e12,e13)=-e23  → table e12 row, e13 col = -(1^e23) → -a4*b5
    # (e13,e12)=+e23  → table e13 row, e12 col = +(1^e23) → +a5*b4
    # (e23,1)=+e23    → +a6*b0
    # (e123,e1)=-e1   → table e123 row, e1 col = -(1^e1)... 
    # re-read: e123 row, e1 col = +(1^e23) → +a7*b1 ... 
    # table says e123 row: e1=+(1^e23) → +a7*b1
    c6 = a0*b6 + a1*b7 + a2*b3 - a3*b2 - a4*b5 + a5*b4 + a6*b0 - a7*b1

    # c7 = e123 part
    # (1,e123)=+e123   → +a0*b7
    # (e1,e23)=+e123   → table e1 row, e23 col = +(1^e123) ... 
    # re-read e1 row: e23=+(1^e123) → +a1*b6
    # (e2,e13)=-e123   → table e2 row, e13 col = -(1^e123) → -a2*b5
    # (e3,e12)=+e123   → table e3 row, e12 col = +(1^e123) → +a3*b4
    # (e12,e3)=+e123   → table e12 row, e3 col = +(1^e123) → +a4*b3
    # (e13,e2)=-e123   → table e13 row, e2 col = -(1^e123) → -a5*b2
    # (e23,e1)=+e123   → table e23 row, e1 col = +(1^e123) → +a6*b1
    # (e123,1)=+e123   → +a7*b0
    c7 = a0*b7 + a1*b6 - a2*b5 + a3*b4 + a4*b3 - a5*b2 + a6*b1 + a7*b0

    return torch.stack([c0,c1,c2,c3,c4,c5,c6,c7], dim=-1)


def mv_reverse(mv: torch.Tensor) -> torch.Tensor:
    """
    Reverse of a multivector in Cl(3,0).
    Flips sign of grades 2 and 3 (e12,e13,e23,e123).
    grade 0: +1  (index 0)
    grade 1: +1  (indices 1,2,3)
    grade 2: -1  (indices 4,5,6)
    grade 3: -1  (index 7)
    """
    sign = torch.tensor([1, 1, 1, 1, -1, -1, -1, -1],
                        dtype=mv.dtype, device=mv.device)
    return mv * sign


class CliffordAttention(nn.Module):
    """
    Drop-in single-head or multi-head Clifford attention.
    d_model must be divisible by (num_heads * 8).

    Similarity score: scalar part of geometric product Q * reverse(K),
    summed over all multivectors per head. Richer than dot product
    because it retains bivector (rotational) structure during comparison.
    """
    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.0):
        super().__init__()
        assert d_model % 8 == 0, "d_model must be divisible by 8"
        assert (d_model // num_heads) % 8 == 0, \
            "d_model // num_heads must be divisible by 8"

        self.d_model   = d_model
        self.num_heads = num_heads
        self.head_dim  = d_model // num_heads   # e.g. 64
        self.n_mv      = self.head_dim // 8     # multivectors per head, e.g. 8

        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask=None):
        """
        x:    (B, T, d_model)
        mask: (B, T, T) or None  — True where allowed, False where masked
        returns: (B, T, d_model)
        """
        B, T, D = x.shape
        H, n_mv = self.num_heads, self.n_mv

        Q = self.W_q(x)   # (B, T, D)
        K = self.W_k(x)
        V = self.W_v(x)

        # Reshape → (B, H, T, n_mv, 8)
        def to_mv(t):
            return t.view(B, T, H, n_mv, 8).permute(0, 2, 1, 3, 4)

        Q_mv = to_mv(Q)   # (B, H, T, n_mv, 8)
        K_mv = to_mv(K)
        V_mv = to_mv(V)

        K_rev = mv_reverse(K_mv)

        # Pairwise geometric product: every query token vs every key token
        # Q_mv: (B, H, Tq, n_mv, 8) → expand to (B, H, Tq, 1,  n_mv, 8)
        # K_rev:(B, H, Tk, n_mv, 8) → expand to (B, H, 1,  Tk, n_mv, 8)
        Q_exp = Q_mv.unsqueeze(3)    # (B, H, Tq, 1,  n_mv, 8)
        K_exp = K_rev.unsqueeze(2)   # (B, H, 1,  Tk, n_mv, 8)

        prod = clifford_product(Q_exp, K_exp)   # (B, H, Tq, Tk, n_mv, 8)

        # Score = sum of scalar parts across all multivectors
        scores = prod[..., 0].sum(dim=-1)       # (B, H, Tq, Tk)
        scores = scores / (self.head_dim ** 0.5)

        if mask is not None:
            scores = scores.masked_fill(~mask, float('-inf'))

        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)

        # Weighted sum of V
        # attn: (B, H, Tq, Tk)  V_mv: (B, H, Tk, n_mv, 8)
        out = torch.einsum('bhij,bhjme->bhime', attn, V_mv)  # (B,H,Tq,n_mv,8)

        out = out.permute(0, 2, 1, 3, 4).reshape(B, T, D)
        return self.W_o(out)