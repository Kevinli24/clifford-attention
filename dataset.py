# dataset.py
import torch
from torch.utils.data import Dataset, DataLoader


class RotationAxisDataset(Dataset):
    """
    Input:  (2, 3) — two tokens, each a 3D unit vector
    Output: (3,)   — u x v (cross product)

    With multi-token input, attention between token 0 (u) and token 1 (v)
    must compute their geometric relationship.
    Dot product score: u·v (symmetric, loses sign of u x v)
    Clifford score:    uv = u·v + u∧v (keeps bivector)
    """
    def __init__(self, n_samples=50000, seed=42):
        torch.manual_seed(seed)
        u = torch.randn(n_samples, 3)
        v = torch.randn(n_samples, 3)
        u = u / u.norm(dim=-1, keepdim=True)
        v = v / v.norm(dim=-1, keepdim=True)
        # x: (N, 2, 3) — 2 tokens of dim 3
        self.x = torch.stack([u, v], dim=1).float()
        self.y = torch.linalg.cross(u, v).float()    # (N, 3)

    def __len__(self): return len(self.x)
    def __getitem__(self, idx): return self.x[idx], self.y[idx]


class OrientationDataset(Dataset):
    """
    Input:  (3, 3) - three tokens, each a 3D point
    Output: binary - CW (1) or CCW (0)

    Attention between token pairs (A,B), (A,C), (B,C) must encode
    the signed area. Dot product is symmetric -> cannot encode handedness.
    Clifford pseudoscalar part encodes det sign directly.
    """
    def __init__(self, n_samples=50000, seed=42):
        torch.manual_seed(seed)
        A = torch.randn(n_samples, 3)
        B = torch.randn(n_samples, 3)
        C = torch.randn(n_samples, 3)
        cross_z = ((B-A)[:, 0] * (C-A)[:, 1]
                 - (B-A)[:, 1] * (C-A)[:, 0])
        # x: (N, 3, 3) - 3 tokens of dim 3
        self.x = torch.stack([A, B, C], dim=1).float()
        self.y = (cross_z < 0).long() # (N,)

    def __len__(self): return len(self.x)
    def __getitem__(self, idx): return self.x[idx], self.y[idx]


def get_loaders(dataset, batch_size=256, val_split=0.1):
    n_val   = int(len(dataset) * val_split)
    n_train = len(dataset) - n_val
    train_ds, val_ds = torch.utils.data.random_split(
        dataset, [n_train, n_val])
    return (DataLoader(train_ds, batch_size=batch_size, shuffle=True),
            DataLoader(val_ds,   batch_size=batch_size, shuffle=False))