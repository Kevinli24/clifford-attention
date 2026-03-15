import torch
import torch.nn as nn
from dataset import RotationAxisDataset, OrientationDataset, get_loaders
from models import StandardTransformerModel, CliffordTransformerModel, count_params

DEVICE     = 'cuda' if torch.cuda.is_available() else 'cpu'
EPOCHS     = 40
BATCH_SIZE = 256
LR         = 1e-3
D_MODEL    = 64
NUM_HEADS  = 1
NUM_LAYERS = 2

print(f"Device: {DEVICE}")


def train_epoch(model, loader, optimizer, loss_fn):
    model.train()
    total = 0.0
    for x, y in loader:
        x, y = x.to(DEVICE), y.to(DEVICE)
        optimizer.zero_grad()
        loss = loss_fn(model(x), y)
        loss.backward()
        optimizer.step()
        total += loss.item() * len(x)
    return total / len(loader.dataset)


@torch.no_grad()
def eval_epoch(model, loader, loss_fn, task):
    model.eval()
    total, correct, n = 0.0, 0, 0
    for x, y in loader:
        x, y = x.to(DEVICE), y.to(DEVICE)
        pred = model(x)
        total += loss_fn(pred, y).item() * len(x)
        if task == 'orientation':
            correct += (pred.argmax(-1) == y).sum().item()
            n += len(y)
    return total / len(loader.dataset), (correct / n if task == 'orientation' else None)


def run_experiment(task):
    print("=" * 60)
    print(f"TASK: {task.upper()}")
    print("=" * 60)

    if task == 'rotation':
        ds        = RotationAxisDataset()
        token_dim = 3
        n_tokens  = 2
        out_dim   = 3
        loss_fn   = nn.MSELoss()
    else:
        ds        = OrientationDataset()
        token_dim = 3
        n_tokens  = 3
        out_dim   = 2
        loss_fn   = nn.CrossEntropyLoss()

    train_loader, val_loader = get_loaders(ds, BATCH_SIZE)

    std = StandardTransformerModel(
        token_dim, n_tokens, D_MODEL, NUM_HEADS, out_dim, NUM_LAYERS).to(DEVICE)
    clf = CliffordTransformerModel(
        token_dim, n_tokens, D_MODEL, NUM_HEADS, out_dim, NUM_LAYERS).to(DEVICE)

    print(f"Standard params: {count_params(std):,}")
    print(f"Clifford params: {count_params(clf):,}")

    std_opt = torch.optim.Adam(std.parameters(), lr=LR)
    clf_opt = torch.optim.Adam(clf.parameters(), lr=LR)

    history = {k: [] for k in
               ['std_train','std_val','clf_train','clf_val','std_acc','clf_acc']}

    for ep in range(1, EPOCHS + 1):
        std_tr = train_epoch(std, train_loader, std_opt, loss_fn)
        clf_tr = train_epoch(clf, train_loader, clf_opt, loss_fn)
        std_vl, std_ac = eval_epoch(std, val_loader, loss_fn, task)
        clf_vl, clf_ac = eval_epoch(clf, val_loader, loss_fn, task)

        history['std_train'].append(std_tr)
        history['std_val'].append(std_vl)
        history['clf_train'].append(clf_tr)
        history['clf_val'].append(clf_vl)

        if task == 'orientation':
            history['std_acc'].append(std_ac)
            history['clf_acc'].append(clf_ac)
            print(f"Ep {ep:02d} | Std loss {std_vl:.4f} acc {std_ac:.3f} | Clf loss {clf_vl:.4f} acc {clf_ac:.3f}")
        else:
            print(f"Ep {ep:02d} | Std MSE {std_vl:.5f} | Clf MSE {clf_vl:.5f}")

    torch.save(history, f'history_{task}.pt')
    print(f"Saved history_{task}.pt")


run_experiment('rotation')
run_experiment('orientation')
