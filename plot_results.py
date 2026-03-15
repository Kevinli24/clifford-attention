import torch
import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 3, figsize=(15, 4))

# Plot 1: Rotation MSE
h = torch.load('history_rotation.pt')
ax = axes[0]
ax.plot(h['std_val'], label='Standard (dot product)', color='royalblue')
ax.plot(h['clf_val'], label='Clifford (geometric product)', color='crimson')
ax.set_title('Rotation Axis Prediction\n(lower MSE = better)')
ax.set_xlabel('Epoch')
ax.set_ylabel('Validation MSE')
ax.legend()
ax.grid(True, alpha=0.3)

# Plot 2: Orientation Loss
h2 = torch.load('history_orientation.pt')
ax = axes[1]
ax.plot(h2['std_val'], label='Standard (dot product)', color='royalblue')
ax.plot(h2['clf_val'], label='Clifford (geometric product)', color='crimson')
ax.set_title('Orientation Classification\n(lower loss = better)')
ax.set_xlabel('Epoch')
ax.set_ylabel('Validation Cross-Entropy')
ax.legend()
ax.grid(True, alpha=0.3)

# Plot 3: Orientation Accuracy
ax = axes[2]
ax.plot(h2['std_acc'], label='Standard (dot product)', color='royalblue')
ax.plot(h2['clf_acc'], label='Clifford (geometric product)', color='crimson')
ax.axhline(0.5, color='gray', linestyle='--', label='Random baseline')
ax.set_title('Orientation Classification\n(higher accuracy = better)')
ax.set_xlabel('Epoch')
ax.set_ylabel('Validation Accuracy')
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('results.png', dpi=150)
print("Saved results.png")
plt.show()