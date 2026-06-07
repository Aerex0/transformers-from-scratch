import matplotlib.pyplot as plt
import pandas as pd


df = pd.read_csv("transformer_training_history.csv")


plt.figure(figsize=(10, 6))
plt.plot(df['epoch'], df['train_loss'], label='Training Loss', color='blue', linewidth=2)
plt.plot(df['epoch'], df['val_loss'], label='Validation Loss', color='orange', linestyle='--', linewidth=2)


plt.title('Transformer Training Convergence Curves', fontsize=14, fontweight='bold')
plt.xlabel('Epochs', fontsize=12)
plt.ylabel('Cross-Entropy Loss', fontsize=12)
plt.grid(True, linestyle=':', alpha=0.6)
plt.legend(fontsize=12)


plt.savefig("loss_curves.png", dpi=300)
# plt.show()