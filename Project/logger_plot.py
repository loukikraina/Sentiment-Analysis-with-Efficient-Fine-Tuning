import pandas as pd
import matplotlib.pyplot as plt

# Load the CSV log
log_df = pd.read_csv('./logs/training_log.csv')

# Plot training loss and gradient norm
plt.figure(figsize=(14, 6))

plt.subplot(1, 2, 1)
plt.plot(log_df['step'], log_df['loss'], label="Training Loss")
plt.xlabel('Step')
plt.ylabel('Loss')
plt.title('Training Loss Progression')
plt.grid()
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(log_df['step'], log_df['gradient_norm'], label="Gradient Norm", color='orange')
plt.xlabel('Step')
plt.ylabel('Gradient Norm')
plt.title('Gradient Norm Progression')
plt.grid()
plt.legend()

plt.tight_layout()
plt.show()
