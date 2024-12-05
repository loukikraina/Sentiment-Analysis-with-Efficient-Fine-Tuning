import pandas as pd
import matplotlib.pyplot as plt

# Load CSV logs for all three models
log_model_1 = pd.read_csv('./logs/training_log_base.csv')
log_model_2 = pd.read_csv('./logs/training_log_lora.csv')
log_model_3 = pd.read_csv('./logs/training_log_adapter.csv')

# Plot training loss and gradient norm for all models
plt.figure(figsize=(14, 12))

# Training Loss Plot
plt.subplot(2, 1, 1)
plt.plot(log_model_1['step'], log_model_1['loss'], label="Base model Loss")
plt.plot(log_model_2['step'], log_model_2['loss'], label="LoRA model Loss")
plt.plot(log_model_3['step'], log_model_3['loss'], label="Adapter model Loss")
plt.xlabel('Step')
plt.ylabel('Loss')
plt.title('Training Loss Progression')
plt.grid()
plt.legend()

# Gradient Norm Plot
plt.subplot(2, 1, 2)
plt.plot(log_model_1['step'], log_model_1['gradient_norm'], label="Base model Gradient Norm", linestyle='dashed')
plt.plot(log_model_2['step'], log_model_2['gradient_norm'], label="LoRA model Gradient Norm", linestyle='dotted')
plt.plot(log_model_3['step'], log_model_3['gradient_norm'], label="Adapter model Gradient Norm", linestyle='dashdot')
plt.xlabel('Step')
plt.ylabel('Gradient Norm')
plt.title('Gradient Norm Progression')
plt.grid()
plt.legend()

plt.tight_layout()
plt.show()
