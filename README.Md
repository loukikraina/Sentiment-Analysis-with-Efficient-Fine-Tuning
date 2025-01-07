# Sentiment Analysis with Fine-Tuning Techniques

This project focuses on comparing three fine-tuning techniques for pretrained language models to perform sentiment analysis on the IMDB dataset. The three models evaluated are:

1. **RoBERTa-large (Baseline Model)**
2. **LoRA (Low-Rank Adaptation)**
3. **Adapter-based Fine-Tuning**

The goal is to assess the trade-offs between accuracy, computational efficiency, and parameter efficiency.

## Table of Contents

- [Introduction](#introduction)
- [Project Setup](#project-setup)
- [Models Implemented](#models-implemented)
- [Dataset](#dataset)
- [Training](#training)
- [Evaluation](#evaluation)
- [Visualization](#visualization)
- [Results](#results)
- [Acknowledgements](#acknowledgements)

## Introduction

This project aims to explore and evaluate three techniques for fine-tuning pretrained transformer models, specifically RoBERTa-large, LoRA, and adapter layers. The IMDB dataset, which contains 50,000 labeled movie reviews, was used for sentiment analysis, with reviews labeled as either positive or negative.

## Project Setup

1. Clone the repository:
    ```bash
    git clone https://github.com/loukikraina/statnlp.git
    cd Project
    ```

2. Install the required dependencies using `pip`:
    ```bash
    pip install -r requirements.txt
    ```

    The `requirements.txt` file contains:
    - `torch`
    - `transformers`
    - `datasets`
    - `pandas`
    - `matplotlib`
    - `scikit-learn`
    - `peft`
  
## Models Implemented

1. **RoBERTa-large (Baseline Model)**:
   Fine-tuned end-to-end on the IMDB dataset. This model is used as a reference to compare the effectiveness of LoRA and adapter-based fine-tuning.

2. **LoRA (Low-Rank Adaptation)**:
   Fine-tunes only low-rank matrices in transformer layers. This technique offers an efficient way to adapt a pretrained model to a new task with fewer trainable parameters.

3. **Adapter-based Fine-Tuning**:
   Adapter layers are inserted into the model to allow fine-tuning with fewer trainable parameters. Only the adapter modules are updated during training, making this approach computationally efficient.

## Dataset

The IMDB dataset is used for training and evaluation. The dataset consists of 50,000 movie reviews, each labeled as either positive or negative. The dataset was split into an 80% training set and a 20% test set.

### Dataset Properties
- Total reviews: 50,000 (25,000 train, 25,000 test).
- Average review length: 230 tokens.
- Preprocessing: Tokenization, truncation to 128 tokens, and padding.

## Training

To train the models, you can run the following command (it will train all the models, if for any model saved model is already present then it will load the model instead of training it):

**Training RoBERTa-large (Baseline)** -> **Training LoRA** -> **Training Adapter-based Model**:
    ```bash
    python final_project.py
    ```


### Hyperparameters:
- Learning rate: `2e-5`
- Batch size: `16`
- Epochs: `5`
- Optimizer: `AdamW` with linear decay scheduler.

## Evaluation

The models are evaluated on the test set using the following metrics:
- Accuracy
- F1-score

After training, evaluation metrics are saved in CSV files (`training_log_model_name.csv`).

## Visualization

To visualize training progress for loss and gradient norm, use the following code:

```python
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
```

## Visualization

This code loads CSV logs for all three models and plots their loss and gradient norms for comparison.

## Results

After training, the following results were observed on the test set:

### RoBERTa-large (Baseline):
– Evaluation Loss: 0.3046
– Accuracy: 91.54%
– Precision: 91.60%
– Recall: 91.54%
– F1-Score: 91.54%

### LoRA:
– Evaluation Loss: 0.1853
– Accuracy: 92.91%
– Precision: 92.91%
– Recall: 92.91%
– F1-Score: 92.91%

### Adapter-based Fine-Tuning:
– Evaluation Loss: 0.3764
– Accuracy: 84.54%
– Precision: 84.58%
– Recall: 84.54%
– F1-Score: 84.53%

### Observations:
- The baseline model performed slightly better than LoRA and adapter-based models, but LoRA and adapters required significantly fewer trainable parameters and computational resources.

## Acknowledgements

This report used AI tools like ChatGPT for generating LaTeX structure and formatting suggestions. All content was written and edited manually, with modifications made to align the project with specific requirements.


## Requirements

- Python 3.7+
- `torch==1.10.0`
- `transformers==4.12.0`
- `datasets==1.18.3`
- `pandas==1.3.3`
- `matplotlib==3.4.3`
- `scikit-learn==0.24.2`
