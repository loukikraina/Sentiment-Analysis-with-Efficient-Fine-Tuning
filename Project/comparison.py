import evaluate
from sklearn.metrics import confusion_matrix, classification_report
import torch
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from transformers import Trainer

def evaluate_model(trainer, model, name, test_dataset):
    """
    Evaluate the given model and compute various metrics.
    """
    
    print(f"\nEvaluating {name} Model...")
    
    # Standard evaluation using Trainer
    results = trainer.evaluate()
    
    # Extract predictions and labels
    predictions, labels, _ = trainer.predict(test_dataset)
    preds = np.argmax(predictions, axis=1)
    
    # Load individual metrics
    accuracy_metric = evaluate.load("accuracy")
    precision_metric = evaluate.load("precision")
    recall_metric = evaluate.load("recall")
    f1_metric = evaluate.load("f1")
    
    # Compute metrics
    accuracy = accuracy_metric.compute(predictions=preds, references=labels)["accuracy"]
    precision = precision_metric.compute(predictions=preds, references=labels, average="weighted")["precision"]
    recall = recall_metric.compute(predictions=preds, references=labels, average="weighted")["recall"]
    f1 = f1_metric.compute(predictions=preds, references=labels, average="weighted")["f1"]
    
    # Compute confusion matrix
    confusion_mat = confusion_matrix(labels, preds)
    
    # Print metrics
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-Score: {f1:.4f}")
    print("\nConfusion Matrix:\n", confusion_mat)
    
    # Visualize confusion matrix
    visualize_confusion_matrix(confusion_mat, name)
    
   # Add results to a dictionary
    metrics = {
        "eval_loss": results["eval_loss"],
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "confusion_matrix": confusion_mat
    }
    
    return metrics

def visualize_confusion_matrix(confusion_mat, model_name):
    """
    Visualize the confusion matrix.
    """
    plt.figure(figsize=(8, 6))
    sns.heatmap(confusion_mat, annot=True, fmt="d", cmap="Blues")
    plt.title(f"Confusion Matrix - {model_name} Model")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.show()

def calculate_model_size(model):
    """
    Calculate total and trainable parameters of a model.
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_params, trainable_params

# Main comparison code
def compare_models(base_model, lora_model, adapter_model, training_args_list, test_dataset, tokenizer):
    """
    Compare Base, LoRA, and Adapter models on various metrics.
    """
    metrics_summary = {}

    # Evaluate Base Model
    base_trainer = Trainer(
        model=base_model,
        args=training_args_list[0],
        eval_dataset=test_dataset,
        tokenizer=tokenizer,
    )
    base_results = evaluate_model(base_trainer, base_model, "Base", test_dataset)
    metrics_summary["Base"] = base_results

    # Evaluate LoRA Model
    lora_trainer = Trainer(
        model=lora_model,
        args=training_args_list[1],
        eval_dataset=test_dataset,
        tokenizer=tokenizer,
    )
    lora_results = evaluate_model(lora_trainer, lora_model, "LoRA", test_dataset)
    metrics_summary["LoRA"] = lora_results

    # Evaluate Adapter Model
    adapter_trainer = Trainer(
        model=adapter_model,
        args=training_args_list[2],
        eval_dataset=test_dataset,
        tokenizer=tokenizer,
    )
    adapter_results = evaluate_model(adapter_trainer, adapter_model, "Adapter", test_dataset)
    metrics_summary["Adapter"] = adapter_results

    # Parameter size comparison
    print("\nParameter Size Comparison:")
    for name, model in zip(["Base", "LoRA", "Adapter"], [base_model, lora_model, adapter_model]):
        total_params, trainable_params = calculate_model_size(model)
        print(f"{name} Model - Total Params: {total_params:,}, Trainable Params: {trainable_params:,}")

    # Summarize all results
    print("\nSummary of Results:")
    for model_name, metrics in metrics_summary.items():
        print(f"{model_name} Model:")
        for metric_name, value in metrics.items():
            if metric_name != "confusion_matrix":
                print(f"  {metric_name}: {value}")
        print()  # Blank line for readability
    
    return metrics_summary

