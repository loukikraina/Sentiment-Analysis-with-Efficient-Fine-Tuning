import pandas as pd
from transformers import TrainerCallback
import os

class CSVLoggerCallback(TrainerCallback):
    def __init__(self, log_file="./logs/training_log.csv"):
        self.log_file = log_file
        if not os.path.exists(log_file):
            # Initialize the CSV with headers if it doesn't exist
            with open(log_file, "w") as f:
                f.write("step,loss,lr,gradient_norm\n")
    
    def on_log(self, args, state, control, logs=None, model=None, **kwargs):
        if logs is not None and model is not None:
            step = state.global_step
            loss = logs.get("loss", None)
            lr = logs.get("learning_rate", None)

            # Calculate the gradient norm
            gradient_norm = logs.get("grad_norm", None)

            # Append to CSV
            if loss is not None:
                with open(self.log_file, "a") as f:
                    f.write(f"{step},{loss},{lr},{gradient_norm}\n")