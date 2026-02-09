import torch
import torch.nn as nn
import torch.nn.utils.prune as tprune


class EarlyStopping:
    def __init__(self, patience=5, min_delta=0, save_path='best_model.pth'):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
        self.save_path = save_path
        self.model_best_state = None

    def __call__(self, val_loss, model):
        if self.best_loss is None:
            self.best_loss = val_loss
            self.model_best_state = model.state_dict()
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
                self.save_checkpoint(model)
        else:
            self.best_loss = val_loss
            self.counter = 0
            self.model_best_state = model.state_dict()

    def save_checkpoint(self, model):
        """Save the model when validation loss decreases."""
        print(f"Validation loss decreased. Saving model...")
        torch.save(self.model_best_state, self.save_path)
