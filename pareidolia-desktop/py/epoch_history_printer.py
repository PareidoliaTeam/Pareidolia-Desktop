import pytorch_lightning as pl
from pytorch_lightning.callbacks import Callback, ModelCheckpoint, EarlyStopping, LearningRateMonitor, RichProgressBar
from pytorch_lightning.loggers import CSVLogger
import pandas as pd
import math

class EpochHistoryPrinter(Callback):
    """
    Author: Armando Vega
    Date Created: 7 April 2026

    Last Modified By: Armando Vega
    Date Last Modified: 7 April 2026

    A PyTorch Lightning callback that prints the training and validation loss and accuracy at the end of each validation epoch. 
    It retrieves the metrics from the trainer's callback_metrics dictionary, handling both tensor and non-tensor values, and ensures 
    that only finite values are printed. The output is formatted for easy readability, showing the current epoch number along with the relevant metrics.
    """
    
    def __init__(self, print_every_n_steps=100):
        super().__init__()
        self.print_every_n_steps = max(1, int(print_every_n_steps))

    def on_validation_epoch_end(self, trainer, pl_module):
        if trainer.sanity_checking:
            return  # skip pre-training sanity pass

        m = trainer.callback_metrics

        def to_float(*names):
            value = None
            for name in names:
                value = m.get(name)
                if value is not None:
                    break

            if value is None:
                return float("nan")

            if hasattr(value, "detach"):
                value = value.detach().cpu().item()
            else:
                value = float(value)

            return value if math.isfinite(value) else float("nan")

        print(
            f"epoch={trainer.current_epoch:02d} "
            f"train_loss={to_float('train_loss', 'train_loss_epoch'):.4f} "
            f"train_acc={to_float('train_acc', 'train_acc_epoch'):.4f} "
            f"val_loss={to_float('val_loss', 'val_loss_epoch'):.4f} "
            f"val_acc={to_float('val_acc', 'val_acc_epoch'):.4f}",
            flush=True,
        )