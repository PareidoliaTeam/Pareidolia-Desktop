import pytorch_lightning as pl
from pytorch_lightning.callbacks import Callback, ModelCheckpoint, EarlyStopping, LearningRateMonitor, RichProgressBar
from pytorch_lightning.loggers import CSVLogger
import pandas as pd
import math

class EpochHistoryPrinter(Callback):
    def __init__(self, print_every_n_steps=100):
        super().__init__()
        self.print_every_n_steps = max(1, int(print_every_n_steps))

    # def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
    #     # Heartbeat so notebook runs don't look frozen when progress widgets fail.
    #     if (batch_idx + 1) % self.print_every_n_steps == 0:
    #         total = trainer.num_training_batches
    #         total_txt = str(total) if total is not None else "?"
    #         print(f"epoch={trainer.current_epoch:02d} batch={batch_idx + 1}/{total_txt}", flush=True)

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