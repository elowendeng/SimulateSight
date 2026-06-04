# utils/utils.py

import random
import gc
import numpy as np
import torch


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def cleanup():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def print_training_summary_simple(config, best_mae, best_epoch, visualizer):
    print(f"\n{'=' * 60}")
    print(f"TRAINING COMPLETED")
    print(f"{'=' * 60}")
    print(f"Best model saved at epoch {best_epoch}")
    print(f"Best MAE: {best_mae:.4f}")

    if visualizer.val_metrics['mae']:
        final_mae = visualizer.val_metrics['mae'][-1]
        print(f"\nFinal Metrics:")
        print(f"  MAE: {final_mae:.4f}")
        for name, values in visualizer.val_metrics.items():
            if name != 'mae' and values:
                print(f"  {name.upper()}: {values[-1]:.4f}")

    print(f"\nCheckpoints saved to: {config.save_dir}")
    print(f"{'=' * 60}")


def print_training_summary(config, best_mae, best_epoch, visualizer):
    print(f"\n{'='*60}")
    print(f"TRAINING COMPLETED")
    print(f"{'='*60}")
    print(f"Best model saved at epoch {best_epoch}")
    print(f"Best MAE: {best_mae:.4f}")
    
    if visualizer.val_metrics.get('mae'):
        final_mae = visualizer.val_metrics['mae'][-1]
        print(f"\nFinal Metrics:")
        print(f"  MAE: {final_mae:.4f}")
        for name, values in visualizer.val_metrics.items():
            if name != 'mae' and values:
                print(f"  {name.upper()}: {values[-1]:.4f}")

    if hasattr(visualizer, 'train_main_losses') and visualizer.train_main_losses and visualizer.train_main_losses[-1] is not None:
        print(f"\nFinal Loss Components:")
        print(f"  Main Loss: {visualizer.train_main_losses[-1]:.4f}")
        if hasattr(visualizer, 'train_aux_losses') and visualizer.train_aux_losses and visualizer.train_aux_losses[-1] is not None:
            print(f"  Aux Loss: {visualizer.train_aux_losses[-1]:.4f}")
    
    print(f"\nCheckpoints saved to: {config.save_dir}")
    print(f"{'='*60}")