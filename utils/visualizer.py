# utils/visualizer.py

import os
import numpy as np
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt


class TrainingVisualizer:
    def __init__(self, save_dir, metrics_names):
        self.save_dir = save_dir
        self.metrics_names = metrics_names
        self.train_losses = []
        self.val_metrics = {name: [] for name in metrics_names}
        self.epochs = []
        self.fig, (self.ax1, self.ax2) = plt.subplots(1, 2, figsize=(15, 5))

    def update(self, epoch, train_loss, val_metrics):
        self.epochs.append(epoch)
        self.train_losses.append(train_loss)

        for name in self.metrics_names:
            if name in val_metrics:
                self.val_metrics[name].append(val_metrics[name])
            else:
                self.val_metrics[name].append(None)

        if epoch % 5 == 0 or epoch == 1:
            self.plot()

    def plot(self):
        self.ax1.clear()
        self.ax2.clear()

        if len(self.epochs) > 0 and len(self.train_losses) > 0:
            min_len = min(len(self.epochs), len(self.train_losses))
            if min_len > 0:
                self.ax1.plot(self.epochs[:min_len], self.train_losses[:min_len],
                              'b-', label='Train Loss', linewidth=2)

        self.ax1.set_xlabel('Epoch')
        self.ax1.set_ylabel('Loss')
        self.ax1.set_title('Training Loss')
        self.ax1.grid(True, alpha=0.3)
        self.ax1.legend()

        colors = ['r', 'g', 'm', 'c', 'y', 'k']

        for i, (name, values) in enumerate(self.val_metrics.items()):
            if values:
                valid_indices = [j for j, v in enumerate(values) if v is not None]
                if valid_indices:
                    valid_epochs = [self.epochs[j] for j in valid_indices]
                    valid_values = [values[j] for j in valid_indices]

                    if len(valid_epochs) > 0 and len(valid_values) > 0:
                        self.ax2.plot(valid_epochs, valid_values,
                                      color=colors[i % len(colors)],
                                      label=name.upper(), linewidth=2,
                                      marker='o', markersize=3)

        self.ax2.set_xlabel('Epoch')
        self.ax2.set_ylabel('Metric Value')
        self.ax2.set_title('Validation Metrics')
        self.ax2.grid(True, alpha=0.3)
        self.ax2.legend()

        plt.tight_layout()
        save_path = os.path.join(self.save_dir, 'training_curves.png')
        self.fig.savefig(save_path, dpi=150, bbox_inches='tight')

    def save_final(self):
        self.plot()
        print(f"\nTraining curves saved to {os.path.join(self.save_dir, 'training_curves.png')}")

        clean_val_metrics = {}
        for name, values in self.val_metrics.items():
            clean_val_metrics[name] = [v for v in values if v is not None]

        data = {
            'epochs': self.epochs,
            'train_losses': self.train_losses,
            'val_metrics': clean_val_metrics
        }
        np.save(os.path.join(self.save_dir, 'training_data.npy'), data)


class TrainingVisualizer_aux:
    def __init__(self, save_dir, metrics_names):
        self.save_dir = save_dir
        self.metrics_names = metrics_names
        self.train_losses = []
        self.train_main_losses = []
        self.train_aux_losses = []
        self.val_metrics = {name: [] for name in metrics_names}
        self.epochs = []
        self.fig, (self.ax1, self.ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
    def update(self, epoch, train_loss, val_metrics):
        self.epochs.append(epoch)
        
        if isinstance(train_loss, dict):
            self.train_losses.append(train_loss['total_loss'])
            self.train_main_losses.append(train_loss['main_loss'])
            self.train_aux_losses.append(train_loss['aux_loss'])
        else:
            self.train_losses.append(train_loss)
            if len(self.train_main_losses) < len(self.epochs):
                self.train_main_losses.append(None)
            if len(self.train_aux_losses) < len(self.epochs):
                self.train_aux_losses.append(None)
        
        for name in self.metrics_names:
            if name in val_metrics:
                self.val_metrics[name].append(val_metrics[name])
            else:
                self.val_metrics[name].append(None)
        
        if epoch % 5 == 0 or epoch == 1:
            self.plot()
    
    def plot(self):
        self.ax1.clear()
        self.ax2.clear()
        
        if len(self.epochs) > 0 and len(self.train_losses) > 0:
            min_len = min(len(self.epochs), len(self.train_losses))
            if min_len > 0:
                self.ax1.plot(self.epochs[:min_len], self.train_losses[:min_len], 
                            'b-', label='Train Loss', linewidth=2)
                
                if any(l is not None for l in self.train_main_losses):
                    valid_main = [(e, l) for e, l in zip(self.epochs, self.train_main_losses) if l is not None]
                    if valid_main:
                        epochs_main, losses_main = zip(*valid_main)
                        self.ax1.plot(epochs_main, losses_main, 'r-', label='Main Loss', linewidth=2, alpha=0.7)
                
                if any(l is not None for l in self.train_aux_losses):
                    valid_aux = [(e, l) for e, l in zip(self.epochs, self.train_aux_losses) if l is not None]
                    if valid_aux:
                        epochs_aux, losses_aux = zip(*valid_aux)
                        self.ax1.plot(epochs_aux, losses_aux, 'g-', label='Aux Loss', linewidth=2, alpha=0.7)
        
        self.ax1.set_xlabel('Epoch')
        self.ax1.set_ylabel('Loss')
        self.ax1.set_title('Training Loss')
        self.ax1.grid(True, alpha=0.3)
        self.ax1.legend()
        
        colors = ['r', 'g', 'm', 'c', 'y', 'k']
        
        for i, (name, values) in enumerate(self.val_metrics.items()):
            if values:
                valid_indices = [j for j, v in enumerate(values) if v is not None]
                if valid_indices:
                    valid_epochs = [self.epochs[j] for j in valid_indices]
                    valid_values = [values[j] for j in valid_indices]
                    if len(valid_epochs) > 0 and len(valid_values) > 0:
                        self.ax2.plot(valid_epochs, valid_values, 
                                    color=colors[i % len(colors)], 
                                    label=name.upper(), linewidth=2, 
                                    marker='o', markersize=3)
        
        self.ax2.set_xlabel('Epoch')
        self.ax2.set_ylabel('Metric Value')
        self.ax2.set_title('Validation Metrics')
        self.ax2.grid(True, alpha=0.3)
        self.ax2.legend()
        
        plt.tight_layout()
        save_path = os.path.join(self.save_dir, 'training_curves.png')
        self.fig.savefig(save_path, dpi=150, bbox_inches='tight')
    
    def save_final(self):
        self.plot()
        print(f"\nTraining curves saved to {os.path.join(self.save_dir, 'training_curves.png')}")
        
        clean_val_metrics = {}
        for name, values in self.val_metrics.items():
            clean_val_metrics[name] = [v for v in values if v is not None]
        
        data = {
            'epochs': self.epochs,
            'train_losses': self.train_losses,
            'train_main_losses': [l for l in self.train_main_losses if l is not None],
            'train_aux_losses': [l for l in self.train_aux_losses if l is not None],
            'val_metrics': clean_val_metrics
        }
        np.save(os.path.join(self.save_dir, 'training_data.npy'), data)