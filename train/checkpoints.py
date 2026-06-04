# train/checkpoints.py

import os
os.environ['NO_ALBUMENTATIONS_UPDATE'] = '1'

import torch


def save_latest_checkpoint_single(config, epoch, model, optimizer, scheduler, metrics, train_loss,
                        best_mae, best_epoch, patience_counter, rebound_counter,
                        prev_mae, visualizer, scaler=None):
    if not config.save_latest:
        return

    latest_path = os.path.join(config.save_dir, 'checkpoint_latest.pth')

    scheduler_state = None
    if scheduler is not None:
        try:
            if hasattr(scheduler, 'state_dict'):
                scheduler_state = scheduler.state_dict()
            else:
                if hasattr(scheduler, 'get_last_lr'):
                    scheduler_state = {'last_lr': scheduler.get_last_lr()}
        except (AttributeError, NotImplementedError, TypeError):
            print(f"Warning: Could not save scheduler state for {type(scheduler).__name__}")

    train_losses = []
    for loss in visualizer.train_losses:
        if loss is not None:
            train_losses.append(float(loss))

    val_metrics = {}
    for k, v in visualizer.val_metrics.items():
        val_metrics[k] = [float(x) for x in v if x is not None]

    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler_state,
        'scaler_state_dict': scaler.state_dict() if scaler else None,
        'metrics': {k: float(v) for k, v in metrics.items()},
        'train_loss': float(train_loss),
        'best_mae': float(best_mae),
        'best_epoch': int(best_epoch),
        'patience_counter': int(patience_counter),
        'rebound_counter': int(rebound_counter),
        'prev_mae': float(prev_mae),
        'train_losses': train_losses,
        'val_metrics': val_metrics,
        'epochs': [int(x) for x in visualizer.epochs if x is not None],
        'scheduler_type': type(scheduler).__name__ if scheduler else None,
        'attn_type': config.attn_type,
        'use_improved_attn': config.use_improved_attn,
    }

    temp_path = latest_path + '.tmp'
    try:
        torch.save(checkpoint, temp_path)
        os.replace(temp_path, latest_path)
        print(f"✓ Latest checkpoint saved to {latest_path}")
    except Exception as e:
        print(f"Warning: Could not save latest checkpoint: {e}")


def load_checkpoint_single(config, model, optimizer, scheduler, visualizer, scaler=None):
    """Loading checkpoint"""
    if not config.resume_training or not os.path.exists(config.resume_checkpoint):
        return 1, float('inf'), 0, 0, 0, float('inf')

    print(f"\n{'=' * 40}")
    print(f"Resuming training from checkpoint: {config.resume_checkpoint}")
    print(f"{'=' * 40}")

    try:
        checkpoint = torch.load(config.resume_checkpoint, map_location=config.device)
    except Exception as e:
        print(f"Error loading checkpoint: {e}")
        return 1, float('inf'), 0, 0, 0, float('inf')

    if 'model_state_dict' in checkpoint:
        try:
            model.load_state_dict(checkpoint['model_state_dict'])
            print("✓ Model weights loaded")
        except Exception as e:
            print(f"Error loading model weights: {e}")
            return 1, float('inf'), 0, 0, 0, float('inf')
    else:
        print("Warning: No model state dict found in checkpoint")
        return 1, float('inf'), 0, 0, 0, float('inf')

    if 'optimizer_state_dict' in checkpoint and optimizer is not None:
        try:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            print("✓ Optimizer state loaded")
        except Exception as e:
            print(f"Warning: Could not load optimizer state: {e}")

    if scheduler and 'scheduler_state_dict' in checkpoint and checkpoint['scheduler_state_dict']:
        try:
            saved_type = checkpoint.get('scheduler_type')
            current_type = type(scheduler).__name__

            if saved_type == current_type:
                if hasattr(scheduler, 'load_state_dict'):
                    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
                    print("✓ Scheduler state loaded")
            else:
                print(f"Warning: Scheduler type mismatch (saved: {saved_type}, current: {current_type})")
        except Exception as e:
            print(f"Warning: Could not load scheduler state: {e}")

    if scaler and 'scaler_state_dict' in checkpoint and checkpoint['scaler_state_dict']:
        try:
            scaler.load_state_dict(checkpoint['scaler_state_dict'])
            print("✓ Scaler state loaded")
        except Exception as e:
            print(f"Warning: Could not load scaler state: {e}")

    start_epoch = checkpoint.get('epoch', 0) + 1
    best_mae = checkpoint.get('best_mae', float('inf'))
    best_epoch = checkpoint.get('best_epoch', 0)
    patience_counter = checkpoint.get('patience_counter', 0)
    rebound_counter = checkpoint.get('rebound_counter', 0)
    prev_mae = checkpoint.get('prev_mae', float('inf'))

    if visualizer:
        if 'train_losses' in checkpoint:
            visualizer.train_losses = checkpoint['train_losses']
        if 'val_metrics' in checkpoint:
            for k, v in checkpoint['val_metrics'].items():
                if k in visualizer.val_metrics:
                    visualizer.val_metrics[k] = v
        if 'epochs' in checkpoint:
            visualizer.epochs = checkpoint['epochs']

    print(f"Resumed from epoch {checkpoint.get('epoch', 0)}")
    print(f"Best MAE so far: {best_mae:.4f} at epoch {best_epoch}")
    print(f"Patience counter: {patience_counter}/{config.patience}")

    return start_epoch, best_mae, best_epoch, patience_counter, rebound_counter, prev_mae



def save_latest_checkpoint_dual(config, epoch, model, optimizer, scheduler, metrics, train_loss,
                        best_mae, best_epoch, patience_counter, rebound_counter,
                        prev_mae, visualizer, analyzer=None, scaler=None):
    if not config.save_latest:
        return

    latest_path = os.path.join(config.save_dir, 'checkpoint_latest.pth')

    scheduler_state = None
    if scheduler is not None:
        try:
            if hasattr(scheduler, 'state_dict'):
                scheduler_state = scheduler.state_dict()
            else:
                if hasattr(scheduler, 'get_last_lr'):
                    scheduler_state = {'last_lr': scheduler.get_last_lr()}
        except (AttributeError, NotImplementedError, TypeError):
            print(f"Warning: Could not save scheduler state for {type(scheduler).__name__}")

    train_losses = []
    for loss in visualizer.train_losses:
        if loss is not None:
            train_losses.append(float(loss))

    val_metrics = {}
    for k, v in visualizer.val_metrics.items():
        val_metrics[k] = [float(x) for x in v if x is not None]

    if isinstance(train_loss, dict):
        train_loss_val = float(train_loss['total_loss'])
        train_main_loss_val = float(train_loss['main_loss'])
        train_aux_loss_val = float(train_loss['aux_loss'])
    else:
        train_loss_val = float(train_loss) if train_loss is not None else 0.0
        train_main_loss_val = None
        train_aux_loss_val = None

    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler_state,
        'scaler_state_dict': scaler.state_dict() if scaler else None,
        'metrics': {k: float(v) for k, v in metrics.items()},
        'train_loss': train_loss_val,
        'best_mae': float(best_mae),
        'best_epoch': int(best_epoch),
        'patience_counter': int(patience_counter),
        'rebound_counter': int(rebound_counter),
        'prev_mae': float(prev_mae),
        'train_losses': train_losses,
        'val_metrics': val_metrics,
        'epochs': [int(x) for x in visualizer.epochs if x is not None],
        'scheduler_type': type(scheduler).__name__ if scheduler else None,
        'use_aux_loss': config.use_aux_loss,
        'attn_type': config.attn_type,
        'use_improved_fusion': config.use_improved_fusion,
    }

    if hasattr(visualizer, 'train_main_losses') and visualizer.train_main_losses:
        checkpoint['train_main_losses'] = [float(x) for x in visualizer.train_main_losses if x is not None]
    if hasattr(visualizer, 'train_aux_losses') and visualizer.train_aux_losses:
        checkpoint['train_aux_losses'] = [float(x) for x in visualizer.train_aux_losses if x is not None]

    if analyzer is not None:
        checkpoint['complementarity_history'] = analyzer.history

    temp_path = latest_path + '.tmp'
    try:
        torch.save(checkpoint, temp_path)
        os.replace(temp_path, latest_path)
        print(f"✓ Latest checkpoint saved to {latest_path}")
    except Exception as e:
        print(f"Warning: Could not save latest checkpoint: {e}")


def load_checkpoint_dual(config, model, optimizer, scheduler, visualizer, analyzer=None, scaler=None):
    if not config.resume_training or not os.path.exists(config.resume_checkpoint):
        return 1, float('inf'), 0, 0, 0, float('inf')

    print(f"\n{'=' * 40}")
    print(f"Resuming training from checkpoint: {config.resume_checkpoint}")
    print(f"{'=' * 40}")

    try:
        checkpoint = torch.load(config.resume_checkpoint, map_location=config.device)
    except Exception as e:
        print(f"Error loading checkpoint: {e}")
        return 1, float('inf'), 0, 0, 0, float('inf')

    if 'model_state_dict' in checkpoint:
        try:
            model.load_state_dict(checkpoint['model_state_dict'])
            print("✓ Model weights loaded")
        except Exception as e:
            print(f"Error loading model weights: {e}")
            return 1, float('inf'), 0, 0, 0, float('inf')
    else:
        print("Warning: No model state dict found in checkpoint")
        return 1, float('inf'), 0, 0, 0, float('inf')

    if 'optimizer_state_dict' in checkpoint and optimizer is not None:
        try:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            print("✓ Optimizer state loaded")
        except Exception as e:
            print(f"Warning: Could not load optimizer state: {e}")

    if scheduler and 'scheduler_state_dict' in checkpoint and checkpoint['scheduler_state_dict']:
        try:
            saved_type = checkpoint.get('scheduler_type')
            current_type = type(scheduler).__name__

            if saved_type == current_type:
                if hasattr(scheduler, 'load_state_dict'):
                    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
                    print("✓ Scheduler state loaded")
            else:
                print(f"Warning: Scheduler type mismatch (saved: {saved_type}, current: {current_type})")
        except Exception as e:
            print(f"Warning: Could not load scheduler state: {e}")

    if scaler and 'scaler_state_dict' in checkpoint and checkpoint['scaler_state_dict']:
        try:
            scaler.load_state_dict(checkpoint['scaler_state_dict'])
            print("✓ Scaler state loaded")
        except Exception as e:
            print(f"Warning: Could not load scaler state: {e}")

    start_epoch = checkpoint.get('epoch', 0) + 1
    best_mae = checkpoint.get('best_mae', float('inf'))
    best_epoch = checkpoint.get('best_epoch', 0)
    patience_counter = checkpoint.get('patience_counter', 0)
    rebound_counter = checkpoint.get('rebound_counter', 0)
    prev_mae = checkpoint.get('prev_mae', float('inf'))

    if visualizer:
        if 'train_losses' in checkpoint:
            visualizer.train_losses = checkpoint['train_losses']
        
        if hasattr(visualizer, 'train_main_losses') and 'train_main_losses' in checkpoint:
            visualizer.train_main_losses = checkpoint['train_main_losses']
        if hasattr(visualizer, 'train_aux_losses') and 'train_aux_losses' in checkpoint:
            visualizer.train_aux_losses = checkpoint['train_aux_losses']
        
        if 'val_metrics' in checkpoint:
            for k, v in checkpoint['val_metrics'].items():
                if k in visualizer.val_metrics:
                    visualizer.val_metrics[k] = v
        if 'epochs' in checkpoint:
            visualizer.epochs = checkpoint['epochs']

    if analyzer is not None and 'complementarity_history' in checkpoint:
        analyzer.history = checkpoint['complementarity_history']

    print(f"Resumed from epoch {checkpoint.get('epoch', 0)}")
    print(f"Best MAE so far: {best_mae:.4f} at epoch {best_epoch}")
    print(f"Patience counter: {patience_counter}/{config.patience}")

    return start_epoch, best_mae, best_epoch, patience_counter, rebound_counter, prev_mae



def save_latest_checkpoint_triple(config, epoch, model, optimizer, scheduler, metrics, train_loss, 
                        best_mae, best_epoch, patience_counter, rebound_counter, 
                        prev_mae, visualizer, scaler=None, analyzer=None):
    if not config.save_latest:
        return
    
    latest_path = os.path.join(config.save_dir, 'checkpoint_latest.pth')
    
    scheduler_state = None
    if scheduler is not None:
        try:
            if hasattr(scheduler, 'state_dict'):
                scheduler_state = scheduler.state_dict()
            else:
                if hasattr(scheduler, 'get_last_lr'):
                    scheduler_state = {'last_lr': scheduler.get_last_lr()}
        except (AttributeError, NotImplementedError, TypeError):
            print(f"Warning: Could not save scheduler state for {type(scheduler).__name__}")
    
    train_losses = []
    for loss in visualizer.train_losses:
        if loss is not None:
            train_losses.append(float(loss))
    
    val_metrics = {}
    for k, v in visualizer.val_metrics.items():
        val_metrics[k] = [float(x) for x in v if x is not None]
    
    if isinstance(train_loss, dict):
        train_loss_val = float(train_loss['total_loss'])
        train_main_loss_val = float(train_loss['main_loss'])
        train_aux_loss_val = float(train_loss['aux_loss'])
    else:
        train_loss_val = float(train_loss)
        train_main_loss_val = None
        train_aux_loss_val = None
    
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler_state,
        'scaler_state_dict': scaler.state_dict() if scaler else None,
        'metrics': {k: float(v) for k, v in metrics.items()},
        'train_loss': train_loss_val,
        'train_main_loss': train_main_loss_val,
        'train_aux_loss': train_aux_loss_val,
        'best_mae': float(best_mae),
        'best_epoch': int(best_epoch),
        'patience_counter': int(patience_counter),
        'rebound_counter': int(rebound_counter),
        'prev_mae': float(prev_mae),
        'train_losses': train_losses,
        'val_metrics': val_metrics,
        'epochs': [int(x) for x in visualizer.epochs if x is not None],
        'scheduler_type': type(scheduler).__name__ if scheduler else None,
        'use_aux_loss': config.use_aux_loss,
        'fusion_type': config.fusion_type,
        'use_improved_fusion': config.use_improved_fusion,
    }
    
    if hasattr(visualizer, 'train_main_losses'):
        checkpoint['train_main_losses'] = [float(x) for x in visualizer.train_main_losses if x is not None]
    if hasattr(visualizer, 'train_aux_losses'):
        checkpoint['train_aux_losses'] = [float(x) for x in visualizer.train_aux_losses if x is not None]
    
    if analyzer is not None:
        checkpoint['correlation_history'] = analyzer.history
    
    temp_path = latest_path + '.tmp'
    try:
        torch.save(checkpoint, temp_path)
        os.replace(temp_path, latest_path)
        print(f"✓ Latest checkpoint saved to {latest_path}")
    except Exception as e:
        print(f"Warning: Could not save latest checkpoint: {e}")


def load_checkpoint_triple(config, model, optimizer, scheduler, visualizer, scaler=None, analyzer=None):
    if not config.resume_training or not os.path.exists(config.resume_checkpoint):
        return 1, float('inf'), 0, 0, 0, float('inf')
    
    print(f"\n{'='*40}")
    print(f"Resuming training from checkpoint: {config.resume_checkpoint}")
    print(f"{'='*40}")
    
    try:
        checkpoint = torch.load(config.resume_checkpoint, map_location=config.device)
    except Exception as e:
        print(f"Error loading checkpoint: {e}")
        return 1, float('inf'), 0, 0, 0, float('inf')
    
    if 'model_state_dict' in checkpoint:
        try:
            model.load_state_dict(checkpoint['model_state_dict'])
            print("✓ Model weights loaded")
        except Exception as e:
            print(f"Error loading model weights: {e}")
            return 1, float('inf'), 0, 0, 0, float('inf')
    else:
        print("Warning: No model state dict found in checkpoint")
        return 1, float('inf'), 0, 0, 0, float('inf')
    
    if 'optimizer_state_dict' in checkpoint and optimizer is not None:
        try:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            print("✓ Optimizer state loaded")
        except Exception as e:
            print(f"Warning: Could not load optimizer state: {e}")
    
    if scheduler and 'scheduler_state_dict' in checkpoint and checkpoint['scheduler_state_dict']:
        try:
            saved_type = checkpoint.get('scheduler_type')
            current_type = type(scheduler).__name__
            
            if saved_type == current_type:
                if hasattr(scheduler, 'load_state_dict'):
                    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
                    print("✓ Scheduler state loaded")
            else:
                print(f"Warning: Scheduler type mismatch (saved: {saved_type}, current: {current_type})")
        except Exception as e:
            print(f"Warning: Could not load scheduler state: {e}")
    
    if scaler and 'scaler_state_dict' in checkpoint and checkpoint['scaler_state_dict']:
        try:
            scaler.load_state_dict(checkpoint['scaler_state_dict'])
            print("✓ Scaler state loaded")
        except Exception as e:
            print(f"Warning: Could not load scaler state: {e}")

    if visualizer:
        if 'train_losses' in checkpoint:
            visualizer.train_losses = checkpoint['train_losses']
        if 'val_metrics' in checkpoint:
            for k, v in checkpoint['val_metrics'].items():
                if k in visualizer.val_metrics:
                    visualizer.val_metrics[k] = v
        if 'epochs' in checkpoint:
            visualizer.epochs = checkpoint['epochs']
        
        if hasattr(visualizer, 'train_main_losses') and 'train_main_losses' in checkpoint:
            visualizer.train_main_losses = checkpoint['train_main_losses']
        if hasattr(visualizer, 'train_aux_losses') and 'train_aux_losses' in checkpoint:
            visualizer.train_aux_losses = checkpoint['train_aux_losses']

    start_epoch = checkpoint.get('epoch', 0) + 1
    best_mae = checkpoint.get('best_mae', float('inf'))
    best_epoch = checkpoint.get('best_epoch', 0)
    patience_counter = checkpoint.get('patience_counter', 0)
    rebound_counter = checkpoint.get('rebound_counter', 0)
    prev_mae = checkpoint.get('prev_mae', float('inf'))
    
    if analyzer is not None and 'correlation_history' in checkpoint:
        analyzer.history = checkpoint['correlation_history']
        print("✓ Correlation history loaded")
    
    print(f"Resumed from epoch {checkpoint.get('epoch', 0)}")
    print(f"Best MAE so far: {best_mae:.4f} at epoch {best_epoch}")
    print(f"Patience counter: {patience_counter}/{config.patience}")
    
    return start_epoch, best_mae, best_epoch, patience_counter, rebound_counter, prev_mae