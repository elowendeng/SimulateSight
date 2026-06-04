"""
Single-stream model training (using only Imgs)
"""

import os
os.environ['NO_ALBUMENTATIONS_UPDATE'] = '1'

import sys
import argparse
import traceback

import torch
import torch.optim as optim
from torch.utils.data import DataLoader

import warnings
warnings.filterwarnings('ignore')

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config import Config_Single
from utils import set_seed, create_scheduler, TrainingVisualizer, cleanup, print_training_summary_simple
from dataset import get_train_transforms_single, SingleStreamDataset, collate_fn_single
from models import AttentionSingleStreamCOD
from train import COD_Loss, train_epoch_single, validate_batch_single, save_latest_checkpoint_single, load_checkpoint_single


def parse_arguments():
    parser = argparse.ArgumentParser(description='Single-Stream Attention COD Training')
    parser.add_argument('--train_img_dir', type=str, help='Training images directory')
    parser.add_argument('--train_gt_dir', type=str, help='Training ground truth directory')
    parser.add_argument('--val_img_dir', type=str, help='Validation images directory')
    parser.add_argument('--val_gt_dir', type=str, help='Validation ground truth directory')
    parser.add_argument('--save_dir', type=str, help='Directory to save checkpoints')
    parser.add_argument('--img_size', type=int, help='Input image size')
    parser.add_argument('--batch_size', type=int, help='Batch size')
    parser.add_argument('--num_epochs', type=int, help='Number of epochs')
    parser.add_argument('--learning_rate', type=float, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, help='Weight decay')
    parser.add_argument('--attn_type', type=str, choices=['spatial', 'channel', 'dual', 'multi', 'none'], help='Attention type')
    parser.add_argument('--use_cosine_annealing', action='store_true', default=True, help='Use cosine_annealing')
    parser.add_argument('--patience', type=int, help='Patience for early stopping')
    parser.add_argument('--rebound_threshold', type=int, help='Rebound threshold')
    parser.add_argument('--resume_training', type=bool, help='Whether to resume training')
    parser.add_argument('--resume_checkpoint', type=str, help='Checkpoint path to resume from')
    parser.add_argument('--use_augmentation', type=bool, help='Use data augmentation')
    parser.add_argument('--loss_alpha', type=float, default=0.7, help='BCE loss weight')
    parser.add_argument('--loss_beta', type=float, default=0.3, help='IoU loss weight')
    return parser.parse_args()


# ==================== main function ====================
def main():
    args = parse_arguments()
    config = Config_Single(args)
    set_seed(42)

    print(f"{'=' * 60}")
    print(f"Single-Stream Attention COD Training")
    print(f"{'=' * 60}")
    print(f"Device: {config.device}")
    print(f"Attention Type: {config.attn_type}")
    print(f"Batch Size: {config.batch_size}")
    print(f"Image Size: {config.img_size}")
    print(f"Use AMP: {config.use_amp}")
    print(f"Data Augmentation: {config.use_augmentation}")
    print(f"Warmup Epochs: {config.warmup_epochs}")
    print(f"Resume Training: {config.resume_training}")
    if config.use_improved_attn:
        print(f"Using Improved Attention")
    else:
        print(f"Using Original Attention")
    print(f"{'=' * 60}\n")

    os.makedirs(config.save_dir, exist_ok=True)

    train_transform = get_train_transforms_single(config) if config.use_augmentation else None
    val_transform = None

    print("Loading datasets...")
    train_dataset = SingleStreamDataset(
        config.train_img_dir, config.train_gt_dir, config.img_size,
        transform=train_transform, is_train=True
    )
    val_dataset = SingleStreamDataset(
        config.val_img_dir, config.val_gt_dir, config.img_size,
        transform=val_transform, is_train=False
    )

    print(f"Train samples: {len(train_dataset)}")
    print(f"Val samples: {len(val_dataset)}")

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=True,
        collate_fn=collate_fn_single,
        persistent_workers=True if config.num_workers > 0 else False
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=True,
        collate_fn=collate_fn_single,
        persistent_workers=True if config.num_workers > 0 else False
    )

    print("\nCreating model...")
    model = AttentionSingleStreamCOD(
        pretrained=True, attn_type=config.attn_type, input_size=config.img_size, use_improved=config.use_improved_attn
    ).to(config.device)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params / 1e6:.2f}M")
    print(f"Trainable parameters: {trainable_params / 1e6:.2f}M")

    optimizer = optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)

    steps_per_epoch = len(train_loader)
    scheduler = create_scheduler(optimizer, config, steps_per_epoch=steps_per_epoch)

    criterion = COD_Loss(alpha=config.loss_alpha, beta=config.loss_beta)
    scaler = torch.cuda.amp.GradScaler() if config.use_amp else None

    visualizer = TrainingVisualizer(config.save_dir, config.plot_metrics)

    if config.resume_training and os.path.exists(config.resume_checkpoint):
        start_epoch, best_mae, best_epoch, patience_counter, rebound_counter, prev_mae = load_checkpoint_single(
            config, model, optimizer, scheduler, visualizer, scaler
        )
    else:
        start_epoch, best_mae, best_epoch, patience_counter, rebound_counter, prev_mae = 1, float(
            'inf'), 0, 0, 0, float('inf')

    for epoch in range(start_epoch, config.num_epochs + 1):
        print(f"\n{'=' * 40}")
        print(f"Epoch {epoch}/{config.num_epochs}")
        print(f"{'=' * 40}")

        train_loss = train_epoch_single(model, train_loader, optimizer, criterion, epoch, config, scaler)
        print(f"Train Loss: {train_loss:.4f}")

        train_dataset.clear_cache()

        val_metrics = validate_batch_single(model, val_loader, config)
        print(f"Validation:")
        for k, v in val_metrics.items():
            print(f"  {k}: {v:.4f}")

        current_mae = val_metrics['mae']

        if current_mae < best_mae:
            best_mae = current_mae
            best_epoch = epoch
            torch.save(model.state_dict(), os.path.join(config.save_dir, 'best_model.pth'))
            print(f"✓ New best model! MAE: {best_mae:.4f}")

            patience_counter = 0
            rebound_counter = 0

            with open(os.path.join(config.save_dir, 'best_metrics.txt'), 'w') as f:
                f.write(f"Best model at epoch {epoch}\n")
                for k, v in val_metrics.items():
                    f.write(f"{k}: {v:.6f}\n")
        else:
            if current_mae > prev_mae:
                rebound_counter += 1
                print(f"MAE rebound {rebound_counter}/{config.rebound_threshold} times")
            else:
                if rebound_counter > 0:
                    print(f"✓ MAE stopped bouncing")
                rebound_counter = 0

            patience_counter += 1
            print(f"No improvement. {patience_counter}/{config.patience} epochs")

            if patience_counter >= config.patience:
                print(f"\n{'!' * 50}")
                print(f"EARLY STOPPING")
                print(f"Best MAE: {best_mae:.4f} at epoch {best_epoch}")
                print(f"{'!' * 50}")

                try:
                    if config.save_latest:
                        save_latest_checkpoint_single(config, epoch, model, optimizer, scheduler, val_metrics,
                                               train_loss, best_mae, best_epoch, patience_counter,
                                               rebound_counter, current_mae, visualizer, scaler)
                except Exception as e:
                    print(f"Warning: Could not save final checkpoint: {e}")
                break

        prev_mae = current_mae

        if epoch % config.plot_interval == 0:
            visualizer.update(epoch, train_loss, val_metrics)

        try:
            if scheduler is not None:
                if isinstance(scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                    scheduler.step(val_metrics['mae'])
                else:
                    scheduler.step()
        except Exception as e:
            print(f"Warning: Error in scheduler step: {e}")

        try:
            if hasattr(scheduler, 'get_last_lr'):
                current_lr = scheduler.get_last_lr()[0]
            else:
                current_lr = optimizer.param_groups[0]['lr']
        except Exception as e:
            current_lr = optimizer.param_groups[0]['lr']

        print(f"Current LR: {current_lr:.2e}")

        if epoch % config.save_interval == 0:
            checkpoint_path = os.path.join(config.save_dir, f'checkpoint_epoch{epoch}.pth')

            try:
                scheduler_state = scheduler.state_dict() if hasattr(scheduler, 'state_dict') else None
            except Exception:
                scheduler_state = None

            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler_state,
                'metrics': val_metrics,
                'train_loss': train_loss,
                'best_mae': best_mae,
                'best_epoch': best_epoch,
                'patience_counter': patience_counter,
                'rebound_counter': rebound_counter,
                'prev_mae': prev_mae,
                'scheduler_type': type(scheduler).__name__ if scheduler else None,
            }

            temp_path = checkpoint_path + '.tmp'
            torch.save(checkpoint, temp_path)
            os.replace(temp_path, checkpoint_path)
            print(f"Checkpoint saved to {checkpoint_path}")

        save_latest_checkpoint_single(config, epoch, model, optimizer, scheduler, val_metrics, train_loss,
                               best_mae, best_epoch, patience_counter, rebound_counter,
                               current_mae, visualizer, scaler)

        if epoch % 5 == 0:
            cleanup()

    visualizer.save_final()
    print_training_summary_simple(config, best_mae, best_epoch, visualizer)


if __name__ == '__main__':
    try:
        if torch.cuda.is_available():
            torch.backends.cudnn.benchmark = True
            print(f"GPU: {torch.cuda.get_device_name(0)}")
            print(f"CUDA Version: {torch.version.cuda}")
        main()
    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
        try:
            if 'model' in locals() and 'config' in locals():
                torch.save(model.state_dict(), os.path.join(config.save_dir, 'interrupted_model.pth'))
                print(f"Model saved to {os.path.join(config.save_dir, 'interrupted_model.pth')}")
        except:
            pass
    except Exception as e:
        print(f"\nTraining failed with error: {e}")
        traceback.print_exc()
    finally:
        cleanup()