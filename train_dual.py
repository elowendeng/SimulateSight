"""
Dual-stream model training (using Imgs + Imgs_CB)
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

from config import Config_Dual
from utils import set_seed, create_scheduler, TrainingVisualizer, TrainingVisualizer_aux, cleanup, print_training_summary, DualStreamCorrelationAnalyzer
from dataset import get_train_transforms_dual, DualStreamDataset, collate_fn_dual
from models import AttentionDualStreamCOD, AttentionDualStreamCODWithAux
from train import COD_Loss, COD_Loss_With_Aux, train_epoch_dual, train_epoch_dual_with_aux, validate_batch_dual, save_latest_checkpoint_dual, load_checkpoint_dual


def parse_arguments():
    parser = argparse.ArgumentParser(description='Dual-Stream Attention COD Training')
    
    parser.add_argument('--train_img_dir', type=str, help='Training images directory')
    parser.add_argument('--train_cb_dir', type=str, help='Training CB images directory')
    parser.add_argument('--train_gt_dir', type=str, help='Training ground truth directory')
    parser.add_argument('--val_img_dir', type=str, help='Validation images directory')
    parser.add_argument('--val_cb_dir', type=str, help='Validation CB images directory')
    parser.add_argument('--val_gt_dir', type=str, help='Validation ground truth directory')
    parser.add_argument('--save_dir', type=str, help='Directory to save checkpoints')
    
    parser.add_argument('--aux_loss_weight', type=float, default=0.3, help='Weight for auxiliary loss')
    parser.add_argument('--img_size', type=int, help='Input image size')
    parser.add_argument('--batch_size', type=int, help='Batch size')
    parser.add_argument('--num_epochs', type=int, help='Number of epochs')
    parser.add_argument('--learning_rate', type=float, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, help='Weight decay')
    parser.add_argument('--attn_type', type=str, choices=['spatial', 'channel', 'dual', 'multi', 'none'], 
                        help='Attention type')
    parser.add_argument('--patience', type=int, help='Patience for early stopping')
    parser.add_argument('--rebound_threshold', type=int, help='Rebound threshold')
    parser.add_argument('--resume_checkpoint', type=str, help='Checkpoint path to resume from')
    parser.add_argument('--loss_alpha', type=float, default=0.7, help='BCE loss weight')
    parser.add_argument('--loss_beta', type=float, default=0.3, help='IoU loss weight')
    
    parser.add_argument('--use_aux_loss', action='store_true', default=False, 
                        help='Use auxiliary loss (disabled by default)')
    parser.add_argument('--share_backbone', action='store_true', default=True, 
                        help='Share backbone weights (enabled by default)')
    parser.add_argument('--no_share_backbone', action='store_true', default=False, 
                        help='Do not share backbone weights')
    parser.add_argument('--use_amp', action='store_true', default=True, 
                        help='Use mixed precision training (enabled by default)')
    parser.add_argument('--no_amp', action='store_true', default=False, 
                        help='Disable mixed precision training')
    parser.add_argument('--resume_training', action='store_true', default=True, 
                        help='Resume training from checkpoint (enabled by default)')
    parser.add_argument('--no_resume', action='store_true', default=False, 
                        help='Do not resume training')
    parser.add_argument('--use_augmentation', action='store_true', default=True, 
                        help='Use data augmentation (enabled by default)')
    parser.add_argument('--no_augmentation', action='store_true', default=False, 
                        help='Disable data augmentation')
    parser.add_argument('--enable_complementarity', action='store_true', default=False, 
                        help='Enable complementarity check (disabled by default)')
    parser.add_argument('--use_cosine_annealing', action='store_true', default=True,
                        help='Use cosine_annealing')
    
    return parser.parse_args()


# ==================== main function ====================
def main():
    args = parse_arguments()
    
    if args.no_share_backbone:
        args.share_backbone = False
    
    if args.no_amp:
        args.use_amp = False
    
    if args.no_resume:
        args.resume_training = False
    
    if args.no_augmentation:
        args.use_augmentation = False
    
    config = Config_Dual(args)
    set_seed(42)

    print(f"{'=' * 60}")
    print(f"Attention-based Dual-Stream COD Training")
    print(f"{'=' * 60}")
    print(f"Device: {config.device}")
    print(f"Attention Type: {config.attn_type}")
    print(f"Share Backbone: {config.share_backbone}")
    print(f"Batch Size: {config.batch_size}")
    print(f"Image Size: {config.img_size}")
    print(f"Use AMP: {config.use_amp}")
    print(f"Data Augmentation: {config.use_augmentation}")
    print(f"Complementarity Check: {config.enable_complementarity_check}")
    print(f"Resume Training: {config.resume_training}")
    print(f"Use Aux Loss: {config.use_aux_loss}")
    if config.use_aux_loss:
        print(f"  Aux Loss Weight: {config.aux_loss_weight}")
    if config.use_improved_fusion:
        print(f"Using Improved Fusion")
    else:
        print(f"Using Original Fusion")
    print(f"Loss Alpha/Beta: {config.loss_alpha}/{config.loss_beta}")
    print(f"Save directory: {config.save_dir}")
    print(f"{'=' * 60}\n")

    os.makedirs(config.save_dir, exist_ok=True)

    # DataSet
    train_transform = get_train_transforms_dual(config) if config.use_augmentation else None
    val_transform = None

    print("Loading datasets...")
    train_dataset = DualStreamDataset(
        config.train_img_dir, config.train_cb_dir, config.train_gt_dir,
        config.img_size, is_train=True, transform=train_transform
    )
    val_dataset = DualStreamDataset(
        config.val_img_dir, config.val_cb_dir, config.val_gt_dir,
        config.img_size, is_train=False, transform=val_transform
    )

    print(f"Train samples: {len(train_dataset)}")
    print(f"Val samples: {len(val_dataset)}")

    # DataLoader with persistent_workers
    train_loader = DataLoader(
        train_dataset, batch_size=config.batch_size, shuffle=True,
        num_workers=config.num_workers, pin_memory=True, collate_fn=collate_fn_dual,
        persistent_workers=True if config.num_workers > 0 else False
    )
    val_loader = DataLoader(
        val_dataset, batch_size=config.batch_size, shuffle=False,
        num_workers=config.num_workers, pin_memory=True, collate_fn=collate_fn_dual,
        persistent_workers=True if config.num_workers > 0 else False
    )

    # create model
    print("\nCreating model...")

    if config.use_aux_loss:
        model = AttentionDualStreamCODWithAux(
            pretrained=True,
            attn_type=config.attn_type,
            input_size=config.img_size,
            share_backbone=config.share_backbone,
            use_improved=config.use_improved_fusion,
            use_aux_loss=config.use_aux_loss
        ).to(config.device)
    else:
        model = AttentionDualStreamCOD(
            pretrained=True,
            attn_type=config.attn_type,
            input_size=config.img_size,
            share_backbone=config.share_backbone,
            use_improved=config.use_improved_fusion
        ).to(config.device)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params / 1e6:.2f}M")
    print(f"Trainable parameters: {trainable_params / 1e6:.2f}M")

    # optimizer
    optimizer = optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)

    # learning rate scheduler（step-based）
    steps_per_epoch = len(train_loader)
    scheduler = create_scheduler(optimizer, config, steps_per_epoch=steps_per_epoch)

    # loss function
    if config.use_aux_loss:
        criterion = COD_Loss_With_Aux(
            aux_weight=config.aux_loss_weight,
            alpha=config.loss_alpha,
            beta=config.loss_beta
        )
        train_func = train_epoch_dual_with_aux
    else:
        criterion = COD_Loss(alpha=config.loss_alpha, beta=config.loss_beta)
        train_func = train_epoch_dual

    scaler = torch.cuda.amp.GradScaler() if config.use_amp else None

    # visualization
    if config.use_aux_loss:
        visualizer = TrainingVisualizer_aux(config.save_dir, config.plot_metrics)
    else:
        visualizer = TrainingVisualizer(config.save_dir, config.plot_metrics)

    # complementarity analyzer
    if config.enable_complementarity_check:
        analyzer = DualStreamCorrelationAnalyzer(config.device)
    else:
        analyzer = None

    # load checkpoint
    if config.resume_training and os.path.exists(config.resume_checkpoint):
        start_epoch, best_mae, best_epoch, patience_counter, rebound_counter, prev_mae = load_checkpoint_dual(
            config, model, optimizer, scheduler, visualizer, analyzer, scaler
        )
    else:
        start_epoch, best_mae, best_epoch, patience_counter, rebound_counter, prev_mae = 1, float(
            'inf'), 0, 0, 0, float('inf')

    # The Training Cycle
    for epoch in range(start_epoch, config.num_epochs + 1):
        print(f"\n{'=' * 40}")
        print(f"Epoch {epoch}/{config.num_epochs}")
        print(f"{'=' * 40}")

        # training
        if config.use_aux_loss:
            train_metrics = train_func(model, train_loader, optimizer, criterion, epoch, config, scaler)
            current_train_loss = train_metrics['total_loss']
            print(f"Train Loss - Total: {current_train_loss:.4f}, "
                  f"Main: {train_metrics['main_loss']:.4f}, "
                  f"Aux: {train_metrics['aux_loss']:.4f}")
        else:
            train_loss = train_func(model, train_loader, optimizer, criterion, epoch, config, scaler)
            current_train_loss = train_loss
            print(f"Train Loss: {current_train_loss:.4f}")

        # clear the training set cache
        train_dataset.clear_cache()

        # verification
        val_metrics = validate_batch_dual(model, val_loader, config)
        print(f"Validation:")
        for k, v in val_metrics.items():
            print(f"  {k}: {v:.4f}")

        # complementary analysis
        if analyzer is not None and epoch % config.complementarity_check_interval == 0:
            print("\nAnalyzing stream complementarity...")
            try:
                comp_results = analyzer.analyze(model, val_loader, config)
                print(f"  Prediction Correlation: {comp_results['prediction_correlation']:.4f}")
                print(f"  Diversity Score: {comp_results['diversity_score']:.4f}")
                print(f"  Is Complementary: {'✓ Yes' if comp_results['is_complementary'] else '✗ No'}")
                val_metrics['complementarity'] = comp_results['diversity_score']
            except Exception as e:
                print(f"  ✗ Complementarity analysis failed: {e}")

        # early stop logic
        current_mae = val_metrics['mae']

        if current_mae < best_mae:
            best_mae = current_mae
            best_epoch = epoch

            # save the best model
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
                break

        prev_mae = current_mae

        if epoch % config.plot_interval == 0:
            if config.use_aux_loss:
                visualizer.update(epoch, train_metrics, val_metrics)
            else:
                visualizer.update(epoch, train_loss, val_metrics)

        try:
            if hasattr(scheduler, 'step'):
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
        except Exception:
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
                'scaler_state_dict': scaler.state_dict() if scaler else None,
                'metrics': val_metrics,
                'train_loss': float(current_train_loss) if current_train_loss is not None else 0.0,
                'best_mae': float(best_mae),
                'best_epoch': int(best_epoch),
                'patience_counter': int(patience_counter),
                'rebound_counter': int(rebound_counter),
                'prev_mae': float(prev_mae),
                'scheduler_type': type(scheduler).__name__ if scheduler else None,
            }

            if analyzer is not None:
                checkpoint['complementarity_history'] = analyzer.history

            temp_path = checkpoint_path + '.tmp'
            torch.save(checkpoint, temp_path)
            os.replace(temp_path, checkpoint_path)
            print(f"Checkpoint saved to {checkpoint_path}")

        if config.use_aux_loss and 'train_metrics' in locals():
            save_latest_checkpoint_dual(config, epoch, model, optimizer, scheduler, val_metrics,
                                   train_metrics['total_loss'], best_mae, best_epoch,
                                   patience_counter, rebound_counter, current_mae,
                                   visualizer, analyzer, scaler)
        elif not config.use_aux_loss and 'train_loss' in locals():
            save_latest_checkpoint_dual(config, epoch, model, optimizer, scheduler, val_metrics,
                                   train_loss, best_mae, best_epoch, patience_counter,
                                   rebound_counter, current_mae, visualizer, analyzer, scaler)

        if epoch % 5 == 0:
            cleanup()

    visualizer.save_final()
    print_training_summary(config, best_mae, best_epoch, visualizer)


if __name__ == '__main__':
    try:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
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