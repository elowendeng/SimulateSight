"""
Triple-stream model training (using Imgs + Imgs_CB + Imgs SL)
"""

import os
os.environ['NO_ALBUMENTATIONS_UPDATE'] = '1'

import sys
import argparse
import traceback
import numpy as np

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import warnings
warnings.filterwarnings('ignore')

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config import Config_Triple
from utils import set_seed, create_scheduler, TrainingVisualizer, TrainingVisualizer_aux, cleanup, print_training_summary, TripleStreamCorrelationAnalyzer
from dataset import get_train_transforms_triple, TripleStreamDataset, collate_fn_triple
from models import AttentionTripleStreamCOD
from train import COD_Loss, COD_Loss_With_Aux, train_epoch_triple, train_epoch_triple_with_aux, validate_batch_triple, save_latest_checkpoint_triple, load_checkpoint_triple


def parse_arguments():
    parser = argparse.ArgumentParser(description='Triple-Stream COD Training')
    
    parser.add_argument('--train_img_dir', type=str, help='Training images directory')
    parser.add_argument('--train_cb_dir', type=str, help='Training CB images directory')
    parser.add_argument('--train_sl_dir', type=str, help='Training SL images directory')
    parser.add_argument('--train_gt_dir', type=str, help='Training ground truth directory')
    parser.add_argument('--val_img_dir', type=str, help='Validation images directory')
    parser.add_argument('--val_cb_dir', type=str, help='Validation CB images directory')
    parser.add_argument('--val_sl_dir', type=str, help='Validation SL images directory')
    parser.add_argument('--val_gt_dir', type=str, help='Validation ground truth directory')
    parser.add_argument('--save_dir', type=str, help='Directory to save checkpoints')
    
    parser.add_argument('--img_size', type=int, help='Input image size')
    parser.add_argument('--batch_size', type=int, help='Batch size')
    parser.add_argument('--num_epochs', type=int, help='Number of epochs')
    parser.add_argument('--learning_rate', type=float, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, help='Weight decay')
    parser.add_argument('--aux_loss_weight', type=float, help='Weight for auxiliary loss')
    parser.add_argument('--patience', type=int, help='Patience for early stopping')
    parser.add_argument('--rebound_threshold', type=int, help='Rebound threshold')
    parser.add_argument('--resume_checkpoint', type=str, help='Checkpoint path to resume from')
    parser.add_argument('--sl_supplement_strength', type=float, default=0.2, 
                        help='SL supplement strength (for hierarchical fusion)')
    
    parser.add_argument('--share_backbone', action='store_true', default=True, 
                        help='Share backbone weights')
    parser.add_argument('--no_share_backbone', action='store_true', default=False, 
                        help='Do not share backbone weights')
    
    parser.add_argument('--use_improved_fusion', action='store_false', default=True,
                        help='Disable improved fusion (enabled by default)')
    
    parser.add_argument('--use_aux_loss', action='store_true', default=False,
                        help='Use auxiliary loss')
    
    parser.add_argument('--resume_training', action='store_true', default=True,
                        help='Resume training')
    parser.add_argument('--no_resume', action='store_true', default=False,
                        help='Do not resume training')
    
    parser.add_argument('--use_augmentation', action='store_true', default=True,
                        help='Use data augmentation')
    parser.add_argument('--no_augmentation', action='store_true', default=False,
                        help='Disable data augmentation')
    
    parser.add_argument('--enable_correlation_check', action='store_true', default=False,
                        help='Enable triple-stream correlation analysis')
    
    parser.add_argument('--use_cosine_annealing', action='store_true', default=True,
                        help='Use cosine_annealing')
 
    parser.add_argument('--fusion_type', type=str, 
                        choices=['direct', 'hierarchical', 'gated', 'residual'], 
                        help='Fusion type')
    
    return parser.parse_args()


# ==================== main function ====================
def main():
    args = parse_arguments()
    
    if args.no_share_backbone:
        args.share_backbone = False
    
    if args.no_resume:
        args.resume_training = False
    
    if args.no_augmentation:
        args.use_augmentation = False

    config = Config_Triple(args)
    set_seed(42)
    
    print(f"{'='*60}")
    print(f"Attention-based Triple-Stream COD Training")
    print(f"{'='*60}")
    print(f"Device: {config.device}")
    print(f"Fusion Type: {config.fusion_type}")
    if config.fusion_type == 'direct':
        print(f"  Use Improved Fusion: {config.use_improved_fusion}")
        if config.use_improved_fusion:
            print(f"  → Using ImprovedTripleAttentionFusion (learnable fusion coefficients)")
        else:
            print(f"  → Using TripleAttentionFusion (original with gate mechanism)")
    elif config.fusion_type == 'hierarchical':
        print(f"  Supplement Strength: {config.sl_supplement_strength}")
    print(f"Share Backbone: {config.share_backbone}")
    print(f"Use Aux Loss: {config.use_aux_loss}")
    if config.use_aux_loss:
        print(f"  Aux Loss Weight: {config.aux_loss_weight}")
    print(f"Batch Size: {config.batch_size}")
    print(f"Image Size: {config.img_size}")
    print(f"Learning Rate: {config.learning_rate}")
    print(f"Gradient Clip Norm: {config.grad_clip_norm}")
    print(f"Warmup Epochs: {config.warmup_epochs}")
    print(f"Use AMP: {config.use_amp}")
    print(f"Data Augmentation: {config.use_augmentation}")
    print(f"Correlation Analysis: {config.enable_correlation_check}")
    if config.enable_correlation_check:
        print(f"  Check Interval: {config.correlation_check_interval}")
    print(f"Resume Training: {config.resume_training}")
    print(f"Save directory: {config.save_dir}")
    print(f"{'='*60}\n")
    
    os.makedirs(config.save_dir, exist_ok=True)
    
    try:
        train_transform = get_train_transforms_triple(config) if config.use_augmentation else None
        val_transform = None
        
        print("Loading datasets...")
        train_dataset = TripleStreamDataset(
            config.train_img_dir, config.train_cb_dir, config.train_sl_dir, config.train_gt_dir,
            config.img_size, transform=train_transform, is_train=True
        )
        val_dataset = TripleStreamDataset(
            config.val_img_dir, config.val_cb_dir, config.val_sl_dir, config.val_gt_dir,
            config.img_size, transform=val_transform, is_train=False
        )
        
        print(f"Train samples: {len(train_dataset)}")
        print(f"Val samples: {len(val_dataset)}")
        
        train_loader = DataLoader(
            train_dataset, batch_size=config.batch_size, shuffle=True,
            num_workers=config.num_workers, pin_memory=True, collate_fn=collate_fn_triple,
            persistent_workers=True if config.num_workers > 0 else False
        )
        val_loader = DataLoader(
            val_dataset, batch_size=config.batch_size, shuffle=False,
            num_workers=config.num_workers, pin_memory=True, collate_fn=collate_fn_triple,
            persistent_workers=True if config.num_workers > 0 else False
        )
        
        print("\nCreating model...")
        
        model = AttentionTripleStreamCOD(
            pretrained=True,
            input_size=config.img_size,
            share_backbone=config.share_backbone,
            use_improved=config.use_improved_fusion,
            use_aux_loss=config.use_aux_loss,
            fusion_type=config.fusion_type,
            supplement_strength=config.sl_supplement_strength
        ).to(config.device)
        
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Total parameters: {total_params/1e6:.2f}M")
        print(f"Trainable parameters: {trainable_params/1e6:.2f}M")
        
        optimizer = optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
        
        steps_per_epoch = len(train_loader)
        scheduler = create_scheduler(optimizer, config, steps_per_epoch=steps_per_epoch)
        
        if config.use_aux_loss:
            criterion = COD_Loss_With_Aux(
                aux_weight=config.aux_loss_weight,
                alpha=config.loss_alpha,
                beta=config.loss_beta
            )
            train_func = train_epoch_triple_with_aux
        else:
            criterion = COD_Loss(alpha=config.loss_alpha, beta=config.loss_beta)
            train_func = train_epoch_triple
        
        scaler = torch.cuda.amp.GradScaler() if config.use_amp else None
        
        if config.use_aux_loss:
            visualizer = TrainingVisualizer_aux(config.save_dir, config.plot_metrics)
            print("Using TrainingVisualizer_aux (with auxiliary loss support)")
        else:
            visualizer = TrainingVisualizer(config.save_dir, config.plot_metrics)
            print("Using TrainingVisualizer (standard)")
        
        if config.enable_correlation_check:
            correlation_analyzer = TripleStreamCorrelationAnalyzer(config.device)
            print("Correlation analyzer enabled")
        else:
            correlation_analyzer = None
            print("Correlation analyzer disabled")
        
        if config.resume_training and os.path.exists(config.resume_checkpoint):
            start_epoch, best_mae, best_epoch, patience_counter, rebound_counter, prev_mae = load_checkpoint_triple(
                config, model, optimizer, scheduler, visualizer, scaler, correlation_analyzer
            )
        else:
            start_epoch, best_mae, best_epoch, patience_counter, rebound_counter, prev_mae = 1, float('inf'), 0, 0, 0, float('inf')
        
        for epoch in range(start_epoch, config.num_epochs + 1):
            print(f"\n{'='*40}")
            print(f"Epoch {epoch}/{config.num_epochs}")
            print(f"{'='*40}")
            
            train_metrics = train_func(model, train_loader, optimizer, criterion, epoch, config, scaler)
            
            if config.use_aux_loss:
                print(f"Train Loss - Total: {train_metrics['total_loss']:.4f}, "
                      f"Main: {train_metrics['main_loss']:.4f}, "
                      f"Aux: {train_metrics['aux_loss']:.4f}")
            else:
                print(f"Train Loss: {train_metrics:.4f}")
            
            train_dataset.clear_cache()
            
            val_metrics = validate_batch_triple(model, val_loader, config)
            print(f"Validation:")
            for k, v in val_metrics.items():
                print(f"  {k}: {v:.4f}")
            
            if correlation_analyzer is not None and epoch % config.correlation_check_interval == 0:
                print("\nAnalyzing triple-stream correlation...")
                try:
                    corr_results = correlation_analyzer.analyze(model, val_loader, config, subset_size=50)
                    
                    if corr_results:
                        print(f"  Image-CB Correlation:   {corr_results['img_cb_correlation']:.4f}")
                        print(f"  Image-SL Correlation:   {corr_results['img_sl_correlation']:.4f}")
                        print(f"  CB-SL Correlation:      {corr_results['cb_sl_correlation']:.4f}")
                        print(f"  Diversity Score:        {corr_results['diversity_score']:.4f}")
                        print(f"  Is Complementary:       {'✓ Yes' if corr_results['is_complementary'] else '✗ No'}")
                        
                        val_metrics['diversity_score'] = corr_results['diversity_score']
                        val_metrics['img_cb_corr'] = corr_results['img_cb_correlation']
                        val_metrics['img_sl_corr'] = corr_results['img_sl_correlation']
                        val_metrics['cb_sl_corr'] = corr_results['cb_sl_correlation']
                        
                except Exception as e:
                    print(f"  ✗ Correlation analysis failed: {e}")
            
            if correlation_analyzer is not None and epoch % 10 == 0 and len(correlation_analyzer.history['diversity_score']) > 0:
                avg_diversity = np.mean(correlation_analyzer.history['diversity_score'][-10:])
                print(f"\n  [Summary] Last 10 epochs avg diversity: {avg_diversity:.4f}")
            
            current_mae = val_metrics['mae']
            
            if current_mae < best_mae:
                best_mae = current_mae
                best_epoch = epoch
                torch.save(model.state_dict(), os.path.join(config.save_dir, 'best_model.pth'))
                print(f"✓ New best model! MAE: {best_mae:.4f}")
                
                with open(os.path.join(config.save_dir, 'best_metrics.txt'), 'w') as f:
                    f.write(f"Best model at epoch {epoch}\n")
                    f.write(f"Fusion Type: {config.fusion_type}\n")
                    if config.fusion_type == 'direct':
                        f.write(f"Use Improved: {config.use_improved_fusion}\n")
                    for k, v in val_metrics.items():
                        f.write(f"{k}: {v:.6f}\n")
                
                patience_counter = 0
                rebound_counter = 0
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
                    print(f"\n{'!'*50}")
                    print(f"EARLY STOPPING")
                    print(f"Best MAE: {best_mae:.4f} at epoch {best_epoch}")
                    print(f"{'!'*50}")
                    
                    save_latest_checkpoint_triple(config, epoch, model, optimizer, scheduler, val_metrics, train_metrics,
                                         best_mae, best_epoch, patience_counter, rebound_counter,
                                         current_mae, visualizer, scaler, correlation_analyzer)
                    break
            
            prev_mae = current_mae
            
            if epoch % config.plot_interval == 0:
                visualizer.update(epoch, train_metrics, val_metrics)
            
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
            except Exception:
                current_lr = optimizer.param_groups[0]['lr']
            
            print(f"Current LR: {current_lr:.2e}")
            
            if epoch % config.save_interval == 0:
                checkpoint_path = os.path.join(config.save_dir, f'checkpoint_epoch{epoch}.pth')
                
                try:
                    scheduler_state = scheduler.state_dict() if hasattr(scheduler, 'state_dict') else None
                except Exception:
                    scheduler_state = None
                
                if config.use_aux_loss:
                    train_loss_val = train_metrics['total_loss']
                    train_main_loss_val = train_metrics['main_loss']
                    train_aux_loss_val = train_metrics['aux_loss']
                else:
                    train_loss_val = train_metrics
                    train_main_loss_val = None
                    train_aux_loss_val = None
                
                checkpoint = {
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler_state,
                    'scaler_state_dict': scaler.state_dict() if scaler else None,
                    'metrics': val_metrics,
                    'train_loss': train_loss_val,
                    'train_main_loss': train_main_loss_val,
                    'train_aux_loss': train_aux_loss_val,
                    'best_mae': best_mae,
                    'best_epoch': best_epoch,
                    'patience_counter': patience_counter,
                    'rebound_counter': rebound_counter,
                    'prev_mae': prev_mae,
                    'scheduler_type': type(scheduler).__name__ if scheduler else None,
                    'use_aux_loss': config.use_aux_loss,
                    'fusion_type': config.fusion_type,
                    'use_improved_fusion': config.use_improved_fusion,
                }
                if correlation_analyzer is not None:
                    checkpoint['correlation_history'] = correlation_analyzer.history
                
                if hasattr(visualizer, 'train_main_losses'):
                    checkpoint['train_main_losses'] = visualizer.train_main_losses
                if hasattr(visualizer, 'train_aux_losses'):
                    checkpoint['train_aux_losses'] = visualizer.train_aux_losses
                
                checkpoint['train_losses'] = visualizer.train_losses
                checkpoint['val_metrics'] = {k: v for k, v in visualizer.val_metrics.items()}
                checkpoint['epochs'] = visualizer.epochs
                
                temp_path = checkpoint_path + '.tmp'
                torch.save(checkpoint, temp_path)
                os.replace(temp_path, checkpoint_path)
                print(f"Checkpoint saved to {checkpoint_path}")
            
            save_latest_checkpoint_triple(config, epoch, model, optimizer, scheduler, val_metrics, train_metrics,
                                 best_mae, best_epoch, patience_counter, rebound_counter,
                                 current_mae, visualizer, scaler, correlation_analyzer)
            
            if epoch % 5 == 0:
                cleanup()
        
        visualizer.save_final()
        print_training_summary(config, best_mae, best_epoch, visualizer)
        if correlation_analyzer:
            correlation_analyzer.print_summary()
        
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