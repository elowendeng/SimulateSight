import os
import re
import sys
import argparse
import warnings
from collections import defaultdict

import torch
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm

warnings.filterwarnings('ignore')

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from dataset import TripleStreamDataset, DualStreamDataset, SingleStreamDataset, collate_fn_triple, collate_fn_dual, collate_fn_single
from models import AttentionTripleStreamCOD, AttentionDualStreamCOD, AttentionSingleStreamCOD
from train.validator import validate_batch_triple, validate_batch_dual, validate_batch_single
from utils import set_seed, cleanup


def parse_checkpoint_name(checkpoint_dir):
    """
    Parse checkpoint directory name to extract model configuration
    
    Args:
        checkpoint_dir: Directory name (e.g., 'single_dual', 'triple_direct_aux')
    
    Returns:
        dict: {
            'model_type': 'single'/'dual'/'triple',
            'attn_type': attention type (for single/dual),
            'fusion_type': fusion type (for triple),
            'use_aux_loss': bool
        }
    """
    # Pattern: {model_type}_{type}_{aux?}
    # Examples: single_dual, single_spatial_aux, triple_direct, triple_hierarchical_aux
    
    parts = checkpoint_dir.split('_')
    
    if len(parts) < 2:
        raise ValueError(f"Invalid checkpoint directory name: {checkpoint_dir}. "
                        f"Expected format: {model_type}_{type}_[aux]")
    
    model_type = parts[0].lower()
    
    if model_type not in ['single', 'dual', 'triple']:
        raise ValueError(f"Invalid model type: {model_type}. Must be 'single', 'dual', or 'triple'")
    
    # Check if aux is present
    use_aux_loss = 'aux' in parts
    
    # Extract the type (attn_type for single/dual, fusion_type for triple)
    if model_type in ['single', 'dual']:
        # For single/dual: parts[1] should be the attention type
        # Possible values: spatial, channel, dual, none
        if len(parts) >= 2:
            attn_type = parts[1].lower()
            if attn_type not in ['spatial', 'channel', 'dual', 'none']:
                print(f"Warning: Unknown attention type '{attn_type}', using 'dual' as default")
                attn_type = 'dual'
        else:
            attn_type = 'dual'
        
        return {
            'model_type': model_type,
            'attn_type': attn_type,
            'fusion_type': None,
            'use_aux_loss': use_aux_loss
        }
    
    else:  # triple
        # For triple: parts[1] should be the fusion type
        # Possible values: direct, hierarchical, gated, residual
        if len(parts) >= 2:
            fusion_type = parts[1].lower()
            if fusion_type not in ['direct', 'hierarchical', 'gated', 'residual']:
                print(f"Warning: Unknown fusion type '{fusion_type}', using 'direct' as default")
                fusion_type = 'direct'
        else:
            fusion_type = 'direct'
        
        return {
            'model_type': model_type,
            'attn_type': None,
            'fusion_type': fusion_type,
            'use_aux_loss': use_aux_loss
        }


class TestConfig:
    """
    Unified configuration class for testing
    Automatically parses model configuration from checkpoint directory name
    """
    def __init__(self, checkpoint_dir=None, args=None, **kwargs):
        """
        Initialize test configuration
        
        Args:
            checkpoint_dir: Checkpoint directory name (will be parsed automatically)
            args: Parsed command line arguments (optional)
            **kwargs: Additional configuration overrides
        """
        # Parse checkpoint directory name
        if checkpoint_dir is not None:
            self.checkpoint_dir = checkpoint_dir
            parsed_config = parse_checkpoint_name(checkpoint_dir)
            self.model_type = parsed_config['model_type']
            self.attn_type = parsed_config['attn_type']
            self.fusion_type = parsed_config['fusion_type']
            self.use_aux_loss = parsed_config['use_aux_loss']
        elif args is not None and args.checkpoint_dir is not None:
            self.checkpoint_dir = args.checkpoint_dir
            parsed_config = parse_checkpoint_name(args.checkpoint_dir)
            self.model_type = parsed_config['model_type']
            self.attn_type = parsed_config['attn_type']
            self.fusion_type = parsed_config['fusion_type']
            self.use_aux_loss = parsed_config['use_aux_loss']
        else:
            # Default values (should not happen in normal usage)
            self.checkpoint_dir = None
            self.model_type = 'triple'
            self.attn_type = 'dual'
            self.fusion_type = 'direct'
            self.use_aux_loss = False
        
        # Override with args if provided
        if args is not None:
            self.img_size = getattr(args, 'img_size', 352)
            self.batch_size = getattr(args, 'batch_size', 1)
            self.num_workers = getattr(args, 'num_workers', 4)
            self.device = getattr(args, 'device', 'cuda')
            self.output_dir = getattr(args, 'output_dir', './results')
            self.verbose = getattr(args, 'verbose', True)
            self.use_improved = getattr(args, 'use_improved', True)
            
            # Override parsed values if explicitly provided
            if hasattr(args, 'attn_type') and args.attn_type is not None:
                self.attn_type = args.attn_type
            if hasattr(args, 'fusion_type') and args.fusion_type is not None:
                self.fusion_type = args.fusion_type
            if hasattr(args, 'use_aux_loss') and args.use_aux_loss is not None:
                self.use_aux_loss = args.use_aux_loss
        else:
            # Default values
            self.img_size = kwargs.get('img_size', 352)
            self.batch_size = kwargs.get('batch_size', 1)
            self.num_workers = kwargs.get('num_workers', 4)
            self.device = kwargs.get('device', 'cuda')
            self.output_dir = kwargs.get('output_dir', './results')
            self.verbose = kwargs.get('verbose', True)
            self.use_improved = kwargs.get('use_improved', True)
        
        # For dual/triple models, use_improved can be overridden
        if self.model_type == 'single':
            self.use_improved = False  # Single stream doesn't use improved fusion
        
        # Test dataset paths
        self.test_datasets = {
            'CAMO': {
                'img': './data/Test/CAMO/Imgs',
                'cb': './data/Test/CAMO/Imgs_CB',
                'sl': './data/Test/CAMO/Imgs_SL',
                'gt': './data/Test/CAMO/GT'
            },
            'CHAMELEON': {
                'img': './data/Test/CHAMELEON/Imgs',
                'cb': './data/Test/CHAMELEON/Imgs_CB',
                'sl': './data/Test/CHAMELEON/Imgs_SL',
                'gt': './data/Test/CHAMELEON/GT'
            },
            'COD10K': {
                'img': './data/Test/COD10K/Imgs',
                'cb': './data/Test/COD10K/Imgs_CB',
                'sl': './data/Test/COD10K/Imgs_SL',
                'gt': './data/Test/COD10K/GT'
            },
            'NC4K': {
                'img': './data/Test/NC4K/Imgs',
                'cb': './data/Test/NC4K/Imgs_CB',
                'sl': './data/Test/NC4K/Imgs_SL',
                'gt': './data/Test/NC4K/GT'
            }
        }
        
        # Metrics to evaluate
        self.metrics_names = ['mae', 'sm', 'wfm', 'adp_fm', 'adp_em', 'max_fm']
        
    @property
    def checkpoint_path(self):
        """Get checkpoint file path"""
        return os.path.join('./checkpoints', self.checkpoint_dir, 'best_model.pth')
    
    @property
    def results_dir(self):
        """Get results directory"""
        return os.path.join(self.output_dir, self.checkpoint_dir)
    
    @property
    def results_path(self):
        """Get results file path"""
        return os.path.join(self.results_dir, 'test_results.txt')
    
    def print_config(self):
        """Print configuration"""
        print(f"\n{'='*60}")
        print(f"TEST CONFIGURATION")
        print(f"{'='*60}")
        print(f"Checkpoint Directory: {self.checkpoint_dir}")
        print(f"Checkpoint Path: {self.checkpoint_path}")
        print(f"Model Type: {self.model_type}")
        if self.model_type in ['single', 'dual']:
            print(f"Attention Type: {self.attn_type}")
        if self.model_type == 'triple':
            print(f"Fusion Type: {self.fusion_type}")
        print(f"Use Aux Loss (training): {self.use_aux_loss}")
        if self.model_type in ['dual', 'triple']:
            print(f"Use Improved Fusion: {self.use_improved}")
        print(f"Image Size: {self.img_size}")
        print(f"Batch Size: {self.batch_size}")
        print(f"Num Workers: {self.num_workers}")
        print(f"Device: {self.device}")
        print(f"Output Directory: {self.results_dir}")
        print(f"Verbose: {self.verbose}")
        print(f"{'='*60}\n")
    
    def check_checkpoint_exists(self):
        """Check if checkpoint exists"""
        if not os.path.exists(self.checkpoint_path):
            raise FileNotFoundError(
                f"Checkpoint not found: {self.checkpoint_path}\n"
                f"Please ensure the directory './checkpoints/{self.checkpoint_dir}/' exists "
                f"and contains 'best_model.pth'"
            )
        if self.verbose:
            print(f"✓ Found checkpoint: {self.checkpoint_path}")
        return True
    
    def check_dataset_exists(self, dataset_name, dataset_config):
        """Check if dataset directories exist"""
        missing_dirs = []
        if self.model_type == 'single':
            required_keys = ['img', 'gt']
        elif self.model_type == 'dual':
            required_keys = ['img', 'cb', 'gt']
        else:  # triple
            required_keys = ['img', 'cb', 'sl', 'gt']
        
        for key in required_keys:
            if not os.path.exists(dataset_config[key]):
                missing_dirs.append(f"{key}: {dataset_config[key]}")
        
        if missing_dirs:
            if self.verbose:
                print(f"Warning: {dataset_name} missing directories: {', '.join(missing_dirs)}")
            return False
        return True


class ModelTester:
    def __init__(self, config):
        """
        Initialize tester
        
        Args:
            config: TestConfig object
        """
        self.config = config
        self.results = {}
        
    def load_model(self):
        """Load model from checkpoint"""
        if self.config.verbose:
            print("\nLoading model...")
        
        # Create model based on type
        if self.config.model_type == 'single':
            if self.config.verbose:
                print(f"  Model: Single-Stream with attention type '{self.config.attn_type}'")
                if self.config.use_aux_loss:
                    print(f"  Note: Model was trained with auxiliary loss (ignored during testing)")
            model = AttentionSingleStreamCOD(
                pretrained=False,
                attn_type=self.config.attn_type,
                input_size=self.config.img_size
            ).to(self.config.device)
            
        elif self.config.model_type == 'dual':
            if self.config.verbose:
                print(f"  Model: Dual-Stream with attention type '{self.config.attn_type}'")
                print(f"  Use Improved Fusion: {self.config.use_improved}")
                if self.config.use_aux_loss:
                    print(f"  Note: Model was trained with auxiliary loss (ignored during testing)")
            model = AttentionDualStreamCOD(
                pretrained=False,
                attn_type=self.config.attn_type,
                input_size=self.config.img_size,
                share_backbone=True,
                use_improved=self.config.use_improved
            ).to(self.config.device)
            
        else:  # triple
            if self.config.verbose:
                print(f"  Model: Triple-Stream with fusion type '{self.config.fusion_type}'")
                print(f"  Use Improved Fusion: {self.config.use_improved}")
                if self.config.use_aux_loss:
                    print(f"  Note: Model was trained with auxiliary loss (ignored during testing)")
            model = AttentionTripleStreamCOD(
                pretrained=False,
                input_size=self.config.img_size,
                share_backbone=True,
                use_improved=self.config.use_improved,
                use_aux_loss=False,  # Always False for testing
                fusion_type=self.config.fusion_type,
                supplement_strength=0.2
            ).to(self.config.device)
        
        # Load checkpoint
        checkpoint = torch.load(self.config.checkpoint_path, map_location=self.config.device)
        
        # Handle different checkpoint formats
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        else:
            state_dict = checkpoint
        
        # Load weights
        model.load_state_dict(state_dict, strict=True)
        model.eval()
        
        if self.config.verbose:
            print(f"✓ Model loaded successfully")
            
            # Print model configuration from checkpoint if available
            if 'fusion_type' in checkpoint:
                print(f"  Fusion Type (from checkpoint): {checkpoint.get('fusion_type', 'unknown')}")
            if 'use_improved_fusion' in checkpoint:
                print(f"  Use Improved Fusion (from checkpoint): {checkpoint.get('use_improved_fusion', 'unknown')}")
            if 'attn_type' in checkpoint:
                print(f"  Attention Type (from checkpoint): {checkpoint.get('attn_type', 'unknown')}")
            if 'use_aux_loss' in checkpoint:
                print(f"  Use Aux Loss (from checkpoint): {checkpoint.get('use_aux_loss', False)}")
        
        return model
    
    def create_validation_config(self):
        """Create validation config object"""
        class ValConfig:
            pass
        
        val_config = ValConfig()
        val_config.device = self.config.device
        val_config.use_amp = False
        
        return val_config
    
    def test_dataset_single(self, model, dataset_name, dataset_config):
        """Test on a single dataset for single-stream model"""
        if self.config.verbose:
            print(f"\n{'='*50}")
            print(f"Testing on {dataset_name}...")
            print(f"{'='*50}")
        
        try:
            # Create dataset
            test_dataset = SingleStreamDataset(
                img_dir=dataset_config['img'],
                gt_dir=dataset_config['gt'],
                img_size=self.config.img_size,
                transform=None,
                is_train=False
            )
            
            if len(test_dataset) == 0:
                print(f"Warning: No valid samples found in {dataset_name}")
                return None
            
            if self.config.verbose:
                print(f"  Samples: {len(test_dataset)}")
            
            # Create dataloader
            test_loader = DataLoader(
                test_dataset,
                batch_size=self.config.batch_size,
                shuffle=False,
                num_workers=self.config.num_workers,
                pin_memory=True,
                collate_fn=collate_fn_single,
                persistent_workers=False
            )
            
            # Create validation config
            val_config = self.create_validation_config()
            
            # Run validation
            metrics = validate_batch_single(model, test_loader, val_config)
            
            # Clear cache
            test_dataset.clear_cache()
            
            if self.config.verbose:
                print(f"\n  Results for {dataset_name}:")
                for k, v in metrics.items():
                    print(f"    {k.upper()}: {v:.4f}")
            
            return metrics
            
        except Exception as e:
            print(f"Error testing on {dataset_name}: {e}")
            return None
    
    def test_dataset_dual(self, model, dataset_name, dataset_config):
        """Test on a single dataset for dual-stream model"""
        if self.config.verbose:
            print(f"\n{'='*50}")
            print(f"Testing on {dataset_name}...")
            print(f"{'='*50}")
        
        try:
            # Create dataset
            test_dataset = DualStreamDataset(
                img_dir=dataset_config['img'],
                cb_dir=dataset_config['cb'],
                gt_dir=dataset_config['gt'],
                img_size=self.config.img_size,
                transform=None,
                is_train=False
            )
            
            if len(test_dataset) == 0:
                print(f"Warning: No valid samples found in {dataset_name}")
                return None
            
            if self.config.verbose:
                print(f"  Samples: {len(test_dataset)}")
            
            # Create dataloader
            test_loader = DataLoader(
                test_dataset,
                batch_size=self.config.batch_size,
                shuffle=False,
                num_workers=self.config.num_workers,
                pin_memory=True,
                collate_fn=collate_fn_dual,
                persistent_workers=False
            )
            
            # Create validation config
            val_config = self.create_validation_config()
            
            # Run validation
            metrics = validate_batch_dual(model, test_loader, val_config)
            
            # Clear cache
            test_dataset.clear_cache()
            
            if self.config.verbose:
                print(f"\n  Results for {dataset_name}:")
                for k, v in metrics.items():
                    print(f"    {k.upper()}: {v:.4f}")
            
            return metrics
            
        except Exception as e:
            print(f"Error testing on {dataset_name}: {e}")
            return None
    
    def test_dataset_triple(self, model, dataset_name, dataset_config):
        """Test on a single dataset for triple-stream model"""
        if self.config.verbose:
            print(f"\n{'='*50}")
            print(f"Testing on {dataset_name}...")
            print(f"{'='*50}")
        
        try:
            # Create dataset
            test_dataset = TripleStreamDataset(
                img_dir=dataset_config['img'],
                cb_dir=dataset_config['cb'],
                sl_dir=dataset_config['sl'],
                gt_dir=dataset_config['gt'],
                img_size=self.config.img_size,
                transform=None,
                is_train=False
            )
            
            if len(test_dataset) == 0:
                print(f"Warning: No valid samples found in {dataset_name}")
                return None
            
            if self.config.verbose:
                print(f"  Samples: {len(test_dataset)}")
            
            # Create dataloader
            test_loader = DataLoader(
                test_dataset,
                batch_size=self.config.batch_size,
                shuffle=False,
                num_workers=self.config.num_workers,
                pin_memory=True,
                collate_fn=collate_fn_triple,
                persistent_workers=False
            )
            
            # Create validation config
            val_config = self.create_validation_config()
            
            # Run validation
            metrics = validate_batch_triple(model, test_loader, val_config)
            
            # Clear cache
            test_dataset.clear_cache()
            
            if self.config.verbose:
                print(f"\n  Results for {dataset_name}:")
                for k, v in metrics.items():
                    print(f"    {k.upper()}: {v:.4f}")
            
            return metrics
            
        except Exception as e:
            print(f"Error testing on {dataset_name}: {e}")
            return None
    
    def run_all_tests(self):
        """Run tests on all datasets"""
        # Print header
        print(f"\n{'#'*60}")
        print(f"{self.config.model_type.upper()}-STREAM COD MODEL TESTING")
        print(f"{'#'*60}")
        print(f"Checkpoint: {self.config.checkpoint_path}")
        print(f"Results will be saved to: {self.config.results_dir}")
        print(f"Device: {self.config.device}")
        print(f"Image Size: {self.config.img_size}")
        print(f"Batch Size: {self.config.batch_size}")
        
        # Check checkpoint
        self.config.check_checkpoint_exists()
        
        # Load model
        model = self.load_model()
        
        # Test on each dataset
        all_metrics = {}
        valid_metrics = []
        
        for dataset_name, dataset_config in self.config.test_datasets.items():
            # Check if dataset exists
            if not self.config.check_dataset_exists(dataset_name, dataset_config):
                continue
            
            # Run test based on model type
            if self.config.model_type == 'single':
                metrics = self.test_dataset_single(model, dataset_name, dataset_config)
            elif self.config.model_type == 'dual':
                metrics = self.test_dataset_dual(model, dataset_name, dataset_config)
            else:  # triple
                metrics = self.test_dataset_triple(model, dataset_name, dataset_config)
            
            if metrics is not None:
                all_metrics[dataset_name] = metrics
                valid_metrics.append(metrics)
        
        # Calculate and display average metrics
        if valid_metrics:
            self._display_results(all_metrics, valid_metrics)
            self._save_results(all_metrics, valid_metrics)
        else:
            print("\n⚠ No valid test results obtained!")
        
        return all_metrics
    
    def _display_results(self, all_metrics, valid_metrics):
        """Display test results"""
        # Calculate averages
        avg_metrics = {}
        for key in self.config.metrics_names:
            avg_metrics[key] = np.mean([m[key] for m in valid_metrics])
        
        print(f"\n{'='*60}")
        print(f"SUMMARY - AVERAGE OVER {len(valid_metrics)} DATASETS")
        print(f"{'='*60}")
        for key, value in avg_metrics.items():
            print(f"  {key.upper()}: {value:.4f}")
        
        # Print detailed table
        print(f"\n{'='*90}")
        print(f"{'Dataset':<12} {'MAE':<10} {'Sm':<10} {'wFm':<10} {'adpFm':<10} {'adpEm':<10} {'maxFm':<10}")
        print(f"{'-'*90}")
        
        for dataset_name, metrics in all_metrics.items():
            print(f"{dataset_name:<12} "
                  f"{metrics['mae']:<10.4f} "
                  f"{metrics['sm']:<10.4f} "
                  f"{metrics['wfm']:<10.4f} "
                  f"{metrics['adp_fm']:<10.4f} "
                  f"{metrics['adp_em']:<10.4f} "
                  f"{metrics['max_fm']:<10.4f}")
        
        print(f"{'-'*90}")
        print(f"{'AVERAGE':<12} "
              f"{avg_metrics['mae']:<10.4f} "
              f"{avg_metrics['sm']:<10.4f} "
              f"{avg_metrics['wfm']:<10.4f} "
              f"{avg_metrics['adp_fm']:<10.4f} "
              f"{avg_metrics['adp_em']:<10.4f} "
              f"{avg_metrics['max_fm']:<10.4f}")
        print(f"{'='*90}")
    
    def _save_results(self, all_metrics, valid_metrics):
        """Save test results to file"""
        # Create results directory
        os.makedirs(self.config.results_dir, exist_ok=True)
        
        # Calculate averages
        avg_metrics = {}
        for key in self.config.metrics_names:
            avg_metrics[key] = np.mean([m[key] for m in valid_metrics])
        
        with open(self.config.results_path, 'w') as f:
            f.write(f"{'='*90}\n")
            f.write(f"{self.config.model_type.upper()}-STREAM COD TEST RESULTS\n")
            f.write(f"Checkpoint: {self.config.checkpoint_path}\n")
            f.write(f"Model Configuration:\n")
            if self.config.model_type == 'single':
                f.write(f"  Attention Type: {self.config.attn_type}\n")
            elif self.config.model_type == 'dual':
                f.write(f"  Attention Type: {self.config.attn_type}\n")
                f.write(f"  Use Improved Fusion: {self.config.use_improved}\n")
            else:  # triple
                f.write(f"  Fusion Type: {self.config.fusion_type}\n")
                f.write(f"  Use Improved Fusion: {self.config.use_improved}\n")
            f.write(f"  Use Aux Loss (training): {self.config.use_aux_loss}\n")
            f.write(f"{'='*90}\n\n")
            
            f.write(f"{'Dataset':<12} {'MAE':<10} {'Sm':<10} {'wFm':<10} {'adpFm':<10} {'adpEm':<10} {'maxFm':<10}\n")
            f.write(f"{'-'*90}\n")
            
            for dataset_name, metrics in all_metrics.items():
                f.write(f"{dataset_name:<12} "
                       f"{metrics['mae']:<10.4f} "
                       f"{metrics['sm']:<10.4f} "
                       f"{metrics['wfm']:<10.4f} "
                       f"{metrics['adp_fm']:<10.4f} "
                       f"{metrics['adp_em']:<10.4f} "
                       f"{metrics['max_fm']:<10.4f}\n")
            
            f.write(f"{'-'*90}\n")
            f.write(f"{'AVERAGE':<12} "
                   f"{avg_metrics['mae']:<10.4f} "
                   f"{avg_metrics['sm']:<10.4f} "
                   f"{avg_metrics['wfm']:<10.4f} "
                   f"{avg_metrics['adp_fm']:<10.4f} "
                   f"{avg_metrics['adp_em']:<10.4f} "
                   f"{avg_metrics['max_fm']:<10.4f}\n")
            f.write(f"{'='*90}\n")
        
        if self.config.verbose:
            print(f"\n✓ Results saved to: {self.config.results_path}")


def parse_arguments():
    parser = argparse.ArgumentParser(description='Test COD Model')
    parser.add_argument('-p', '--checkpoint_dir', type=str, required=True,
                        help='Checkpoint directory name (e.g., single_dual, triple_direct_aux)')
    
    # Optional overrides (usually not needed as they are parsed from checkpoint_dir)
    parser.add_argument('--model_type', type=str, default=None,
                        choices=['single', 'dual', 'triple'],
                        help='Override model type (auto-detected from checkpoint_dir)')
    parser.add_argument('--attn_type', type=str, default=None,
                        choices=['spatial', 'channel', 'dual', 'none'],
                        help='Override attention type for single/dual models (auto-detected)')
    parser.add_argument('--fusion_type', type=str, default=None,
                        choices=['direct', 'hierarchical', 'gated', 'residual'],
                        help='Override fusion type for triple model (auto-detected)')
    parser.add_argument('--use_aux_loss', action='store_true', default=None,
                        help='Override use_aux_loss flag (auto-detected)')
    parser.add_argument('--no_aux_loss', action='store_true', default=False,
                        help='Force use_aux_loss=False')
    
    # Model parameters
    parser.add_argument('--use_improved', action='store_true', default=True,
                        help='Use improved fusion for dual/triple models (default: True)')
    parser.add_argument('--no_improved', action='store_true', default=False,
                        help='Disable improved fusion')
    
    # Testing parameters
    parser.add_argument('--img_size', type=int, default=352,
                        help='Input image size (default: 352)')
    parser.add_argument('--batch_size', type=int, default=1,
                        help='Batch size for testing (default: 1)')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of data loading workers (default: 4)')
    parser.add_argument('--device', type=str, default='cuda',
                        choices=['cuda', 'cpu'],
                        help='Device to use (default: cuda)')
    parser.add_argument('--output_dir', type=str, default='./results',
                        help='Output directory for results (default: ./results)')
    parser.add_argument('--verbose', action='store_true', default=True,
                        help='Print verbose output (default: True)')
    parser.add_argument('--quiet', action='store_true', default=False,
                        help='Suppress verbose output')
    
    return parser.parse_args()


def main():
    args = parse_arguments()
    
    # Handle quiet mode
    if args.quiet:
        args.verbose = False
    
    # Handle use_improved flag
    if args.no_improved:
        args.use_improved = False
    
    # Handle use_aux_loss override
    if args.no_aux_loss:
        args.use_aux_loss = False
    elif args.use_aux_loss:
        args.use_aux_loss = True
    # Otherwise, use_aux_loss will be parsed from checkpoint_dir
    
    # Create configuration
    config = TestConfig(args=args)
    
    # Print configuration
    config.print_config()
    
    # Check CUDA availability
    if config.device == 'cuda' and not torch.cuda.is_available():
        print("Warning: CUDA not available, using CPU instead")
        config.device = 'cpu'
    
    # Set random seed
    set_seed(42)
    
    # Run tests
    tester = ModelTester(config)
    
    try:
        results = tester.run_all_tests()
    except Exception as e:
        print(f"\nError during testing: {e}")
        raise
    finally:
        cleanup()


if __name__ == '__main__':
    main()