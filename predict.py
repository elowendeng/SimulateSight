"""
Prediction script for Single/Dual/Triple-Stream COD models
Predict saliency maps for images in a given folder
"""

import os
import sys
import argparse
import warnings
from datetime import datetime

import cv2
import torch
import numpy as np
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

warnings.filterwarnings('ignore')

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from models import AttentionTripleStreamCOD, AttentionDualStreamCOD, AttentionSingleStreamCOD
from utils import set_seed, cleanup


class PredictionDataset(Dataset):
    """Dataset for prediction"""
    
    def __init__(self, img_dir, img_size=352, cb_dir=None, sl_dir=None):
        """
        Args:
            img_dir: Directory containing images to predict
            img_size: Target image size
            cb_dir: Directory containing CB images (for dual/triple stream)
            sl_dir: Directory containing SL images (for triple stream)
        """
        self.img_dir = img_dir
        self.img_size = img_size
        self.cb_dir = cb_dir
        self.sl_dir = sl_dir
        
        # Get all image files
        valid_ext = ['.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG', '.bmp', '.BMP']
        self.img_paths = []
        self.names = []
        
        for f in os.listdir(img_dir):
            name, ext = os.path.splitext(f)
            if ext.lower() in valid_ext:
                self.img_paths.append(os.path.join(img_dir, f))
                self.names.append(name)
        
        if len(self.img_paths) == 0:
            raise ValueError(f"No valid images found in {img_dir}")
        
        print(f"Found {len(self.img_paths)} images for prediction")
        
    def __len__(self):
        return len(self.img_paths)
    
    def __getitem__(self, idx):
        # Load image
        img_path = self.img_paths[idx]
        img = cv2.imread(img_path)
        if img is None:
            raise ValueError(f"Failed to load image: {img_path}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Store original size
        orig_h, orig_w = img.shape[:2]
        
        # Resize
        img_resized = cv2.resize(img, (self.img_size, self.img_size))
        img_resized = img_resized.astype(np.float32) / 255.0
        img_tensor = torch.from_numpy(img_resized).permute(2, 0, 1).float()
        
        result = {
            'img': img_tensor,
            'name': self.names[idx],
            'orig_size': (orig_h, orig_w),
            'original_img': img
        }
        
        # Load CB image if provided
        if self.cb_dir is not None:
            cb_path = os.path.join(self.cb_dir, f"{self.names[idx]}.png")
            if not os.path.exists(cb_path):
                # Try other extensions
                for ext in ['.jpg', '.jpeg', '.JPG', '.JPEG']:
                    alt_path = os.path.join(self.cb_dir, f"{self.names[idx]}{ext}")
                    if os.path.exists(alt_path):
                        cb_path = alt_path
                        break
            
            if os.path.exists(cb_path):
                cb = cv2.imread(cb_path)
                if cb is not None:
                    cb = cv2.cvtColor(cb, cv2.COLOR_BGR2RGB)
                    cb_resized = cv2.resize(cb, (self.img_size, self.img_size))
                    cb_resized = cb_resized.astype(np.float32) / 255.0
                    cb_tensor = torch.from_numpy(cb_resized).permute(2, 0, 1).float()
                    result['cb'] = cb_tensor
            else:
                print(f"Warning: CB image not found for {self.names[idx]}, using zeros")
                result['cb'] = torch.zeros_like(result['img'])
        
        # Load SL image if provided
        if self.sl_dir is not None:
            sl_path = os.path.join(self.sl_dir, f"{self.names[idx]}.png")
            if not os.path.exists(sl_path):
                for ext in ['.jpg', '.jpeg', '.JPG', '.JPEG']:
                    alt_path = os.path.join(self.sl_dir, f"{self.names[idx]}{ext}")
                    if os.path.exists(alt_path):
                        sl_path = alt_path
                        break
            
            if os.path.exists(sl_path):
                sl = cv2.imread(sl_path)
                if sl is not None:
                    sl = cv2.cvtColor(sl, cv2.COLOR_BGR2RGB)
                    sl_resized = cv2.resize(sl, (self.img_size, self.img_size))
                    sl_resized = sl_resized.astype(np.float32) / 255.0
                    sl_tensor = torch.from_numpy(sl_resized).permute(2, 0, 1).float()
                    result['sl'] = sl_tensor
            else:
                print(f"Warning: SL image not found for {self.names[idx]}, using zeros")
                result['sl'] = torch.zeros_like(result['img'])
        
        return result


def parse_checkpoint_name(checkpoint_dir):
    """Parse checkpoint directory name to extract model configuration"""
    parts = checkpoint_dir.split('_')
    
    if len(parts) < 2:
        raise ValueError(f"Invalid checkpoint directory name: {checkpoint_dir}")
    
    model_type = parts[0].lower()
    
    if model_type not in ['single', 'dual', 'triple']:
        raise ValueError(f"Invalid model type: {model_type}")
    
    if model_type in ['single', 'dual']:
        attn_type = parts[1].lower() if len(parts) >= 2 else 'dual'
        if attn_type not in ['spatial', 'channel', 'dual', 'none']:
            attn_type = 'dual'
        return {
            'model_type': model_type,
            'attn_type': attn_type,
            'fusion_type': None
        }
    else:
        fusion_type = parts[1].lower() if len(parts) >= 2 else 'direct'
        if fusion_type not in ['direct', 'hierarchical', 'gated', 'residual']:
            fusion_type = 'direct'
        return {
            'model_type': model_type,
            'attn_type': None,
            'fusion_type': fusion_type
        }


class Predictor:
    def __init__(self, checkpoint_dir, device='cuda', img_size=352, use_improved=True):
        """
        Initialize predictor
        
        Args:
            checkpoint_dir: Checkpoint directory name (e.g., 'single_dual', 'triple_direct')
            device: Device to use ('cuda' or 'cpu')
            img_size: Input image size
            use_improved: Whether to use improved fusion (for dual/triple)
        """
        self.checkpoint_dir = checkpoint_dir
        self.checkpoint_path = os.path.join('./checkpoints', checkpoint_dir, 'best_model.pth')
        self.device = device
        self.img_size = img_size
        self.use_improved = use_improved
        
        # Parse model configuration
        self.config = parse_checkpoint_name(checkpoint_dir)
        self.model_type = self.config['model_type']
        self.attn_type = self.config['attn_type']
        self.fusion_type = self.config['fusion_type']
        
        # Load model
        self.model = self._load_model()
        
    def _load_model(self):
        """Load model from checkpoint"""
        print(f"\n{'='*50}")
        print(f"Loading Model")
        print(f"{'='*50}")
        print(f"Checkpoint: {self.checkpoint_path}")
        print(f"Model Type: {self.model_type}")
        
        if self.model_type == 'single':
            print(f"Attention Type: {self.attn_type}")
            model = AttentionSingleStreamCOD(
                pretrained=False,
                attn_type=self.attn_type,
                input_size=self.img_size
            ).to(self.device)
        elif self.model_type == 'dual':
            print(f"Attention Type: {self.attn_type}")
            print(f"Use Improved Fusion: {self.use_improved}")
            model = AttentionDualStreamCOD(
                pretrained=False,
                attn_type=self.attn_type,
                input_size=self.img_size,
                share_backbone=True,
                use_improved=self.use_improved
            ).to(self.device)
        else:  # triple
            print(f"Fusion Type: {self.fusion_type}")
            print(f"Use Improved Fusion: {self.use_improved}")
            model = AttentionTripleStreamCOD(
                pretrained=False,
                input_size=self.img_size,
                share_backbone=True,
                use_improved=self.use_improved,
                use_aux_loss=False,
                fusion_type=self.fusion_type,
                supplement_strength=0.2
            ).to(self.device)
        
        # Load checkpoint
        checkpoint = torch.load(self.checkpoint_path, map_location=self.device)
        
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        else:
            state_dict = checkpoint
        
        model.load_state_dict(state_dict, strict=True)
        model.eval()
        
        print(f"✓ Model loaded successfully")
        print(f"{'='*50}\n")
        
        return model
    
    def predict_folder(self, img_dir, output_dir, cb_dir=None, sl_dir=None, save_original_size=True):
        """
        Predict all images in a folder
        
        Args:
            img_dir: Directory containing input images
            output_dir: Directory to save prediction results
            cb_dir: Directory containing CB images (for dual/triple stream)
            sl_dir: Directory containing SL images (for triple stream)
            save_original_size: Whether to resize predictions to original image size
        """
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
    
        # Create dataset
        dataset = PredictionDataset(
            img_dir=img_dir,
            img_size=self.img_size,
            cb_dir=cb_dir if self.model_type in ['dual', 'triple'] else None,
            sl_dir=sl_dir if self.model_type == 'triple' else None
        )
    
        # Create dataloader
        dataloader = DataLoader(
            dataset,
            batch_size=1,
            shuffle=False,
            num_workers=2,
            pin_memory=True
        )
    
        print(f"\n{'='*50}")
        print(f"Predicting {len(dataset)} images")
        print(f"Output directory: {output_dir}")
        print(f"{'='*50}\n")
        
        # Predict
        with torch.no_grad():
            for batch in tqdm(dataloader, desc='Predicting'):
                # Get data
                img = batch['img'].to(self.device)
                name = batch['name'][0]
            
                # Get original size from original_img
                orig_img = batch['original_img'][0]
                if torch.is_tensor(orig_img):
                    orig_img = orig_img.numpy()
                orig_h, orig_w = orig_img.shape[:2]
            
                # Forward pass
                if self.model_type == 'single':
                    pred = self.model(img)
                elif self.model_type == 'dual':
                    cb = batch['cb'].to(self.device)
                    pred = self.model(img, cb)
                else:  # triple
                    cb = batch['cb'].to(self.device)
                    sl = batch['sl'].to(self.device)
                    pred = self.model(img, cb, sl)
            
                # Apply sigmoid
                pred = torch.sigmoid(pred).cpu().numpy()
                pred = pred[0, 0] if pred.ndim > 2 else pred[0]
            
                # Resize to original size if needed
                if save_original_size and (pred.shape[0] != orig_h or pred.shape[1] != orig_w):
                    pred = cv2.resize(pred, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)
            
                # Save prediction
                pred_uint8 = (pred * 255).astype(np.uint8)
                output_path = os.path.join(output_dir, f"{name}.png")
                cv2.imwrite(output_path, pred_uint8)
    
        print(f"\n✓ All predictions saved to: {output_dir}")
        
    def predict_single(self, img_path, output_path=None, cb_path=None, sl_path=None):
        """
        Predict a single image
        
        Args:
            img_path: Path to input image
            output_path: Path to save prediction (if None, will not save)
            cb_path: Path to CB image (for dual/triple stream)
            sl_path: Path to SL image (for triple stream)
        
        Returns:
            Prediction map as numpy array (float32, range 0-1)
        """
        # Load image
        img = cv2.imread(img_path)
        if img is None:
            raise ValueError(f"Failed to load image: {img_path}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        orig_h, orig_w = img.shape[:2]
        
        # Resize
        img_resized = cv2.resize(img, (self.img_size, self.img_size))
        img_tensor = torch.from_numpy(img_resized).permute(2, 0, 1).float() / 255.0
        img_tensor = img_tensor.unsqueeze(0).to(self.device)
        
        # Prepare inputs based on model type
        with torch.no_grad():
            if self.model_type == 'single':
                pred = self.model(img_tensor)
            elif self.model_type == 'dual':
                if cb_path is None:
                    raise ValueError("CB image required for dual-stream model")
                cb = cv2.imread(cb_path)
                cb = cv2.cvtColor(cb, cv2.COLOR_BGR2RGB)
                cb_resized = cv2.resize(cb, (self.img_size, self.img_size))
                cb_tensor = torch.from_numpy(cb_resized).permute(2, 0, 1).float() / 255.0
                cb_tensor = cb_tensor.unsqueeze(0).to(self.device)
                pred = self.model(img_tensor, cb_tensor)
            else:  # triple
                if cb_path is None or sl_path is None:
                    raise ValueError("CB and SL images required for triple-stream model")
                cb = cv2.imread(cb_path)
                cb = cv2.cvtColor(cb, cv2.COLOR_BGR2RGB)
                cb_resized = cv2.resize(cb, (self.img_size, self.img_size))
                cb_tensor = torch.from_numpy(cb_resized).permute(2, 0, 1).float() / 255.0
                cb_tensor = cb_tensor.unsqueeze(0).to(self.device)
                
                sl = cv2.imread(sl_path)
                sl = cv2.cvtColor(sl, cv2.COLOR_BGR2RGB)
                sl_resized = cv2.resize(sl, (self.img_size, self.img_size))
                sl_tensor = torch.from_numpy(sl_resized).permute(2, 0, 1).float() / 255.0
                sl_tensor = sl_tensor.unsqueeze(0).to(self.device)
                
                pred = self.model(img_tensor, cb_tensor, sl_tensor)
        
        # Apply sigmoid and convert to numpy
        pred = torch.sigmoid(pred).cpu().numpy()
        pred = pred[0, 0] if pred.ndim > 2 else pred[0]
        
        # Resize to original size
        pred = cv2.resize(pred, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)
        
        # Save if output path provided
        if output_path is not None:
            pred_uint8 = (pred * 255).astype(np.uint8)
            cv2.imwrite(output_path, pred_uint8)
            print(f"Prediction saved to: {output_path}")
        
        return pred


def main():
    parser = argparse.ArgumentParser(description='Predict saliency maps using trained COD model')
    parser.add_argument('-p', '--checkpoint_dir', type=str, required=True,
                        help='Checkpoint directory name (e.g., single_dual, triple_direct)')
    parser.add_argument('-i', '--input_dir', type=str, required=True,
                        help='Directory containing input images to predict')
    parser.add_argument('-o', '--output_dir', type=str, default=None,
                        help='Output directory for predictions (default: ./results/{checkpoint_dir}/{input_dir_name})')
    
    # Optional directories for CB and SL images
    parser.add_argument('--cb_dir', type=str, default=None,
                        help='Directory containing CB images (required for dual/triple stream)')
    parser.add_argument('--sl_dir', type=str, default=None,
                        help='Directory containing SL images (required for triple stream)')
    
    # Model parameters
    parser.add_argument('--img_size', type=int, default=352,
                        help='Input image size (default: 352)')
    parser.add_argument('--use_improved', action='store_true', default=True,
                        help='Use improved fusion for dual/triple models (default: True)')
    parser.add_argument('--no_improved', action='store_true', default=False,
                        help='Disable improved fusion')
    
    # Other parameters
    parser.add_argument('--device', type=str, default='cuda',
                        choices=['cuda', 'cpu'],
                        help='Device to use (default: cuda)')
    parser.add_argument('--no_resize', action='store_true', default=False,
                        help='Do not resize predictions to original size')
    
    args = parser.parse_args()
    
    # Handle use_improved flag
    if args.no_improved:
        args.use_improved = False
    
    # Set default output directory
    if args.output_dir is None:
        # Obtain the name of the input folder
        input_path = os.path.normpath(args.input_dir)
        dir_name = os.path.basename(input_path)
        # If the last level is "Imgs", then take the name of the parent directory.
        if dir_name.lower() == 'imgs':
            dir_name = os.path.basename(os.path.dirname(input_path))
        args.output_dir = os.path.join('./results', args.checkpoint_dir, dir_name)
    
    # Check CUDA availability
    if args.device == 'cuda' and not torch.cuda.is_available():
        print("Warning: CUDA not available, using CPU instead")
        args.device = 'cpu'
    
    # Set random seed
    set_seed(42)
    
    # Print configuration
    print(f"\n{'#'*60}")
    print(f"COD MODEL PREDICTION")
    print(f"{'#'*60}")
    print(f"Checkpoint: {args.checkpoint_dir}")
    print(f"Input Directory: {args.input_dir}")
    print(f"Output Directory: {args.output_dir}")
    if args.cb_dir:
        print(f"CB Directory: {args.cb_dir}")
    if args.sl_dir:
        print(f"SL Directory: {args.sl_dir}")
    print(f"Device: {args.device}")
    print(f"Image Size: {args.img_size}")
    print(f"Save Original Size: {not args.no_resize}")
    print(f"{'#'*60}")
    
    # Create predictor
    predictor = Predictor(
        checkpoint_dir=args.checkpoint_dir,
        device=args.device,
        img_size=args.img_size,
        use_improved=args.use_improved
    )
    
    # Run prediction
    try:
        predictor.predict_folder(
            img_dir=args.input_dir,
            output_dir=args.output_dir,
            cb_dir=args.cb_dir,
            sl_dir=args.sl_dir,
            save_original_size=not args.no_resize
        )
    except Exception as e:
        print(f"\nError during prediction: {e}")
        raise
    finally:
        cleanup()


if __name__ == '__main__':
    main()