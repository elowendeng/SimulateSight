# config/dual_stream.py

import os
os.environ['NO_ALBUMENTATIONS_UPDATE'] = '1'
import torch
from pathlib import Path

# Get the directory where the current file is located
CONFIG_DIR = Path(__file__).resolve().parent
# Get the parent directory of this directory
PROJECT_ROOT = CONFIG_DIR.parent


class Config_Dual:
    def __init__(self, args=None):
        # parameters of the attention mechanism
        self.attn_type = 'dual'  # dual, spatial, channel, multi
        # weight sharing configuration
        self.share_backbone = True
        # model improvement options
        self.use_improved_fusion = True
        self.use_residual = True
        self.learnable_weights = True

        # path configuration
        self.train_img_dir = str(PROJECT_ROOT / "data/Train/Imgs")
        self.train_cb_dir = str(PROJECT_ROOT / "data/Train/Imgs_CB")
        self.train_gt_dir = str(PROJECT_ROOT / "data/Train/GT")

        self.val_img_dir = str(PROJECT_ROOT / "data/Val/Imgs")
        self.val_cb_dir = str(PROJECT_ROOT / "data/Val/Imgs_CB")
        self.val_gt_dir = str(PROJECT_ROOT / "data/Val/GT")

        # training parameters
        self.img_size = 352
        self.batch_size = 16
        self.num_epochs = 160
        self.learning_rate = 1e-4
        self.weight_decay = 0.01
        self.num_workers = 8
        self.grad_clip_norm = 2.0
        self.warmup_epochs = 5
        self.warmup_start_lr = 2e-5

        # auxiliary loss configuration
        self.use_aux_loss = False
        self.aux_loss_weight = 0.3

        # early stop parameter
        self.patience = 10
        self.rebound_threshold = 100

        # storing data using models
        self.save_dir = None
        self.log_interval = 50
        self.save_interval = 10
        self.save_latest = True

        # Learning rate
        self.use_cosine_annealing = True
        # mixed-precision training
        self.use_amp = True

        # data augmentation parameters
        self.use_augmentation = True
        self.hflip_prob = 0.5
        self.vflip_prob = 0.5
        self.rotation_degree = 10
        self.brightness_limit = 0.1
        self.contrast_limit = 0.1
        self.saturation_limit = 0.1
        self.hue_limit = 0.05

        # loss function parameters
        self.loss_alpha = 0.7
        self.loss_beta = 0.3

        # visualization parameters
        self.plot_interval = 1
        self.plot_metrics = ['mae', 'sm', 'wfm', 'adp_fm']

        # complementary verification parameters
        self.enable_complementarity_check = False
        self.complementarity_check_interval = 5

        # device
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        if args:
            self._update_from_args(args)

        if self.save_dir is None:
            self.save_dir = self._get_default_save_dir()
        
        # breakpoint continuation training configuration
        self.resume_training = True
        self.resume_checkpoint = str(Path(self.save_dir) / "checkpoint_latest.pth")

        os.makedirs(self.save_dir, exist_ok=True)
        self._validate_config()
    
    def _get_default_save_dir(self):
        if self.use_aux_loss:
            return str(PROJECT_ROOT / f"checkpoints/dual_{self.attn_type}_aux")
        else:
            return str(PROJECT_ROOT / f"checkpoints/dual_{self.attn_type}")

    def _validate_config(self):
        """verify the validity of the configuration parameters"""
        assert self.img_size >= 32, f"img_size must be >= 32, got {self.img_size}"
        assert self.batch_size >= 1, f"batch_size must be >= 1, got {self.batch_size}"
        assert self.learning_rate > 0, f"learning_rate must be positive, got {self.learning_rate}"
        assert self.grad_clip_norm > 0, f"grad_clip_norm must be positive, got {self.grad_clip_norm}"

        if self.warmup_epochs > self.num_epochs:
            print(
                f"Warning: warmup_epochs ({self.warmup_epochs}) > num_epochs ({self.num_epochs}), reducing to {self.num_epochs // 2}")
            self.warmup_epochs = self.num_epochs // 2

        valid_attn_types = ['spatial', 'channel', 'dual', 'multi', 'none']
        if self.attn_type not in valid_attn_types:
            raise ValueError(f"attn_type must be one of {valid_attn_types}, got {self.attn_type}")

    def _update_from_args(self, args):
        for key, value in vars(args).items():
            if value is not None and hasattr(self, key):
                setattr(self, key, value)
                print(f"Config updated: {key} = {value}")