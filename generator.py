# generator.py

import sys
import os
import shutil
from pathlib import Path
import logging
import numpy as np
import cv2
import random
import gc
import hashlib
from typing import Optional, Tuple, Dict, List, Union
from datetime import datetime
import threading
import multiprocessing
import argparse
import traceback

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# ==================== Configure the logging system ====================
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s', 
                             datefmt='%Y-%m-%d %H:%M:%S')

console_handler = logging.StreamHandler(sys.stdout)
console_handler.setLevel(logging.INFO)
console_handler.setFormatter(formatter)

root_logger = logging.getLogger()
if root_logger.handlers:
    root_logger.handlers.clear()
root_logger.setLevel(logging.INFO)
root_logger.addHandler(console_handler)

logger = logging.getLogger(__name__)

# ==================== Import other dependencies ====================
try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False
    logger.warning("psutil not installed. Memory monitoring will be disabled.")

try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False
    logger.warning("tqdm not installed. Progress bar will be disabled.")


def parse_arguments():
    parser = argparse.ArgumentParser(description='Data Generator for COD Dataset')
    parser.add_argument('--path', '-p', type=str, required=True,
                        help='Path to the dataset folder (e.g., ./data/Train)')
    parser.add_argument('--output-dir', '-o', type=str, default=None,
                        help='Output directory (default: same as input)')
    parser.add_argument('--sunglass-mode', '-s', type=str, default='random',
                        choices=['brown', 'blue', 'strong_glare', 'vintage', 'random'],
                        help='Sunglass effect mode')
    parser.add_argument('--colorblind-type', '-c', type=str, default='deuteranopia',
                        choices=['protanopia', 'deuteranopia', 'tritanopia', 'random'],
                        help='Colorblind simulation type')
    parser.add_argument('--edge-mode', '-e', type=str, default='soft',
                        choices=['soft', 'sharp', 'binary'],
                        help='Edge detection mode')
    parser.add_argument('--output-format', '-f', type=str, default='png',
                        choices=['png', 'jpg'],
                        help='Output image format')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--colorblind-severity', type=float, default=1.0,
                        help='Colorblind severity (0.0-1.0)')
    parser.add_argument('--no-progress', action='store_true',
                        help='Disable progress bar')
    parser.add_argument('--force-reprocess', action='store_true',
                        help='Force reprocess all images (ignore cache)')
    parser.add_argument('--no-backup', action='store_true',
                        help='Do not backup existing output directory')
    parser.add_argument('--skip-edge', action='store_true',
                        help='Skip generating edge GT (useful for test datasets)')
    parser.add_argument('--skip-sunglass', action='store_true',
                        help='Skip generating sunglass images')
    parser.add_argument('--skip-colorblind', action='store_true',
                        help='Skip generating colorblind images')
    return parser.parse_args()


# ==================== Global Configuration ====================
INPUT_IMAGE_PATH = None
INPUT_GT_PATH = None
OUTPUT_BASE = None

OUTPUT_SUBDIRS = {
    'sunglass': 'Imgs_SL',
    'colorblind': 'Imgs_CB',
    'edge': 'GT_Edge'
}

SKIP_EDGE = False
SKIP_SUNGLASS = False
SKIP_COLORBLIND = False

RANDOM_SEED = 42

SUNGLASS_MODE = 'random'
USE_FIXED_SUNGLASS_PARAMS = False

COLORBLIND_TYPE = 'deuteranopia'
COLORBLIND_SEVERITY = 1.0
USE_FIXED_COLORBLIND_PARAMS = True

EDGE_MODE = 'soft'
EDGE_THRESHOLD_METHOD = 'otsu'
EDGE_PERCENTILE = 90

OUTPUT_FORMAT = "png"
OUTPUT_QUALITY = 100

SHOW_PROGRESS_BAR = True and TQDM_AVAILABLE

MAX_WORKERS = 1
USE_MULTIPROCESSING = False

MAX_RETRIES = 3
RETRY_DELAY = 1

ENABLE_CACHE = False
CACHE_FILE = None

ENABLE_MEMORY_MONITOR = True
MEMORY_THRESHOLD = 90
FORCE_GC_AFTER_BATCH = True

# ==================== dependency-check ====================
try:
    import skimage.feature as sk_feature
    SKIMAGE_AVAILABLE = True
except ImportError:
    SKIMAGE_AVAILABLE = False
    logger.warning("scikit-image not installed. LoG edge detection will be disabled.")


# ==================== Utility Functions ====================

def get_memory_usage() -> float:
    if not PSUTIL_AVAILABLE:
        return 0.0
    try:
        return psutil.virtual_memory().percent
    except:
        return 0.0


def check_memory_threshold() -> bool:
    if not ENABLE_MEMORY_MONITOR:
        return True
    memory_percent = get_memory_usage()
    if memory_percent > MEMORY_THRESHOLD:
        logger.warning(f"Memory usage high: {memory_percent:.1f}% > {MEMORY_THRESHOLD}%")
        return False
    return True


def get_file_based_seed(file_path: Path, base_seed: int = RANDOM_SEED) -> int:
    path_str = file_path.name
    hash_obj = hashlib.md5(f"{base_seed}_{path_str}".encode())
    seed = int(hash_obj.hexdigest()[:8], 16)
    return seed


def ensure_bgr(img: np.ndarray) -> np.ndarray:
    if img is None:
        return None
    if len(img.shape) == 2:
        return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    elif len(img.shape) == 3 and img.shape[2] == 1:
        return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    elif len(img.shape) == 3 and img.shape[2] == 3:
        return img
    elif len(img.shape) == 3 and img.shape[2] == 4:
        return cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)
    else:
        return img


def validate_image(img: np.ndarray, img_path: Path) -> Tuple[bool, str]:
    if img is None:
        return False, "Image is None"
    if img.size == 0:
        return False, "Image has zero size"
    if img.shape[0] == 0 or img.shape[1] == 0:
        return False, f"Invalid image dimensions: {img.shape}"
    if len(img.shape) not in [2, 3]:
        return False, f"Invalid number of dimensions: {len(img.shape)}"
    return True, ""


def get_file_hash(file_path: Path) -> str:
    hash_md5 = hashlib.md5()
    try:
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_md5.update(chunk)
        return hash_md5.hexdigest()
    except Exception as e:
        logger.error(f"Failed to calculate hash: {str(e)}")
        return ""


def backup_existing_outputs(output_dir: Path, no_backup: bool = False):
    if no_backup:
        if output_dir.exists():
            logger.warning(f"Removing existing output directory: {output_dir}")
            shutil.rmtree(output_dir)
        return
    
    if output_dir.exists() and any(output_dir.iterdir()):
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        backup_dir = output_dir.parent / f"{output_dir.name}_backup_{timestamp}"
        logger.warning(f"Output directory not empty, backing up to {backup_dir}")
        shutil.move(str(output_dir), str(backup_dir))
        logger.info(f"Backup completed: {backup_dir}")


def validate_paths(image_path: Path, gt_path: Path) -> bool:
    if not image_path.exists():
        logger.error(f"Image path does not exist: {image_path}")
        return False
    
    if not gt_path.exists():
        logger.error(f"GT path does not exist: {gt_path}")
        return False
    
    image_files = collect_image_files(image_path)
    if not image_files:
        logger.error(f"No image files found in {image_path}")
        return False
    
    has_gt = False
    for img_file in image_files[:10]:
        gt_file = find_matching_gt(img_file, gt_path)
        if gt_file:
            has_gt = True
            break
    
    if not has_gt:
        logger.warning("No matching GT files found in the first 10 images")
    
    return True


# ==================== Color space conversion tool function ====================

def srgb_to_linear(img: np.ndarray) -> np.ndarray:
    mask = img <= 0.04045
    linear = np.zeros_like(img)
    linear[mask] = img[mask] / 12.92
    linear[~mask] = ((img[~mask] + 0.055) / 1.055) ** 2.4
    return linear


def linear_to_srgb(img: np.ndarray) -> np.ndarray:
    mask = img <= 0.0031308
    srgb = np.zeros_like(img)
    srgb[mask] = img[mask] * 12.92
    srgb[~mask] = 1.055 * (img[~mask] ** (1/2.4)) - 0.055
    return np.clip(srgb, 0, 1)


def bgr_to_linear_rgb(bgr_img: np.ndarray) -> np.ndarray:
    rgb = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    return srgb_to_linear(rgb)


def linear_rgb_to_bgr(linear_rgb: np.ndarray) -> np.ndarray:
    srgb = linear_to_srgb(linear_rgb)
    bgr = cv2.cvtColor((srgb * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
    return bgr


# ==================== Basic Generator Class ====================

class BaseSunglassGenerator:
    def __init__(self):
        self.reset_to_defaults()
        self._precompute_tint_matrices()
    
    def reset_to_defaults(self):
        self.DARK_INTENSITY = 0.6
        self.SATURATION_BOOST = 1.3
        self.GLARE_THRESHOLD = 0.85
        self.GLARE_REDUCTION = 0.3
        self.CONTRAST_GAMMA = 0.9
        self.ADD_TINT = False
        self.TINT_MATRIX = None
        self.ENABLE_SHARPEN = True
        self.SHARPEN_STRENGTH = 0.5
    
    def _precompute_tint_matrices(self):
        self.tint_matrices = {
            'brown': np.array([[0.9, 0.1, 0.0], [0.1, 0.7, 0.1], [0.0, 0.1, 0.6]]),
            'blue': np.array([[0.7, 0.1, 0.0], [0.1, 0.8, 0.1], [0.0, 0.1, 1.1]]),
            'vintage': np.array([[1.0, 0.1, 0.0], [0.0, 0.9, 0.1], [0.0, 0.0, 0.7]]),
            'neutral': np.array([[0.7, 0.0, 0.0], [0.0, 0.7, 0.0], [0.0, 0.0, 0.7]])
        }
    
    def configure_from_mode(self, mode: str, seed: Optional[int] = None):
        self.reset_to_defaults()
        if mode == 'brown':
            self.config_brown_sunglasses()
        elif mode == 'blue':
            self.config_blue_polarized()
        elif mode == 'strong_glare':
            self.config_strong_glare_reduction()
        elif mode == 'vintage':
            self.config_vintage()
        elif mode == 'random':
            self.config_random_augmentation(seed)

    def config_brown_sunglasses(self):
        self.DARK_INTENSITY = 0.6
        self.SATURATION_BOOST = 1.2
        self.ADD_TINT = True
        self.TINT_MATRIX = self.tint_matrices['brown']
        self.GLARE_REDUCTION = 0.3
        self.CONTRAST_GAMMA = 0.9

    def config_blue_polarized(self):
        self.DARK_INTENSITY = 0.55
        self.SATURATION_BOOST = 1.4
        self.ADD_TINT = True
        self.TINT_MATRIX = self.tint_matrices['blue']
        self.GLARE_REDUCTION = 0.45
        self.CONTRAST_GAMMA = 0.95

    def config_strong_glare_reduction(self):
        self.DARK_INTENSITY = 0.7
        self.GLARE_THRESHOLD = 0.75
        self.GLARE_REDUCTION = 0.5
        self.ADD_TINT = False
        self.SATURATION_BOOST = 1.1

    def config_vintage(self):
        self.DARK_INTENSITY = 0.65
        self.SATURATION_BOOST = 0.9
        self.ADD_TINT = True
        self.TINT_MATRIX = self.tint_matrices['vintage']
        self.CONTRAST_GAMMA = 0.85
        self.GLARE_REDUCTION = 0.2

    def config_random_augmentation(self, seed: Optional[int] = None):
        if seed is None:
            seed = random.randint(0, 2**32 - 1)
        random.seed(seed)
        np.random.seed(seed)
        
        self.DARK_INTENSITY = random.uniform(0.5, 0.8)
        self.SATURATION_BOOST = random.uniform(0.9, 1.5)
        self.GLARE_REDUCTION = random.uniform(0.2, 0.5)
        self.CONTRAST_GAMMA = random.uniform(0.8, 1.0)
        self.SHARPEN_STRENGTH = random.uniform(0.3, 0.7)
        
        if random.random() > 0.5:
            self.ADD_TINT = True
            tint_types = ['brown', 'blue', 'vintage', 'neutral']
            selected_tint = random.choice(tint_types)
            self.TINT_MATRIX = self.tint_matrices[selected_tint]
        else:
            self.ADD_TINT = False

    def apply_sunglasses_effect(self, img: np.ndarray) -> Optional[np.ndarray]:
        if img is None:
            return None
        try:
            if len(img.shape) == 2:
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            elif len(img.shape) == 3 and img.shape[2] == 1:
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            
            img_bgr = ensure_bgr(img)
            original_h, original_w = img_bgr.shape[:2]
            img_linear = bgr_to_linear_rgb(img_bgr)
            h, w, c = img_linear.shape
            pixels = img_linear.reshape(-1, 3)
            
            pixels = pixels * self.DARK_INTENSITY
            luminance = 0.2126 * pixels[:, 0] + 0.7152 * pixels[:, 1] + 0.0722 * pixels[:, 2]
            glare_mask = np.clip((luminance - self.GLARE_THRESHOLD) / (1.0 - self.GLARE_THRESHOLD + 1e-6), 0, 1)
            for i in range(3):
                pixels[:, i] = pixels[:, i] * (1.0 - glare_mask * self.GLARE_REDUCTION)
            
            if abs(self.SATURATION_BOOST - 1.0) > 1e-6:
                gray = 0.2126 * pixels[:, 0] + 0.7152 * pixels[:, 1] + 0.0722 * pixels[:, 2]
                gray = np.stack([gray] * 3, axis=1)
                pixels = pixels * self.SATURATION_BOOST + gray * (1 - self.SATURATION_BOOST)
            
            if self.ADD_TINT and self.TINT_MATRIX is not None:
                pixels = pixels @ self.TINT_MATRIX.T
            
            pixels = np.power(np.clip(pixels, 0, 1), self.CONTRAST_GAMMA)
            
            if self.ENABLE_SHARPEN and self.SHARPEN_STRENGTH > 0:
                img_temp = pixels.reshape(h, w, c)
                blurred = cv2.GaussianBlur(img_temp, (0, 0), 3)
                high_freq = img_temp - blurred
                img_temp = img_temp + high_freq * self.SHARPEN_STRENGTH
                pixels = img_temp.reshape(-1, 3)
            
            pixels = np.clip(pixels, 0, 1)
            img_processed = pixels.reshape(h, w, c)
            result = linear_rgb_to_bgr(img_processed)
            
            if result.shape[:2] != (original_h, original_w):
                result = cv2.resize(result, (original_w, original_h))
            return result
        except Exception as e:
            logger.error(f"Error in sunglasses effect: {str(e)}")
            return None


class BaseColorBlindGenerator:
    def __init__(self):
        self.reset_to_defaults()
        self._precompute_correct_matrices()
    
    def reset_to_defaults(self):
        self.COLORBLIND_TYPE = 'deuteranopia'
        self.SEVERITY = 1.0
    
    def _precompute_correct_matrices(self):
        self.rgb_to_lms = np.array([
            [0.4002, 0.7076, -0.0808],
            [-0.2263, 1.1653, 0.0457],
            [0.0, 0.0, 0.9182]
        ])
        self.lms_to_rgb = np.linalg.inv(self.rgb_to_lms)
        self.vischeck_matrices = {
            'protanopia': np.array([[0.1121, 0.8854, -0.0005], [0.1121, 0.8854, -0.0005], [0.0040, 0.0000, 0.9990]]),
            'deuteranopia': np.array([[0.2920, 0.7050, -0.0000], [0.2920, 0.7050, -0.0000], [-0.0210, 0.0300, 1.0000]]),
            'tritanopia': np.array([[0.959, 0.041, 0.000], [0.041, 0.959, 0.000], [0.000, 0.000, 0.000]])
        }
    
    def configure_from_type(self, cb_type: str, severity: Union[float, str], seed: Optional[int] = None):
        self.reset_to_defaults()
        if cb_type != 'random':
            if cb_type in ['protanopia', 'deuteranopia', 'tritanopia']:
                self.COLORBLIND_TYPE = cb_type
            if severity == 'random':
                if seed is not None:
                    random.seed(seed)
                    self.SEVERITY = random.uniform(0.5, 1.0)
                else:
                    self.SEVERITY = random.uniform(0.5, 1.0)
            else:
                self.SEVERITY = float(severity)
        else:
            self.config_random_augmentation(seed)
    
    def config_random_augmentation(self, seed: Optional[int] = None):
        if seed is None:
            seed = random.randint(0, 2**32 - 1)
        random.seed(seed)
        np.random.seed(seed)
        types = ['protanopia', 'deuteranopia', 'tritanopia']
        self.COLORBLIND_TYPE = random.choice(types)
        self.SEVERITY = random.uniform(0.5, 1.0)

    def simulate_colorblindness(self, img: np.ndarray, type_str: str = 'protanopia', 
                                severity: float = 1.0) -> Optional[np.ndarray]:
        if img is None:
            return None
        try:
            if len(img.shape) == 2:
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            elif len(img.shape) == 3 and img.shape[2] == 1:
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            
            img_bgr = ensure_bgr(img)
            original_h, original_w = img_bgr.shape[:2]
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
            img_linear = srgb_to_linear(img_rgb)
            
            h, w, c = img_linear.shape
            pixels = img_linear.reshape(-1, 3)
            lms_pixels = pixels @ self.rgb_to_lms.T
            lms_proj = self.vischeck_matrices[type_str]
            lms_simulated = lms_pixels @ lms_proj.T
            rgb_simulated = lms_simulated @ self.lms_to_rgb.T
            rgb_simulated = pixels * (1.0 - severity) + rgb_simulated * severity
            rgb_simulated = np.clip(rgb_simulated, 0.0, 1.0)
            
            img_sim_linear = rgb_simulated.reshape(h, w, c)
            img_sim_srgb = linear_to_srgb(img_sim_linear)
            img_result = (img_sim_srgb * 255.0).astype(np.uint8)
            result = cv2.cvtColor(img_result, cv2.COLOR_RGB2BGR)
            
            if result.shape[:2] != (original_h, original_w):
                result = cv2.resize(result, (original_w, original_h))
            return result
        except Exception as e:
            logger.error(f"Error in colorblind simulation: {str(e)}")
            return None


# ==================== Enhanced Generator ====================

class EnhancedSunglassGenerator(BaseSunglassGenerator):
    
    def __init__(self):
        super().__init__()
        self.ENABLE_EDGE_ENHANCE = True
        self.EDGE_STRENGTH = 0.4
        self.ENABLE_ILLUMINATION_NORMALIZATION = True
        self.ILLUMINATION_STRENGTH = 0.3
    
    def config_random_augmentation(self, seed: Optional[int] = None):
        super().config_random_augmentation(seed)
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
        self.EDGE_STRENGTH = random.uniform(0.3, 0.6)
        self.ILLUMINATION_STRENGTH = random.uniform(0.2, 0.4)
    
    def extract_multiscale_edges(self, gray: np.ndarray) -> np.ndarray:
        edge_maps = []
        for low_thresh, high_thresh in [(30, 80), (50, 150), (70, 200)]:
            edges = cv2.Canny(gray, low_thresh, high_thresh)
            edge_maps.append(edges)
        for sigma in [1, 2]:
            blurred = cv2.GaussianBlur(gray, (0, 0), sigma)
            grad_x = cv2.Sobel(blurred, cv2.CV_64F, 1, 0, ksize=3)
            grad_y = cv2.Sobel(blurred, cv2.CV_64F, 0, 1, ksize=3)
            magnitude = np.sqrt(grad_x**2 + grad_y**2)
            magnitude = (magnitude / (magnitude.max() + 1e-6) * 255).astype(np.uint8)
            edge_maps.append(magnitude)
        return np.max(edge_maps, axis=0)
    
    def apply_edge_enhancement(self, img: np.ndarray) -> np.ndarray:
        if not self.ENABLE_EDGE_ENHANCE:
            return img
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        edges = self.extract_multiscale_edges(gray)
        edges_float = edges.astype(np.float32) / 255.0
        edges_float = np.power(edges_float, 0.8)
        edges_enhanced = (edges_float * 255).astype(np.uint8)
        edges_3ch = cv2.cvtColor(edges_enhanced, cv2.COLOR_GRAY2BGR)
        return cv2.addWeighted(img, 1 - self.EDGE_STRENGTH, edges_3ch, self.EDGE_STRENGTH, 0)
    
    def apply_illumination_normalization(self, img: np.ndarray) -> np.ndarray:
        if not self.ENABLE_ILLUMINATION_NORMALIZATION:
            return img
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        l_norm = cv2.normalize(l.astype(np.float32), None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        mean_l = np.mean(l_norm)
        gamma = np.log(0.5) / np.log(mean_l / 255.0 + 1e-6) if mean_l > 0 else 1.0
        gamma = np.clip(gamma, 0.5, 1.5)
        l_gamma = (np.power(l_norm / 255.0, gamma) * 255).astype(np.uint8)
        l_fused = cv2.addWeighted(l, 1 - self.ILLUMINATION_STRENGTH, l_gamma, self.ILLUMINATION_STRENGTH, 0)
        lab_norm = cv2.merge([l_fused, a, b])
        return cv2.cvtColor(lab_norm, cv2.COLOR_LAB2BGR)
    
    def apply_sunglasses_effect(self, img: np.ndarray) -> Optional[np.ndarray]:
        result = super().apply_sunglasses_effect(img)
        if result is None:
            return None
        result = self.apply_edge_enhancement(result)
        result = self.apply_illumination_normalization(result)
        return result


class EnhancedColorBlindGenerator(BaseColorBlindGenerator):
    
    def __init__(self):
        super().__init__()
        self.ENABLE_TEXTURE_ENHANCE = True
        self.ENABLE_CLAHE = True
        self.TEXTURE_STRENGTH = 0.3
        self.CLAHE_CLIP_LIMIT = 2.0
    
    def config_random_augmentation(self, seed: Optional[int] = None):
        super().config_random_augmentation(seed)
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
        self.TEXTURE_STRENGTH = random.uniform(0.2, 0.5)
        self.CLAHE_CLIP_LIMIT = random.uniform(1.5, 2.5)
    
    def apply_texture_enhancement(self, img: np.ndarray) -> np.ndarray:
        if not self.ENABLE_TEXTURE_ENHANCE:
            return img
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        laplacian = np.abs(laplacian)
        laplacian = (laplacian / (laplacian.max() + 1e-6) * 255).astype(np.uint8)
        laplacian_3ch = cv2.cvtColor(laplacian, cv2.COLOR_GRAY2BGR)
        
        sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        gradient_mag = np.sqrt(sobel_x**2 + sobel_y**2)
        gradient_mag = (gradient_mag / (gradient_mag.max() + 1e-6) * 255).astype(np.uint8)
        gradient_3ch = cv2.cvtColor(gradient_mag, cv2.COLOR_GRAY2BGR)
        
        texture_fused = cv2.addWeighted(laplacian_3ch, 0.5, gradient_3ch, 0.5, 0)
        return cv2.addWeighted(img, 1 - self.TEXTURE_STRENGTH, texture_fused, self.TEXTURE_STRENGTH, 0)
    
    def apply_clahe(self, img: np.ndarray) -> np.ndarray:
        if not self.ENABLE_CLAHE:
            return img
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=self.CLAHE_CLIP_LIMIT, tileGridSize=(8, 8))
        l_enhanced = clahe.apply(l)
        lab_enhanced = cv2.merge([l_enhanced, a, b])
        return cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2BGR)
    
    def simulate_colorblindness(self, img: np.ndarray, type_str: str = 'protanopia', 
                                severity: float = 1.0) -> Optional[np.ndarray]:
        result = super().simulate_colorblindness(img, type_str, severity)
        if result is None:
            return None
        result = self.apply_texture_enhancement(result)
        result = self.apply_clahe(result)
        return np.clip(result, 0, 255).astype(np.uint8)


class EnhancedEdgeGenerator:
    
    def __init__(self):
        self.EDGE_MODE = 'soft'
        self.ENABLE_MULTISCALE = True
        self.SOFT_EDGES = True
        self.SOFT_EDGE_BLUR = 1.0
        self.THIN_EDGES = False
    
    def configure_from_mode(self, mode: str):
        self.EDGE_MODE = mode
        if mode == 'soft':
            self.SOFT_EDGES = True
            self.SOFT_EDGE_BLUR = 1.5
            self.THIN_EDGES = False
        elif mode == 'sharp':
            self.SOFT_EDGES = False
            self.SOFT_EDGE_BLUR = 0.0
            self.THIN_EDGES = True
        elif mode == 'binary':
            self.SOFT_EDGES = False
            self.SOFT_EDGE_BLUR = 0.0
            self.THIN_EDGES = True
    
    def generate_edge_gt(self, img: np.ndarray) -> Optional[np.ndarray]:
        if img is None:
            return None
        try:
            if len(img.shape) == 3:
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            else:
                gray = img.copy()
            
            original_h, original_w = gray.shape[:2]
            
            if self.ENABLE_MULTISCALE:
                edges_list = []
                edges1 = cv2.Canny(gray, 30, 80)
                edges_list.append(edges1)
                edges2 = cv2.Canny(gray, 50, 150)
                edges_list.append(edges2)
                blurred = cv2.GaussianBlur(gray, (5, 5), 2)
                edges3 = cv2.Canny(blurred, 70, 200)
                edges_list.append(edges3)
                edge_map = np.max(edges_list, axis=0)
            else:
                edge_map = cv2.Canny(gray, 50, 150)
            
            if self.SOFT_EDGES and self.SOFT_EDGE_BLUR > 0:
                edge_map = cv2.GaussianBlur(edge_map.astype(np.float32), (0, 0), self.SOFT_EDGE_BLUR)
                edge_map = (edge_map / (edge_map.max() + 1e-6) * 255).astype(np.uint8)
            
            if edge_map.shape[0] != original_h or edge_map.shape[1] != original_w:
                edge_map = cv2.resize(edge_map, (original_w, original_h), interpolation=cv2.INTER_NEAREST)
            return edge_map
        except Exception as e:
            logger.error(f"Error generating edge GT: {str(e)}")
            return None


# ==================== data generator ====================

class SimpleDataGenerator:
    def __init__(self, sg_class, cb_class, eg, worker_id=0):
        self.sg_class = sg_class
        self.cb_class = cb_class
        self.eg = eg
        self.worker_id = worker_id
        self.lock = threading.Lock()
        self.stats = {'processed': 0, 'failed': 0, 'sunglass': 0, 'colorblind': 0, 'edge': 0}
    
    def _safe_update_stats(self, key, value=1):
        with self.lock:
            self.stats[key] = self.stats.get(key, 0) + value
    
    def _safe_imwrite(self, path: Path, img: np.ndarray, params: List[int]) -> bool:
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            return cv2.imwrite(str(path), img, params)
        except Exception as e:
            logger.error(f"Error writing {path}: {str(e)}")
            return False
    
    def process_file(self, img_path: Path, gt_path: Path, output_dirs: Dict) -> bool:
        sg = self.sg_class()
        cb = self.cb_class()
    
        try:
            img = cv2.imread(str(img_path))
            gt = cv2.imread(str(gt_path))
        
            if img is None:
                logger.error(f"Failed to read image: {img_path}")
                self._safe_update_stats('failed')
                return False
        
            if gt is None:
                logger.error(f"Failed to read GT: {gt_path}")
                self._safe_update_stats('failed')
                return False
        
            valid, msg = validate_image(img, img_path)
            if not valid:
                logger.error(f"Invalid image {img_path}: {msg}")
                self._safe_update_stats('failed')
                return False
        
            if img.shape[:2] != gt.shape[:2]:
                logger.info(f"Resizing GT for {img_path.name}: {gt.shape[:2]} -> {img.shape[:2]}")
                gt = cv2.resize(gt, (img.shape[1], img.shape[0]))
        
            stem = img_path.stem
            ext = f".{OUTPUT_FORMAT}"
            save_params = [cv2.IMWRITE_PNG_COMPRESSION, 9] if OUTPUT_FORMAT == "png" else [cv2.IMWRITE_JPEG_QUALITY, OUTPUT_QUALITY]
        
            file_seed = get_file_based_seed(img_path, RANDOM_SEED)
            success = True
        
            if not SKIP_SUNGLASS:
                if SUNGLASS_MODE == 'random' and not USE_FIXED_SUNGLASS_PARAMS:
                    sg.config_random_augmentation(seed=file_seed)
                else:
                    sg.configure_from_mode(SUNGLASS_MODE, seed=file_seed)
            
                sunglass_result = sg.apply_sunglasses_effect(img)
                if sunglass_result is not None:
                    sunglass_path = output_dirs['sunglass'] / f"{stem}{ext}"
                    if self._safe_imwrite(sunglass_path, sunglass_result, save_params):
                        self._safe_update_stats('sunglass')
                    else:
                        success = False
                else:
                    logger.warning(f"Failed to generate sunglass effect for {img_path.name}")
                    success = False
            else:
                logger.debug(f"Skipping sunglass generation for {img_path.name}")
        
            if not SKIP_COLORBLIND:
                if COLORBLIND_TYPE == 'random' and not USE_FIXED_COLORBLIND_PARAMS:
                    cb.config_random_augmentation(seed=file_seed)
                else:
                    cb.configure_from_type(COLORBLIND_TYPE, COLORBLIND_SEVERITY, seed=file_seed)
            
                colorblind_result = cb.simulate_colorblindness(img, type_str=cb.COLORBLIND_TYPE, severity=cb.SEVERITY)
                if colorblind_result is not None:
                    colorblind_path = output_dirs['colorblind'] / f"{stem}{ext}"
                    if self._safe_imwrite(colorblind_path, colorblind_result, save_params):
                        self._safe_update_stats('colorblind')
                    else:
                        success = False
                else:
                    logger.warning(f"Failed to generate colorblind effect for {img_path.name}")
                    success = False
            else:
                logger.debug(f"Skipping colorblind generation for {img_path.name}")
        
            if not SKIP_EDGE:
                edge_map = self.eg.generate_edge_gt(gt)
                if edge_map is not None:
                    edge_path = output_dirs['edge'] / f"{stem}.png"
                    if self._safe_imwrite(edge_path, edge_map, [cv2.IMWRITE_PNG_COMPRESSION, 9]):
                        self._safe_update_stats('edge')
                    else:
                        success = False
                else:
                    logger.warning(f"Failed to generate edge GT for {img_path.name}")
                    success = False
            else:
                logger.debug(f"Skipping edge GT generation for {img_path.name}")
        
            if success:
                self._safe_update_stats('processed')
            else:
                self._safe_update_stats('failed')
        
            return success
        
        except Exception as e:
            logger.error(f"Error processing {img_path.name}: {str(e)}")
            logger.error(traceback.format_exc())
            self._safe_update_stats('failed')
            return False
    
    def get_stats(self) -> Dict:
        with self.lock:
            return self.stats.copy()


# ==================== auxiliary function ====================

def create_output_dirs(base_path: str) -> Dict:
    base = Path(base_path)
    dirs = {}
    for key, subdir in OUTPUT_SUBDIRS.items():
        dir_path = base / subdir
        dir_path.mkdir(parents=True, exist_ok=True)
        dirs[key] = dir_path
    return dirs


def find_matching_gt(image_path: Path, gt_path: Path) -> Optional[Path]:
    stem = image_path.stem
    patterns = [
        image_path.name,
        f"{stem}.png",
        f"{stem}.jpg",
        f"{stem}_gt.png",
        f"{stem}_mask.png",
        f"{stem}_gt.jpg",
        f"{stem}_mask.jpg"
    ]
    
    for pattern in patterns:
        potential = gt_path / pattern
        if potential.exists():
            return potential
    return None


def collect_image_files(image_path: Path) -> List[Path]:
    extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp', '.tif', '.jfif']
    files = []
    for ext in extensions:
        files.extend(list(image_path.glob(f'*{ext}')))
        files.extend(list(image_path.glob(f'*{ext.upper()}')))
    # 使用 dict 去重并保持顺序
    return list(dict.fromkeys(files))


def run_generation_pipeline(force_reprocess: bool = False, no_backup: bool = False):
    import time as time_module
    
    image_path = Path(INPUT_IMAGE_PATH)
    gt_path = Path(INPUT_GT_PATH)
    
    if not validate_paths(image_path, gt_path):
        logger.error("Path validation failed")
        return
    
    output_base = Path(OUTPUT_BASE)
    
    if output_base != image_path.parent:
        backup_existing_outputs(output_base, no_backup)
    
    dirs_to_create = {}
    if not SKIP_SUNGLASS:
        dirs_to_create['sunglass'] = OUTPUT_SUBDIRS['sunglass']
    if not SKIP_COLORBLIND:
        dirs_to_create['colorblind'] = OUTPUT_SUBDIRS['colorblind']
    if not SKIP_EDGE:
        dirs_to_create['edge'] = OUTPUT_SUBDIRS['edge']
    
    # output_dirs = create_output_dirs(OUTPUT_BASE)
    
    output_dirs = {}
    for key, subdir in dirs_to_create.items():
        dir_path = output_base / subdir
        dir_path.mkdir(parents=True, exist_ok=True)
        output_dirs[key] = dir_path
    
    image_files = collect_image_files(image_path)
    
    if not image_files:
        logger.info(f"No images found in {INPUT_IMAGE_PATH}")
        return
    
    valid_pairs = []
    missing_gt = []
    for img_file in sorted(image_files):
        gt_file = find_matching_gt(img_file, gt_path)
        if gt_file:
            valid_pairs.append((img_file, gt_file))
        else:
            missing_gt.append(img_file.name)
    
    if missing_gt:
        logger.warning(f"Found {len(missing_gt)} images without matching GT")
        if len(missing_gt) <= 10:
            for name in missing_gt:
                logger.warning(f"  - {name}")
    
    if not valid_pairs:
        logger.error("No valid image-GT pairs found")
        return
    
    print(f"\n{'='*70}")
    print(f"Enhanced Data Generation Pipeline (Tri-Stream Support)")
    print(f"{'='*70}")
    print(f"Input directory: {INPUT_IMAGE_PATH}")
    print(f"Output directory: {OUTPUT_BASE}")
    print(f"Found {len(valid_pairs)} valid image-GT pairs")
    print(f"Missing GT: {len(missing_gt)}")
    print(f"\nGenerators:")
    print(f"  - Colorblind: Texture-enhanced (CLAHE + Multi-scale Texture)")
    print(f"  - Sunglass: Edge-enhanced (Multi-scale Edges + Illumination Normalization)")
    print(f"  - Edge: {'Soft' if EDGE_MODE=='soft' else 'Sharp'} edges, Multi-scale: Yes")
    print(f"{'='*70}\n")
    
    eg = EnhancedEdgeGenerator()
    eg.configure_from_mode(EDGE_MODE)
    
    data_gen = SimpleDataGenerator(EnhancedSunglassGenerator, EnhancedColorBlindGenerator, eg, 0)
    
    print("Start processing...\n")
    
    if SHOW_PROGRESS_BAR and TQDM_AVAILABLE:
        from time import time
        start_time = time()
        
        with tqdm(total=len(valid_pairs), desc="Processing", unit="img",
                  ncols=80, bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]",
                  position=0, leave=True) as pbar:
            
            for idx, (img_file, gt_file) in enumerate(valid_pairs):
                success = data_gen.process_file(img_file, gt_file, output_dirs)
                pbar.update(1)
                
                stats = data_gen.get_stats()
                elapsed = time_module.time() - start_time
                speed = stats['processed'] / elapsed if elapsed > 0 else 0
                pbar.set_postfix({
                    'S': stats['processed'],
                    'F': stats['failed'],
                    'Spd': f'{speed:.1f}'
                })
                
                if FORCE_GC_AFTER_BATCH and (idx + 1) % 50 == 0:
                    gc.collect()
                    if check_memory_threshold():
                        time_module.sleep(0.5)
        
        print()
    else:
        for idx, (img_file, gt_file) in enumerate(valid_pairs):
            success = data_gen.process_file(img_file, gt_file, output_dirs)
            if (idx + 1) % 10 == 0:
                stats = data_gen.get_stats()
                print(f"  Progress: {idx+1}/{len(valid_pairs)} | Success: {stats['processed']} | Failed: {stats['failed']}")
            
            if FORCE_GC_AFTER_BATCH and (idx + 1) % 100 == 0:
                gc.collect()
                if check_memory_threshold():
                    time_module.sleep(0.5)
    
    total_stats = data_gen.get_stats()
    
    print(f"\n{'='*70}")
    print(f"Generation Completed!")
    print(f"Total pairs: {len(valid_pairs)}")
    print(f"Successfully processed: {total_stats['processed']}")
    print(f"Failed: {total_stats['failed']}")
    print(f"\nGenerated files:")
    print(f"  - Sunglass (Edge-enhanced): {total_stats.get('sunglass', 0)}")
    print(f"  - Colorblind (Texture-enhanced): {total_stats.get('colorblind', 0)}")
    print(f"  - Edge: {total_stats.get('edge', 0)}")
    print(f"{'='*70}")


def main():
    args = parse_arguments()
    
    global INPUT_IMAGE_PATH, INPUT_GT_PATH, OUTPUT_BASE, CACHE_FILE
    global SUNGLASS_MODE, COLORBLIND_TYPE, COLORBLIND_SEVERITY, EDGE_MODE
    global OUTPUT_FORMAT, RANDOM_SEED, SHOW_PROGRESS_BAR
    global SKIP_EDGE, SKIP_SUNGLASS, SKIP_COLORBLIND
    
    data_path = Path(args.path)
    INPUT_IMAGE_PATH = str(data_path / "Imgs")
    INPUT_GT_PATH = str(data_path / "GT")
    
    SKIP_EDGE = args.skip_edge
    SKIP_SUNGLASS = args.skip_sunglass
    SKIP_COLORBLIND = args.skip_colorblind
    
    if args.output_dir:
        OUTPUT_BASE = args.output_dir
    else:
        OUTPUT_BASE = str(data_path)
    
    CACHE_FILE = os.path.join(OUTPUT_BASE, "processing_cache.json")
    
    SUNGLASS_MODE = args.sunglass_mode
    COLORBLIND_TYPE = args.colorblind_type
    COLORBLIND_SEVERITY = args.colorblind_severity
    EDGE_MODE = args.edge_mode
    OUTPUT_FORMAT = args.output_format
    RANDOM_SEED = args.seed
    SHOW_PROGRESS_BAR = not args.no_progress
    
    print(f"\n{'='*70}")
    print(f"Enhanced Data Generation Pipeline")
    print(f"{'='*70}")
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Input dataset: {data_path}")
    print(f"Output directory: {OUTPUT_BASE}")
    print(f"\nStream Configurations:")
    if not SKIP_SUNGLASS:
        print(f"  - Sunglass Stream: {SUNGLASS_MODE}, edge-enhanced")
    else:
        print(f"  - Sunglass Stream: SKIPPED")
    
    if not SKIP_COLORBLIND:
        print(f"  - Colorblind Stream: {COLORBLIND_TYPE}, severity={COLORBLIND_SEVERITY}, texture-enhanced")
    else:
        print(f"  - Colorblind Stream: SKIPPED")
    
    if not SKIP_EDGE:
        print(f"  - Edge Supervision: {EDGE_MODE} edges")
    else:
        print(f"  - Edge Supervision: SKIPPED")
    
    print(f"{'='*70}\n")
    
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    
    run_generation_pipeline(force_reprocess=args.force_reprocess, no_backup=args.no_backup)
    
    print(f"\nEnd time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()