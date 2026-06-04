# dataset/datasets.py

import os
os.environ['NO_ALBUMENTATIONS_UPDATE'] = '1'

import cv2
import numpy as np
import torch
from utils.lru_cache import LRUCache


class SingleStreamDataset(torch.utils.data.Dataset):
    """Single-flow COD dataset"""

    def __init__(self, img_dir, gt_dir, img_size=352, transform=None, is_train=False):
        for dir_path, dir_name in [(img_dir, 'Image'), (gt_dir, 'GT')]:
            if not os.path.exists(dir_path):
                raise FileNotFoundError(f"{dir_name} directory not found: {dir_path}")

        self.img_dir = img_dir
        self.gt_dir = gt_dir
        self.img_size = img_size
        self.transform = transform
        self.is_train = is_train

        self.cache_size = 0 if is_train else 200
        self.img_cache = LRUCache(maxsize=self.cache_size)
        self.gt_cache = LRUCache(maxsize=self.cache_size)

        self.samples = []
        valid_ext = ['.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG', '.bmp', '.BMP']

        self.gt_map = {}
        if os.path.exists(gt_dir):
            for f in os.listdir(gt_dir):
                name, ext = os.path.splitext(f)
                if ext.lower() in valid_ext:
                    self.gt_map[name] = os.path.join(gt_dir, f)

        for f in os.listdir(img_dir):
            name, ext = os.path.splitext(f)
            if ext.lower() in valid_ext:
                img_path = os.path.join(img_dir, f)
                gt_path = self.gt_map.get(name)

                if gt_path is not None:
                    self.samples.append((img_path, gt_path, name))

        print(f"Found {len(self.samples)} valid samples in {img_dir}")

        if len(self.samples) == 0:
            raise ValueError(f"No valid samples found in {img_dir}")

    def __len__(self):
        return len(self.samples)

    def _load_image(self, img_path):
        """Image loading with cache"""
        cached_img = self.img_cache.get(img_path)
        if cached_img is not None:
            return cached_img

        img = cv2.imread(img_path)
        if img is None:
            raise ValueError(f"Failed to load image: {img_path}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        if not self.is_train:
            self.img_cache.put(img_path, img)

        return img

    def _load_gt(self, gt_path):
        """GT loading with cache"""
        cached_gt = self.gt_cache.get(gt_path)
        if cached_gt is not None:
            return cached_gt

        gt = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)
        if gt is None:
            raise ValueError(f"Failed to load GT: {gt_path}")

        if not self.is_train:
            self.gt_cache.put(gt_path, gt)

        return gt

    def __getitem__(self, idx):
        try:
            img_path, gt_path, name = self.samples[idx]

            img = self._load_image(img_path)
            gt = self._load_gt(gt_path)
            orig_h, orig_w = gt.shape[:2]

            img = cv2.resize(img, (self.img_size, self.img_size))
            gt = cv2.resize(gt, (self.img_size, self.img_size), interpolation=cv2.INTER_NEAREST)

            img = img.astype(np.float32) / 255.0
            gt = (gt > 128).astype(np.float32)

            if self.transform is not None and self.is_train:
                transformed = self.transform(image=img, mask=gt)
                img = transformed['image']
                gt = transformed['mask']
                if gt.dim() == 2:
                    gt = gt.unsqueeze(0)
            else:
                img = torch.from_numpy(img).permute(2, 0, 1).float()
                gt = torch.from_numpy(gt).unsqueeze(0).float()

            return img, gt, (orig_h, orig_w)

        except Exception as e:
            print(f"Error loading sample {idx}: {e}")
            return self.__getitem__((idx + 1) % len(self))

    def clear_cache(self):
        """clear cache"""
        self.img_cache.clear()
        self.gt_cache.clear()


def collate_fn_single(batch):
    imgs = torch.stack([item[0] for item in batch])
    gts = torch.stack([item[1] for item in batch])
    orig_sizes = [(item[2][0], item[2][1]) for item in batch]
    return imgs, gts, orig_sizes


class DualStreamDataset(torch.utils.data.Dataset):
    """Dual-flow COD dataset"""

    def __init__(self, img_dir, cb_dir, gt_dir, img_size=352, is_train=True, transform=None):
        for dir_path, dir_name in [(img_dir, 'Image'), (cb_dir, 'CB'), (gt_dir, 'GT')]:
            if not os.path.exists(dir_path):
                raise FileNotFoundError(f"{dir_name} directory not found: {dir_path}")

        self.img_dir = img_dir
        self.cb_dir = cb_dir
        self.gt_dir = gt_dir
        self.img_size = img_size
        self.is_train = is_train
        self.transform = transform

        self.cache_size = 0 if is_train else 200
        self.img_cache = LRUCache(maxsize=self.cache_size)
        self.cb_cache = LRUCache(maxsize=self.cache_size)
        self.gt_cache = LRUCache(maxsize=self.cache_size)

        self.samples = []
        valid_ext = ['.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG', '.bmp', '.BMP']

        self.gt_map = {}
        for f in os.listdir(gt_dir):
            name, ext = os.path.splitext(f)
            if ext.lower() in valid_ext:
                self.gt_map[name] = os.path.join(gt_dir, f)

        self.cb_map = {}
        for f in os.listdir(cb_dir):
            name, ext = os.path.splitext(f)
            if ext.lower() in valid_ext:
                self.cb_map[name] = os.path.join(cb_dir, f)

        for f in os.listdir(img_dir):
            name, ext = os.path.splitext(f)
            if ext.lower() in valid_ext:
                img_path = os.path.join(img_dir, f)
                gt_path = self.gt_map.get(name)
                cb_path = self.cb_map.get(name)

                if gt_path is not None and cb_path is not None:
                    self.samples.append((img_path, cb_path, gt_path, name))

        print(f"Found {len(self.samples)} valid samples in {img_dir}")

        if len(self.samples) == 0:
            raise ValueError(f"No valid samples found in {img_dir}")

    def __len__(self):
        return len(self.samples)

    def _load_image(self, img_path):
        """Image loading with cache"""
        cached_img = self.img_cache.get(img_path)
        if cached_img is not None:
            return cached_img

        img = cv2.imread(img_path)
        if img is None:
            raise ValueError(f"Failed to load image: {img_path}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        if not self.is_train:
            self.img_cache.put(img_path, img)

        return img

    def _load_cb(self, cb_path):
        """CB image loading with cache"""
        cached_cb = self.cb_cache.get(cb_path)
        if cached_cb is not None:
            return cached_cb

        cb = cv2.imread(cb_path)
        if cb is None:
            raise ValueError(f"Failed to load CB image: {cb_path}")
        cb = cv2.cvtColor(cb, cv2.COLOR_BGR2RGB)

        if not self.is_train:
            self.cb_cache.put(cb_path, cb)

        return cb

    def _load_gt(self, gt_path):
        """GT loading with cache"""
        cached_gt = self.gt_cache.get(gt_path)
        if cached_gt is not None:
            return cached_gt

        gt = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)
        if gt is None:
            raise ValueError(f"Failed to load GT: {gt_path}")

        if not self.is_train:
            self.gt_cache.put(gt_path, gt)

        return gt

    def __getitem__(self, idx):
        try:
            img_path, cb_path, gt_path, name = self.samples[idx]

            img = self._load_image(img_path)
            cb = self._load_cb(cb_path)
            gt = self._load_gt(gt_path)
            orig_h, orig_w = gt.shape[:2]

            # Resize
            img = cv2.resize(img, (self.img_size, self.img_size))
            cb = cv2.resize(cb, (self.img_size, self.img_size))
            gt = cv2.resize(gt, (self.img_size, self.img_size), interpolation=cv2.INTER_NEAREST)

            img = img.astype(np.float32) / 255.0
            cb = cb.astype(np.float32) / 255.0
            gt = (gt > 128).astype(np.float32)

            # Data augmentation (only for the training set)
            if self.transform is not None and self.is_train:
                transformed = self.transform(image=img, mask=gt, cb=cb)
                img = transformed['image']
                cb = transformed['cb']
                gt = transformed['mask']
                if gt.dim() == 2:
                    gt = gt.unsqueeze(0)
            else:
                img = torch.from_numpy(img).permute(2, 0, 1).float()
                cb = torch.from_numpy(cb).permute(2, 0, 1).float()
                gt = torch.from_numpy(gt).unsqueeze(0).float()

            return img, cb, gt, (orig_h, orig_w)

        except Exception as e:
            print(f"Error loading sample {idx}: {e}")
            return self.__getitem__((idx + 1) % len(self))

    def clear_cache(self):
        """clear cache"""
        self.img_cache.clear()
        self.cb_cache.clear()
        self.gt_cache.clear()


def collate_fn_dual(batch):
    imgs = torch.stack([item[0] for item in batch])
    cbs = torch.stack([item[1] for item in batch])
    gts = torch.stack([item[2] for item in batch])
    orig_sizes = [(item[3][0], item[3][1]) for item in batch]
    return imgs, cbs, gts, orig_sizes


class TripleStreamDataset(torch.utils.data.Dataset):
    """Triple-flow COD dataset"""
    
    def __init__(self, img_dir, cb_dir, sl_dir, gt_dir, img_size=352, 
                 transform=None, is_train=False):
        for dir_path, dir_name in [(img_dir, 'Image'), (cb_dir, 'CB'), 
                                   (sl_dir, 'SL'), (gt_dir, 'GT')]:
            if not os.path.exists(dir_path):
                raise FileNotFoundError(f"{dir_name} directory not found: {dir_path}")
        
        self.img_dir = img_dir
        self.cb_dir = cb_dir
        self.sl_dir = sl_dir
        self.gt_dir = gt_dir
        self.img_size = img_size
        self.transform = transform
        self.is_train = is_train
        
        self.cache_size = 0 if is_train else 200
        self.img_cache = LRUCache(maxsize=self.cache_size)
        self.cb_cache = LRUCache(maxsize=self.cache_size)
        self.sl_cache = LRUCache(maxsize=self.cache_size)
        self.gt_cache = LRUCache(maxsize=self.cache_size)
        
        self.samples = []
        valid_ext = ['.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG', '.bmp', '.BMP']
        
        self.gt_map = {}
        for f in os.listdir(gt_dir):
            name, ext = os.path.splitext(f)
            if ext.lower() in valid_ext:
                self.gt_map[name] = os.path.join(gt_dir, f)
        
        self.cb_map = {}
        for f in os.listdir(cb_dir):
            name, ext = os.path.splitext(f)
            if ext.lower() in valid_ext:
                self.cb_map[name] = os.path.join(cb_dir, f)
        
        self.sl_map = {}
        for f in os.listdir(sl_dir):
            name, ext = os.path.splitext(f)
            if ext.lower() in valid_ext:
                self.sl_map[name] = os.path.join(sl_dir, f)
        
        for f in os.listdir(img_dir):
            name, ext = os.path.splitext(f)
            if ext.lower() in valid_ext:
                img_path = os.path.join(img_dir, f)
                gt_path = self.gt_map.get(name)
                cb_path = self.cb_map.get(name)
                sl_path = self.sl_map.get(name)
                
                if all([gt_path, cb_path, sl_path]):
                    self.samples.append((img_path, cb_path, sl_path, gt_path, name))
        
        print(f"Found {len(self.samples)} valid samples in {img_dir}")
        
        if len(self.samples) == 0:
            raise ValueError(f"No valid samples found in {img_dir}")
    
    def __len__(self):
        return len(self.samples)
    
    def _load_image(self, img_path):
        """Image loading with cache"""
        cached_img = self.img_cache.get(img_path)
        if cached_img is not None:
            return cached_img
        
        img = cv2.imread(img_path)
        if img is None:
            raise ValueError(f"Failed to load image: {img_path}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        if not self.is_train:
            self.img_cache.put(img_path, img)
        
        return img
    
    def _load_cb(self, cb_path):
        """CB image loading with cache"""
        cached_cb = self.cb_cache.get(cb_path)
        if cached_cb is not None:
            return cached_cb
        
        cb = cv2.imread(cb_path)
        if cb is None:
            raise ValueError(f"Failed to load CB image: {cb_path}")
        cb = cv2.cvtColor(cb, cv2.COLOR_BGR2RGB)
        
        if not self.is_train:
            self.cb_cache.put(cb_path, cb)
        
        return cb
    
    def _load_sl(self, sl_path):
        """SL image loading with cache"""
        cached_sl = self.sl_cache.get(sl_path)
        if cached_sl is not None:
            return cached_sl
        
        sl = cv2.imread(sl_path)
        if sl is None:
            raise ValueError(f"Failed to load SL image: {sl_path}")
        sl = cv2.cvtColor(sl, cv2.COLOR_BGR2RGB)
        
        if not self.is_train:
            self.sl_cache.put(sl_path, sl)
        
        return sl
    
    def _load_gt(self, gt_path):
        """GT loading with cache"""
        cached_gt = self.gt_cache.get(gt_path)
        if cached_gt is not None:
            return cached_gt
        
        gt = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)
        if gt is None:
            raise ValueError(f"Failed to load GT: {gt_path}")
        
        if not self.is_train:
            self.gt_cache.put(gt_path, gt)
        
        return gt
    
    def __getitem__(self, idx):
        try:
            img_path, cb_path, sl_path, gt_path, name = self.samples[idx]
            
            img = self._load_image(img_path)
            cb = self._load_cb(cb_path)
            sl = self._load_sl(sl_path)
            gt = self._load_gt(gt_path)
            orig_h, orig_w = gt.shape[:2]
            
            img = cv2.resize(img, (self.img_size, self.img_size))
            cb = cv2.resize(cb, (self.img_size, self.img_size))
            sl = cv2.resize(sl, (self.img_size, self.img_size))
            gt = cv2.resize(gt, (self.img_size, self.img_size), interpolation=cv2.INTER_NEAREST)
            
            img = img.astype(np.float32) / 255.0
            cb = cb.astype(np.float32) / 255.0
            sl = sl.astype(np.float32) / 255.0
            gt = (gt > 128).astype(np.float32)
            
            if self.transform is not None and self.is_train:
                transformed = self.transform(
                    image=img,
                    mask=gt,
                    cb=cb,
                    sl=sl
                )
                img = transformed['image']
                cb = transformed['cb']
                sl = transformed['sl']
                gt = transformed['mask']
                if gt.dim() == 2:
                    gt = gt.unsqueeze(0)
            else:
                img = torch.from_numpy(img).permute(2, 0, 1).float()
                cb = torch.from_numpy(cb).permute(2, 0, 1).float()
                sl = torch.from_numpy(sl).permute(2, 0, 1).float()
                gt = torch.from_numpy(gt).unsqueeze(0).float()
            
            return img, cb, sl, gt, (orig_h, orig_w)
            
        except Exception as e:
            print(f"Error loading sample {idx}: {e}")
            return self.__getitem__((idx + 1) % len(self))
    
    def clear_cache(self):
        """clear cache"""
        self.img_cache.clear()
        self.cb_cache.clear()
        self.sl_cache.clear()
        self.gt_cache.clear()


def collate_fn_triple(batch):
    imgs = torch.stack([item[0] for item in batch])
    cbs = torch.stack([item[1] for item in batch])
    sls = torch.stack([item[2] for item in batch])
    gts = torch.stack([item[3] for item in batch])
    orig_sizes = [(item[4][0], item[4][1]) for item in batch]
    return imgs, cbs, sls, gts, orig_sizes