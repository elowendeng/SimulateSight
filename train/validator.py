# train/validator.py

import os
os.environ['NO_ALBUMENTATIONS_UPDATE'] = '1'

import cv2
from tqdm import tqdm
import torch
from .indicators import compute_all_metrics


@torch.no_grad()
def validate_batch_single(model, loader, config):

    model.eval()
    metrics = {'mae': 0, 'sm': 0, 'wfm': 0, 'adp_fm': 0, 'adp_em': 0, 'max_fm': 0}
    total_samples = 0

    for imgs, gts, orig_sizes in tqdm(loader, desc='Validating'):
        imgs = imgs.to(config.device, non_blocking=True)

        if config.use_amp:
            with torch.cuda.amp.autocast():
                preds = model(imgs)
        else:
            preds = model(imgs)

        preds = torch.sigmoid(preds).cpu().numpy()
        gts_np = gts.cpu().numpy()

        batch_size = imgs.size(0)

        for i in range(batch_size):
            pred = preds[i, 0] if preds[i].ndim > 2 else preds[i]
            gt = gts_np[i, 0] if gts_np[i].ndim > 2 else gts_np[i]

            orig_h, orig_w = orig_sizes[i]
            orig_h, orig_w = int(orig_h), int(orig_w)

            if pred.shape != (orig_h, orig_w):
                pred = cv2.resize(pred, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)
            if gt.shape != (orig_h, orig_w):
                gt = cv2.resize(gt, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)

            result = compute_all_metrics(pred, gt)
            for k in metrics:
                metrics[k] += result[k]
            total_samples += 1

    for k in metrics:
        metrics[k] /= max(total_samples, 1)

    return metrics


@torch.no_grad()
def validate_batch_dual(model, loader, config):
    model.eval()
    metrics = {'mae': 0, 'sm': 0, 'wfm': 0, 'adp_fm': 0, 'adp_em': 0, 'max_fm': 0}
    total_samples = 0

    for imgs, cbs, gts, orig_sizes in tqdm(loader, desc='Validating'):
        imgs = imgs.to(config.device, non_blocking=True)
        cbs = cbs.to(config.device, non_blocking=True)

        if config.use_amp:
            with torch.cuda.amp.autocast():
                preds = model(imgs, cbs)
        else:
            preds = model(imgs, cbs)

        preds = torch.sigmoid(preds).cpu().numpy()
        gts_np = gts.cpu().numpy()

        batch_size = imgs.size(0)

        for i in range(batch_size):
            pred = preds[i, 0] if preds[i].ndim > 2 else preds[i]
            gt = gts_np[i, 0] if gts_np[i].ndim > 2 else gts_np[i]

            orig_h, orig_w = orig_sizes[i]
            orig_h, orig_w = int(orig_h), int(orig_w)

            if pred.shape != (orig_h, orig_w):
                pred = cv2.resize(pred, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)
            if gt.shape != (orig_h, orig_w):
                gt = cv2.resize(gt, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)

            result = compute_all_metrics(pred, gt)
            for k in metrics:
                metrics[k] += result[k]
            total_samples += 1

    for k in metrics:
        metrics[k] /= max(total_samples, 1)

    return metrics


@torch.no_grad()
def validate_batch_triple(model, loader, config):
    model.eval()
    metrics = {'mae': 0, 'sm': 0, 'wfm': 0, 'adp_fm': 0, 'adp_em': 0, 'max_fm': 0}
    total_samples = 0
    
    for imgs, cbs, sls, gts, orig_sizes in tqdm(loader, desc='Validating'):
        imgs = imgs.to(config.device, non_blocking=True)
        cbs = cbs.to(config.device, non_blocking=True)
        sls = sls.to(config.device, non_blocking=True)
        
        if config.use_amp:
            with torch.cuda.amp.autocast():
                preds = model(imgs, cbs, sls, return_aux=False)
        else:
            preds = model(imgs, cbs, sls, return_aux=False)
        
        preds = torch.sigmoid(preds).cpu().numpy()
        gts_np = gts.cpu().numpy()
        
        batch_size = imgs.size(0)
        
        for i in range(batch_size):
            pred = preds[i, 0] if preds[i].ndim > 2 else preds[i]
            gt = gts_np[i, 0] if gts_np[i].ndim > 2 else gts_np[i]
            
            orig_h, orig_w = orig_sizes[i]
            orig_h, orig_w = int(orig_h), int(orig_w)
            
            if pred.shape != (orig_h, orig_w):
                pred = cv2.resize(pred, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)
            if gt.shape != (orig_h, orig_w):
                gt = cv2.resize(gt, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)
            
            result = compute_all_metrics(pred, gt)
            for k in metrics:
                metrics[k] += result[k]
            total_samples += 1
    
    for k in metrics:
        metrics[k] /= max(total_samples, 1)
    
    return metrics