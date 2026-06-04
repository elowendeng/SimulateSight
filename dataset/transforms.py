# dataset/data_aug.py

import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2


def get_train_transforms_single(config):
    transforms = []

    if config.hflip_prob > 0:
        transforms.append(A.HorizontalFlip(p=config.hflip_prob))
    if config.vflip_prob > 0:
        transforms.append(A.VerticalFlip(p=config.vflip_prob))
    if config.rotation_degree > 0:
        transforms.append(A.Rotate(limit=config.rotation_degree, p=0.5, border_mode=cv2.BORDER_CONSTANT, value=0))

    if hasattr(config, 'brightness_limit') and config.brightness_limit > 0:
        transforms.append(
            A.ColorJitter(
                brightness=config.brightness_limit,
                contrast=config.contrast_limit,
                saturation=config.saturation_limit,
                hue=config.hue_limit,
                p=0.5
            )
        )

    transforms.append(ToTensorV2())
    return A.Compose(transforms, additional_targets={'mask': 'mask'})


def get_train_transforms_dual(config):
    transforms = []
    
    if config.hflip_prob > 0:
        transforms.append(A.HorizontalFlip(p=config.hflip_prob))
    if config.vflip_prob > 0:
        transforms.append(A.VerticalFlip(p=config.vflip_prob))
    if config.rotation_degree > 0:
        transforms.append(A.Rotate(limit=config.rotation_degree, p=0.5, border_mode=cv2.BORDER_CONSTANT, value=0))
    
    if hasattr(config, 'brightness_limit') and config.brightness_limit > 0:
        transforms.append(
            A.ColorJitter(
                brightness=config.brightness_limit,
                contrast=config.contrast_limit,
                saturation=config.saturation_limit,
                hue=config.hue_limit,
                p=0.5
            )
        )
    
    transforms.append(ToTensorV2())
    return A.Compose(
        transforms, 
        additional_targets={
            'mask': 'mask',
            'cb': 'image'
        }
    )


def get_train_transforms_triple(config):
    transforms = []
    
    if config.hflip_prob > 0:
        transforms.append(A.HorizontalFlip(p=config.hflip_prob))
    if config.vflip_prob > 0:
        transforms.append(A.VerticalFlip(p=config.vflip_prob))
    if config.rotation_degree > 0:
        transforms.append(A.Rotate(limit=config.rotation_degree, p=0.5, border_mode=cv2.BORDER_CONSTANT, value=0))
    
    if hasattr(config, 'brightness_limit') and config.brightness_limit > 0:
        transforms.append(
            A.ColorJitter(
                brightness=config.brightness_limit,
                contrast=config.contrast_limit,
                saturation=config.saturation_limit,
                hue=config.hue_limit,
                p=0.5
            )
        )
    
    transforms.append(ToTensorV2())
    return A.Compose(
        transforms, 
        additional_targets={
            'mask': 'mask',
            'cb': 'image',
            'sl': 'image'
        }
    )


def get_val_transforms(config):
    return A.Compose([ToTensorV2()])