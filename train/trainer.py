# train/trainer.py

import os
os.environ['NO_ALBUMENTATIONS_UPDATE'] = '1'

from tqdm import tqdm
import torch
import torch.nn.functional as F
from utils import cleanup


def train_epoch_single(model, loader, optimizer, criterion, epoch, config, scaler=None):
    model.train()
    total_loss = 0
    skipped_batches = 0

    pbar = tqdm(loader, desc=f'Epoch {epoch}')
    for batch_idx, (imgs, gts, _) in enumerate(pbar):
        imgs = imgs.to(config.device)
        gts = gts.to(config.device)

        if gts.dim() == 3:
            gts = gts.unsqueeze(1)

        optimizer.zero_grad()

        try:
            if config.use_amp and scaler is not None:
                with torch.cuda.amp.autocast():
                    preds = model(imgs)
                    if preds.dim() == 3:
                        preds = preds.unsqueeze(1)

                    if preds.shape[2:] != gts.shape[2:]:
                        preds = F.interpolate(preds, size=gts.shape[2:], mode='bilinear', align_corners=False)
                    loss = criterion(preds, gts)

                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)

                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip_norm)

                if torch.isnan(grad_norm) or grad_norm > config.grad_clip_norm * 10:
                    skipped_batches += 1
                    if skipped_batches <= 5:
                        print(f"Warning: Abnormal gradient norm {grad_norm:.2f}, skipping batch {batch_idx}")
                    optimizer.zero_grad()
                    scaler.step(optimizer)
                    scaler.update()
                    continue

                scaler.step(optimizer)
                scaler.update()
            else:
                preds = model(imgs)
                if preds.dim() == 3:
                    preds = preds.unsqueeze(1)

                if preds.shape[2:] != gts.shape[2:]:
                    preds = F.interpolate(preds, size=gts.shape[2:], mode='bilinear', align_corners=False)
                loss = criterion(preds, gts)
                loss.backward()

                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip_norm)

                if torch.isnan(grad_norm) or grad_norm > config.grad_clip_norm * 10:
                    skipped_batches += 1
                    if skipped_batches <= 5:
                        print(f"Warning: Abnormal gradient norm {grad_norm:.2f}, skipping batch {batch_idx}")
                    optimizer.zero_grad()
                    continue

                optimizer.step()

            total_loss += loss.item()

        except RuntimeError as e:
            if "unscale_" in str(e):
                print(f"Warning: Scaler error at batch {batch_idx}, skipping")
                optimizer.zero_grad()
                if scaler is not None:
                    try:
                        scaler.step(optimizer)
                        scaler.update()
                    except:
                        pass
                continue
            else:
                raise e

        if batch_idx % config.log_interval == 0:
            pbar.set_postfix({'loss': f'{loss.item():.4f}', 'skip': skipped_batches})

    if skipped_batches > 0:
        print(f"Epoch {epoch}: Skipped {skipped_batches} batches due to gradient issues")

    return total_loss / len(loader)


def train_epoch_dual(model, loader, optimizer, criterion, epoch, config, scaler=None):
    model.train()
    total_loss = 0
    skipped_batches = 0

    pbar = tqdm(loader, desc=f'Epoch {epoch}')
    for batch_idx, (imgs, cbs, gts, _) in enumerate(pbar):
        imgs = imgs.to(config.device)
        cbs = cbs.to(config.device)
        gts = gts.to(config.device)

        if gts.dim() == 3:
            gts = gts.unsqueeze(1)

        optimizer.zero_grad()

        try:
            if config.use_amp and scaler is not None:
                with torch.cuda.amp.autocast():
                    preds = model(imgs, cbs)
                    if preds.dim() == 3:
                        preds = preds.unsqueeze(1)

                    if preds.shape[2:] != gts.shape[2:]:
                        preds = F.interpolate(preds, size=gts.shape[2:], mode='bilinear', align_corners=False)
                    loss = criterion(preds, gts)

                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)

                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip_norm)

                if torch.isnan(grad_norm) or grad_norm > config.grad_clip_norm * 10:
                    skipped_batches += 1
                    if skipped_batches <= 5:
                        print(f"Warning: Abnormal gradient norm {grad_norm:.2f}, skipping batch {batch_idx}")
                    optimizer.zero_grad()
                    scaler.step(optimizer)
                    scaler.update()
                    continue

                scaler.step(optimizer)
                scaler.update()
            else:
                preds = model(imgs, cbs)
                if preds.dim() == 3:
                    preds = preds.unsqueeze(1)

                if preds.shape[2:] != gts.shape[2:]:
                    preds = F.interpolate(preds, size=gts.shape[2:], mode='bilinear', align_corners=False)
                loss = criterion(preds, gts)
                loss.backward()

                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip_norm)

                if torch.isnan(grad_norm) or grad_norm > config.grad_clip_norm * 10:
                    skipped_batches += 1
                    if skipped_batches <= 5:
                        print(f"Warning: Abnormal gradient norm {grad_norm:.2f}, skipping batch {batch_idx}")
                    optimizer.zero_grad()
                    continue

                optimizer.step()

            total_loss += loss.item()

        except RuntimeError as e:
            if "unscale_" in str(e):
                print(f"Warning: Scaler error at batch {batch_idx}, skipping")
                optimizer.zero_grad()
                if scaler is not None:
                    try:
                        scaler.step(optimizer)
                        scaler.update()
                    except:
                        pass
                continue
            else:
                raise e

        if batch_idx % config.log_interval == 0:
            pbar.set_postfix({'loss': f'{loss.item():.4f}', 'skip': skipped_batches})

    if skipped_batches > 0:
        print(f"Epoch {epoch}: Skipped {skipped_batches} batches due to gradient issues")

    return total_loss / len(loader)


def train_epoch_dual_with_aux(model, loader, optimizer, criterion, epoch, config, scaler=None):
    model.train()
    total_loss = 0
    total_main_loss = 0
    total_aux_loss = 0
    skipped_batches = 0

    pbar = tqdm(loader, desc=f'Epoch {epoch}')
    for batch_idx, (imgs, cbs, gts, _) in enumerate(pbar):
        imgs = imgs.to(config.device)
        cbs = cbs.to(config.device)
        gts = gts.to(config.device)

        if gts.dim() == 3:
            gts = gts.unsqueeze(1)

        optimizer.zero_grad()

        try:
            if config.use_amp and scaler is not None:
                with torch.cuda.amp.autocast():
                    main_pred, aux_preds = model(imgs, cbs, return_aux=True)

                    if main_pred.shape[2:] != gts.shape[2:]:
                        main_pred = F.interpolate(main_pred, size=gts.shape[2:],
                                                  mode='bilinear', align_corners=False)

                    for i in range(len(aux_preds)):
                        if aux_preds[i].shape[2:] != gts.shape[2:]:
                            aux_preds[i] = F.interpolate(aux_preds[i], size=gts.shape[2:],
                                                         mode='bilinear', align_corners=False)

                    loss, main_loss, aux_loss = criterion.forward_with_aux(main_pred, aux_preds, gts)

                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)

                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip_norm)

                if torch.isnan(grad_norm) or grad_norm > config.grad_clip_norm * 10:
                    skipped_batches += 1
                    if skipped_batches <= 5:
                        print(f"Warning: Abnormal gradient norm {grad_norm:.2f}, skipping batch {batch_idx}")
                    optimizer.zero_grad()
                    scaler.step(optimizer)
                    scaler.update()
                    continue

                scaler.step(optimizer)
                scaler.update()

            else:
                main_pred, aux_preds = model(imgs, cbs, return_aux=True)

                if main_pred.shape[2:] != gts.shape[2:]:
                    main_pred = F.interpolate(main_pred, size=gts.shape[2:],
                                              mode='bilinear', align_corners=False)

                for i in range(len(aux_preds)):
                    if aux_preds[i].shape[2:] != gts.shape[2:]:
                        aux_preds[i] = F.interpolate(aux_preds[i], size=gts.shape[2:],
                                                     mode='bilinear', align_corners=False)

                loss, main_loss, aux_loss = criterion.forward_with_aux(main_pred, aux_preds, gts)
                loss.backward()

                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip_norm)

                if torch.isnan(grad_norm) or grad_norm > config.grad_clip_norm * 10:
                    skipped_batches += 1
                    if skipped_batches <= 5:
                        print(f"Warning: Abnormal gradient norm {grad_norm:.2f}, skipping batch {batch_idx}")
                    optimizer.zero_grad()
                    continue

                optimizer.step()

            total_loss += loss.item()
            total_main_loss += main_loss.item()
            total_aux_loss += aux_loss.item()

        except RuntimeError as e:
            print(f"Warning: {e} at batch {batch_idx}, skipping")
            skipped_batches += 1
            optimizer.zero_grad()
            if scaler is not None:
                try:
                    scaler.step(optimizer)
                    scaler.update()
                except:
                    pass
            continue

        if batch_idx % config.log_interval == 0:
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'main': f'{main_loss.item():.4f}',
                'aux': f'{aux_loss.item():.4f}',
                'skip': skipped_batches
            })

        if batch_idx % 100 == 0:
            cleanup()

    if skipped_batches > 0:
        print(f"Epoch {epoch}: Skipped {skipped_batches} batches due to errors")

    return {
        'total_loss': total_loss / len(loader),
        'main_loss': total_main_loss / len(loader),
        'aux_loss': total_aux_loss / len(loader)
    }


def train_epoch_triple(model, loader, optimizer, criterion, epoch, config, scaler=None):
    model.train()
    total_loss = 0
    skipped_batches = 0
    
    pbar = tqdm(loader, desc=f'Epoch {epoch}')
    for batch_idx, (imgs, cbs, sls, gts, _) in enumerate(pbar):
        imgs = imgs.to(config.device)
        cbs = cbs.to(config.device)
        sls = sls.to(config.device)
        gts = gts.to(config.device)
        
        if gts.dim() == 3:
            gts = gts.unsqueeze(1)
        
        optimizer.zero_grad()
        
        try:
            if config.use_amp and scaler is not None:
                with torch.cuda.amp.autocast():
                    preds = model(imgs, cbs, sls, return_aux=False)
                    if preds.dim() == 3:
                        preds = preds.unsqueeze(1)
                    
                    if preds.shape[2:] != gts.shape[2:]:
                        preds = F.interpolate(preds, size=gts.shape[2:], mode='bilinear', align_corners=False)
                    loss = criterion(preds, gts)
                
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip_norm)
                
                if torch.isnan(grad_norm) or grad_norm > config.grad_clip_norm * 10:
                    skipped_batches += 1
                    if skipped_batches <= 5:
                        print(f"Warning: Abnormal gradient norm {grad_norm:.2f}, skipping batch {batch_idx}")
                    optimizer.zero_grad()
                    scaler.step(optimizer)
                    scaler.update()
                    continue
                
                scaler.step(optimizer)
                scaler.update()
            else:
                preds = model(imgs, cbs, sls, return_aux=False)
                if preds.dim() == 3:
                    preds = preds.unsqueeze(1)
                
                if preds.shape[2:] != gts.shape[2:]:
                    preds = F.interpolate(preds, size=gts.shape[2:], mode='bilinear', align_corners=False)
                loss = criterion(preds, gts)
                loss.backward()
                
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip_norm)
                
                if torch.isnan(grad_norm) or grad_norm > config.grad_clip_norm * 10:
                    skipped_batches += 1
                    if skipped_batches <= 5:
                        print(f"Warning: Abnormal gradient norm {grad_norm:.2f}, skipping batch {batch_idx}")
                    optimizer.zero_grad()
                    continue
                
                optimizer.step()
            
            total_loss += loss.item()
            
        except RuntimeError as e:
            if "unscale_" in str(e):
                print(f"Warning: Scaler error at batch {batch_idx}, skipping")
                optimizer.zero_grad()
                if scaler is not None:
                    try:
                        scaler.step(optimizer)
                        scaler.update()
                    except:
                        pass
                continue
            else:
                raise e
        
        if batch_idx % config.log_interval == 0:
            pbar.set_postfix({'loss': f'{loss.item():.4f}', 'skip': skipped_batches})
    
    if skipped_batches > 0:
        print(f"Epoch {epoch}: Skipped {skipped_batches} batches due to gradient issues")
    
    return total_loss / len(loader)


def train_epoch_triple_with_aux(model, loader, optimizer, criterion, epoch, config, scaler=None):
    model.train()
    total_loss = 0
    total_main_loss = 0
    total_aux_loss = 0
    skipped_batches = 0
    
    pbar = tqdm(loader, desc=f'Epoch {epoch}')
    for batch_idx, (imgs, cbs, sls, gts, _) in enumerate(pbar):
        imgs = imgs.to(config.device)
        cbs = cbs.to(config.device)
        sls = sls.to(config.device)
        gts = gts.to(config.device)
        
        if gts.dim() == 3:
            gts = gts.unsqueeze(1)
        
        optimizer.zero_grad()
        
        try:
            if config.use_amp and scaler is not None:
                with torch.cuda.amp.autocast():
                    main_pred, aux_preds = model(imgs, cbs, sls, return_aux=True)
                    
                    if main_pred.dim() == 3:
                        main_pred = main_pred.unsqueeze(1)
                    
                    if main_pred.shape[2:] != gts.shape[2:]:
                        main_pred = F.interpolate(main_pred, size=gts.shape[2:], 
                                                 mode='bilinear', align_corners=False)
                    
                    for i in range(len(aux_preds)):
                        if aux_preds[i].dim() == 3:
                            aux_preds[i] = aux_preds[i].unsqueeze(1)
                        if aux_preds[i].shape[2:] != gts.shape[2:]:
                            aux_preds[i] = F.interpolate(aux_preds[i], size=gts.shape[2:],
                                                        mode='bilinear', align_corners=False)
                    
                    loss, main_loss, aux_loss = criterion.forward_with_aux(main_pred, aux_preds, gts)
                
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip_norm)
                
                if torch.isnan(grad_norm) or grad_norm > config.grad_clip_norm * 10:
                    skipped_batches += 1
                    if skipped_batches <= 5:
                        print(f"Warning: Abnormal gradient norm {grad_norm:.2f}, skipping batch {batch_idx}")
                    optimizer.zero_grad()
                    scaler.step(optimizer)
                    scaler.update()
                    continue
                
                scaler.step(optimizer)
                scaler.update()
            else:
                main_pred, aux_preds = model(imgs, cbs, sls, return_aux=True)
                
                if main_pred.dim() == 3:
                    main_pred = main_pred.unsqueeze(1)
                
                if main_pred.shape[2:] != gts.shape[2:]:
                    main_pred = F.interpolate(main_pred, size=gts.shape[2:], 
                                             mode='bilinear', align_corners=False)
                
                for i in range(len(aux_preds)):
                    if aux_preds[i].dim() == 3:
                        aux_preds[i] = aux_preds[i].unsqueeze(1)
                    if aux_preds[i].shape[2:] != gts.shape[2:]:
                        aux_preds[i] = F.interpolate(aux_preds[i], size=gts.shape[2:],
                                                    mode='bilinear', align_corners=False)
                
                loss, main_loss, aux_loss = criterion.forward_with_aux(main_pred, aux_preds, gts)
                loss.backward()
                
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip_norm)
                
                if torch.isnan(grad_norm) or grad_norm > config.grad_clip_norm * 10:
                    skipped_batches += 1
                    if skipped_batches <= 5:
                        print(f"Warning: Abnormal gradient norm {grad_norm:.2f}, skipping batch {batch_idx}")
                    optimizer.zero_grad()
                    continue
                
                optimizer.step()
            
            total_loss += loss.item()
            total_main_loss += main_loss.item()
            total_aux_loss += aux_loss.item()
            
        except (ValueError, RuntimeError) as e:
            print(f"Warning: {e} at batch {batch_idx}, skipping")
            skipped_batches += 1
            optimizer.zero_grad()
            if scaler is not None:
                try:
                    scaler.step(optimizer)
                    scaler.update()
                except:
                    pass
            continue
        
        if batch_idx % config.log_interval == 0:
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'main': f'{main_loss.item():.4f}',
                'aux': f'{aux_loss.item():.4f}',
                'skip': skipped_batches
            })
        
        if batch_idx % 100 == 0:
            cleanup()
    
    if skipped_batches > 0:
        print(f"Epoch {epoch}: Skipped {skipped_batches} batches due to errors")
    
    return {
        'total_loss': total_loss / len(loader),
        'main_loss': total_main_loss / len(loader),
        'aux_loss': total_aux_loss / len(loader)
    }