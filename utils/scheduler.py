# utils/scheduler.py

import torch.optim as optim


def create_scheduler(optimizer, config, steps_per_epoch=None):
    """Step-based learning rate scheduler (with warm-up)"""
    if steps_per_epoch is not None and config.warmup_epochs > 0:
        warmup_steps = config.warmup_epochs * steps_per_epoch
        total_steps = config.num_epochs * steps_per_epoch

        warmup_scheduler = optim.lr_scheduler.LinearLR(
            optimizer,
            start_factor=config.warmup_start_lr / config.learning_rate,
            end_factor=1.0,
            total_iters=warmup_steps
        )

        if config.use_cosine_annealing:
            main_scheduler = optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=total_steps - warmup_steps,
                eta_min=1e-7
            )
        else:
            main_scheduler = optim.lr_scheduler.LinearLR(
                optimizer,
                start_factor=1.0,
                end_factor=1e-7 / config.learning_rate,
                total_iters=total_steps - warmup_steps
            )

        scheduler = optim.lr_scheduler.SequentialLR(
            optimizer,
            schedulers=[warmup_scheduler, main_scheduler],
            milestones=[warmup_steps]
        )
        print(f"Using SequentialLR (step-based): warmup {warmup_steps} steps")
        return scheduler

    elif config.warmup_epochs > 0:
        warmup_scheduler = optim.lr_scheduler.LinearLR(
            optimizer,
            start_factor=config.warmup_start_lr / config.learning_rate,
            end_factor=1.0,
            total_iters=config.warmup_epochs
        )

        if config.use_cosine_annealing:
            main_scheduler = optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=config.num_epochs - config.warmup_epochs,
                eta_min=1e-7
            )
        else:
            main_scheduler = optim.lr_scheduler.LinearLR(
                optimizer,
                start_factor=1.0,
                end_factor=1e-7 / config.learning_rate,
                total_iters=config.num_epochs - config.warmup_epochs
            )

        scheduler = optim.lr_scheduler.SequentialLR(
            optimizer,
            schedulers=[warmup_scheduler, main_scheduler],
            milestones=[config.warmup_epochs]
        )
        print(f"Using SequentialLR (epoch-based): warmup {config.warmup_epochs} epochs")
        return scheduler
    else:
        if config.use_cosine_annealing:
            scheduler = optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=config.num_epochs,
                eta_min=1e-7
            )
        else:
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode='min', factor=0.5, patience=5, verbose=True, min_lr=1e-7
            )
        print("Using CosineAnnealingLR or ReduceLROnPlateau (no warmup)")
        return scheduler