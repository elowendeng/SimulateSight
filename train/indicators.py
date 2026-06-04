# train/indicators.py

import numpy as np
from scipy import ndimage
from scipy.ndimage import distance_transform_edt


def cal_mae(smap, gt_img):
    """
    Calculate Mean Absolute Error (MAE)
    
    Args:
        smap: Saliency map (H x W) in range [0, 1]
        gt_img: Ground truth binary image (H x W)
    
    Returns:
        mae: Mean Absolute Error score
    """
    if smap.shape != gt_img.shape:
        raise ValueError('Saliency map and gt Image have different sizes!')
    # Convert gt to binary if needed
    if gt_img.dtype != bool:
        gt_img = gt_img > 0.5
    # Ensure smap is float in [0, 1]
    smap = smap.astype(np.float64)
    fg_pixels = smap[gt_img]
    fg_err_sum = len(fg_pixels) - np.sum(fg_pixels)
    bg_err_sum = np.sum(smap[~gt_img])
    mae = (fg_err_sum + bg_err_sum) / gt_img.size
    return mae


def enhanced_measure(fm, gt):
    """
    Enhanced-alignment Measure (E-measure)
    
    Args:
        fm: Binary foreground map (H x W), type: float in range [0, 1]
        gt: Binary ground truth (H x W)
    
    Returns:
        score: Enhanced alignment score
    """
    # Convert to boolean
    fm_bool = fm > 0.5 if fm.dtype != bool else fm
    gt_bool = gt > 0.5 if gt.dtype != bool else gt
    # Use double for computations
    d_fm = fm_bool.astype(np.float64)
    d_gt = gt_bool.astype(np.float64)
    # Special cases
    if np.sum(d_gt) == 0:  # GT is completely black
        enhanced_matrix = 1.0 - d_fm
    elif np.sum(~d_gt.astype(bool)) == 0:  # GT is completely white
        enhanced_matrix = d_fm
    else:
        # Normal case
        align_matrix = alignment_term(d_fm, d_gt)
        enhanced_matrix = enhanced_alignment_term(align_matrix)
    # E-measure score
    h, w = gt.shape
    score = np.sum(enhanced_matrix) / (w * h - 1 + np.finfo(float).eps)
    return score


def alignment_term(d_fm, d_gt):
    """Alignment Term for E-measure"""
    mu_fm = np.mean(d_fm)
    mu_gt = np.mean(d_gt)
    align_fm = d_fm - mu_fm
    align_gt = d_gt - mu_gt
    align_matrix = 2.0 * (align_gt * align_fm) / (align_gt * align_gt + align_fm * align_fm + np.finfo(float).eps)
    return align_matrix


def enhanced_alignment_term(align_matrix):
    """Enhanced Alignment Term: f(x) = 1/4*(1 + x)^2"""
    return ((align_matrix + 1) ** 2) / 4


def fmeasure_calu(s_map, gt_map, threshold=None):
    """
    Calculate F-measure
    
    Args:
        s_map: Saliency map (H x W)
        gt_map: Ground truth binary map (H x W)
        threshold: Threshold value (if None, uses 2*mean)
    
    Returns:
        precision, recall, fmeasure
    """
    if threshold is None:
        threshold = min(2 * np.mean(s_map), 1)
    elif threshold > 1:
        threshold = 1
    # Binarize saliency map
    label3 = np.zeros_like(s_map, dtype=bool)
    label3[s_map >= threshold] = True
    num_rec = np.sum(label3)
    label_and = label3 & gt_map
    num_and = np.sum(label_and)
    num_obj = np.sum(gt_map)
    if num_and == 0:
        return 0.0, 0.0, 0.0
    else:
        precision = num_and / num_rec
        recall = num_and / num_obj
        fmeasure = (1.3 * precision * recall) / (0.3 * precision + recall + np.finfo(float).eps) 
        return precision, recall, fmeasure


def original_wfb(fg, gt):
    """
    Weighted F-beta measure (WFb)
    
    Args:
        fg: Foreground map in range [0, 1], type: double
        gt: Binary ground truth, type: bool
    
    Returns:
        Q: Weighted F-beta score
    """
    # Check input
    if fg.dtype != np.float64:
        fg = fg.astype(np.float64)
    if np.max(fg) > 1 or np.min(fg) < 0:
        fg = np.clip(fg, 0, 1)
    if gt.dtype != bool:
        gt = gt > 0.5
    d_gt = gt.astype(np.float64)
    # Error
    e = np.abs(fg - d_gt)
    # Pixel dependency
    dist, idx = distance_transform_edt(~gt, return_indices=True)
    # Handle 2D indices properly
    if idx.ndim == 3:
        idx_y, idx_x = idx[0], idx[1]
    else:
        idx_y, idx_x = np.indices(gt.shape)
    # Create Gaussian kernel
    k = gaussian_kernel(7, 5)
    et = e.copy()
    # For non-GT pixels, use the error value from the nearest GT pixel
    mask = ~gt
    if np.any(mask):
        et[mask] = e[idx_y[mask], idx_x[mask]]
    ea = ndimage.convolve(et, k, mode='constant')
    min_e_ea = e.copy()
    # Where GT is true and EA < E, use EA
    mask_gt = gt & (ea < e)
    min_e_ea[mask_gt] = ea[mask_gt]
    # Pixel importance
    b = np.ones_like(gt, dtype=np.float64)
    b[~gt] = 2.0 - 1.0 * np.exp(np.log(1 - 0.5) / 5.0 * dist[~gt])
    ew = min_e_ea * b
    tpw = np.sum(d_gt) - np.sum(ew[gt])
    fpw = np.sum(ew[~gt])
    # Weighted Recall and Precision
    r = 1 - np.mean(ew[gt])
    p = tpw / (np.finfo(float).eps + tpw + fpw)
    # Beta=1
    q = (2 * r * p) / (np.finfo(float).eps + r + p)
    return q


def gaussian_kernel(size, sigma):
    """Create a Gaussian kernel"""
    kernel = np.fromfunction(
        lambda x, y: (1/(2*np.pi*sigma**2)) * np.exp(-((x-(size-1)/2)**2 + (y-(size-1)/2)**2)/(2*sigma**2)),
        (size, size)
    )
    return kernel / np.sum(kernel)


def s_object(prediction, gt):
    """
    Object similarity for S-measure
    """
    # Compute foreground similarity
    prediction_fg = prediction.copy()
    prediction_fg[~gt] = 0
    o_fg = _object(prediction_fg, gt)
    # Compute background similarity
    prediction_bg = 1.0 - prediction
    prediction_bg[gt] = 0
    o_bg = _object(prediction_bg, ~gt)
    # Combine
    u = np.mean(gt.astype(np.float64))
    q = u * o_fg + (1 - u) * o_bg
    return q


def _object(prediction, gt):
    """Helper function for object similarity"""
    if np.sum(gt) == 0:
        return 0.0
    # Ensure correct types
    if prediction.dtype != np.float64:
        prediction = prediction.astype(np.float64)
    prediction = np.clip(prediction, 0, 1)
    # Mean of foreground/background
    x = np.mean(prediction[gt])
    # Standard deviation
    sigma_x = np.std(prediction[gt])
    score = 2.0 * x / (x**2 + 1.0 + sigma_x + np.finfo(float).eps)
    return score


def s_region(prediction, gt):
    """
    Region similarity for S-measure
    """
    # Find centroid of GT
    x, y = _centroid(gt)
    # Divide GT into 4 regions
    gt_1, gt_2, gt_3, gt_4, w1, w2, w3, w4 = _divide_gt(gt, x, y)
    # Divide prediction into 4 regions
    pred_1, pred_2, pred_3, pred_4 = _divide_prediction(prediction, x, y)
    # Compute SSIM for each region
    q1 = _ssim(pred_1, gt_1)
    q2 = _ssim(pred_2, gt_2)
    q3 = _ssim(pred_3, gt_3)
    q4 = _ssim(pred_4, gt_4)
    # Sum weighted scores
    q = w1 * q1 + w2 * q2 + w3 * q3 + w4 * q4
    return q


def _centroid(gt):
    """Find centroid of ground truth"""
    h, w = gt.shape
    if np.sum(gt) == 0:
        return w // 2, h // 2
    # Calculate centroid
    y_indices, x_indices = np.where(gt)
    x = int(np.round(np.mean(x_indices)))
    y = int(np.round(np.mean(y_indices)))
    # Ensure within bounds
    x = max(0, min(w-1, x))
    y = max(0, min(h-1, y))
    return x, y


def _divide_gt(gt, x, y):
    """Divide GT into 4 regions based on centroid"""
    h, w = gt.shape
    area = w * h
    # Split at centroid
    lt = gt[:y, :x]
    rt = gt[:y, x:]
    lb = gt[y:, :x]
    rb = gt[y:, x:]
    # Calculate weights
    w1 = (x * y) / area if x > 0 and y > 0 else 0
    w2 = ((w - x) * y) / area if (w-x) > 0 and y > 0 else 0
    w3 = (x * (h - y)) / area if x > 0 and (h-y) > 0 else 0
    w4 = 1.0 - w1 - w2 - w3
    return lt, rt, lb, rb, w1, w2, w3, w4


def _divide_prediction(prediction, x, y):
    """Divide prediction into 4 regions based on centroid"""
    lt = prediction[:y, :x]
    rt = prediction[:y, x:]
    lb = prediction[y:, :x]
    rb = prediction[y:, x:]
    return lt, rt, lb, rb


def _ssim(prediction, gt):
    """Region similarity (SSIM-like) for S-measure"""
    if prediction.size == 0 or gt.size == 0:
        return 0.0
    d_gt = gt.astype(np.float64)
    n = prediction.size
    # Means
    x = np.mean(prediction)
    y = np.mean(d_gt)
    # Variances
    sigma_x2 = np.sum((prediction - x)**2) / (n - 1 + np.finfo(float).eps)
    sigma_y2 = np.sum((d_gt - y)**2) / (n - 1 + np.finfo(float).eps)
    # Covariance
    sigma_xy = np.sum((prediction - x) * (d_gt - y)) / (n - 1 + np.finfo(float).eps)
    alpha = 4 * x * y * sigma_xy
    beta = (x**2 + y**2) * (sigma_x2 + sigma_y2)
    if alpha != 0:
        q = alpha / (beta + np.finfo(float).eps)
    elif alpha == 0 and beta == 0:
        q = 1.0
    else:
        q = 0.0
    return q


def structure_measure(prediction, gt):
    """
    Structure-measure (S-measure)
    """
    # Check input
    if prediction.dtype != np.float64:
        prediction = prediction.astype(np.float64)
    prediction = np.clip(prediction, 0, 1)
    if gt.dtype != bool:
        gt = gt > 0.5
    y = np.mean(gt.astype(np.float64))
    if y == 0:  # GT completely black
        x = np.mean(prediction)
        q = 1.0 - x
    elif y == 1:  # GT completely white
        x = np.mean(prediction)
        q = x
    else:
        alpha = 0.5
        q = alpha * s_object(prediction, gt) + (1 - alpha) * s_region(prediction, gt)
        if q < 0:
            q = 0 
    return q


def compute_all_metrics(pred, gt):
    """
    Compute all evaluation metrics for a prediction-gt pair
    
    Args:
        pred: Prediction map (H x W) in range [0, 1]
        gt: Ground truth binary map (H x W)
    
    Returns:
        Dictionary with all metrics
    """
    # Ensure correct types
    if pred.dtype != np.float64:
        pred = pred.astype(np.float64)
    if gt.dtype != bool:
        gt = gt > 0.5
    # Resize if needed
    if pred.shape != gt.shape:
        from skimage.transform import resize
        pred = resize(pred, gt.shape, preserve_range=True)
    # MAE
    mae = cal_mae(pred, gt)
    # S-measure
    sm = structure_measure(pred, gt)
    # Weighted F-measure
    wfm = original_wfb(pred, gt)
    # Adaptive threshold metrics
    threshold = min(2 * np.mean(pred), 1)
    p, r, adp_fm = fmeasure_calu(pred, gt, threshold)
    # E-measure with adaptive threshold
    bi_pred = pred > threshold
    adp_em = enhanced_measure(bi_pred.astype(np.float64), gt)
    # Max F-measure (over all thresholds)
    thresholds = np.arange(0, 1, 0.01)
    max_fm = 0
    for t in thresholds:
        _, _, f = fmeasure_calu(pred, gt, t)
        if f > max_fm:
            max_fm = f
    return {
        'mae': mae,
        'sm': sm,
        'wfm': wfm,
        'adp_fm': adp_fm,
        'adp_em': adp_em,
        'max_fm': max_fm
    }