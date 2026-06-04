# utils/complementarity.py

import numpy as np
from tqdm import tqdm
import torch


class DualStreamCorrelationAnalyzer:
    def __init__(self, device):
        self.device = device
        self.history = {
            'prediction_correlation': [],
            'diversity_score': []
        }

    def compute_prediction_correlation(self, pred1, pred2):
        prob1 = torch.sigmoid(pred1)
        prob2 = torch.sigmoid(pred2)

        flat1 = prob1.view(prob1.shape[0], -1).detach().cpu().numpy()
        flat2 = prob2.view(prob2.shape[0], -1).detach().cpu().numpy()

        correlations = []
        for i in range(flat1.shape[0]):
            corr = np.corrcoef(flat1[i], flat2[i])[0, 1]
            if not np.isnan(corr):
                correlations.append(corr)

        return np.mean(correlations) if correlations else 0.5

    def compute_diversity_score(self, pred1, pred2):
        similarity = self.compute_prediction_correlation(pred1, pred2)
        return 1 - similarity

    def _get_stream_outputs(self, model, imgs, cbs, config):
        target_size = imgs.shape[2:]

        if hasattr(model, 'aux_decoder_img') and model.aux_decoder_img is not None:
            if hasattr(model, 'extract_features'):
                feats1, feats2 = model.extract_features(imgs, cbs)
            else:
                if model.share_backbone:
                    feats1 = model.backbone(imgs)
                    feats2 = model.backbone(cbs)
                else:
                    feats1 = model.backbone1(imgs)
                    feats2 = model.backbone2(cbs)

            out1 = model.aux_decoder_img(feats1, target_size=target_size)
            out2 = model.aux_decoder_cb(feats2, target_size=target_size)
            return out1, out2

        if hasattr(model, 'extract_features'):
            feats1, feats2 = model.extract_features(imgs, cbs)
        else:
            if model.share_backbone:
                feats1 = model.backbone(imgs)
                feats2 = model.backbone(cbs)
            else:
                feats1 = model.backbone1(imgs)
                feats2 = model.backbone2(cbs)

        out1 = model.decoder(feats1, feats1, target_size=target_size)
        out2 = model.decoder(feats2, feats2, target_size=target_size)

        return out1, out2

    def analyze(self, model, loader, config):
        model.eval()
        all_pred_corr = []

        with torch.no_grad():
            for imgs, cbs, gts, _ in tqdm(loader, desc='Analyzing complementarity', leave=False):
                imgs = imgs.to(config.device)
                cbs = cbs.to(config.device)

                try:
                    out1, out2 = self._get_stream_outputs(model, imgs, cbs, config)
                    pred_corr = self.compute_prediction_correlation(out1, out2)
                    all_pred_corr.append(pred_corr)
                except Exception as e:
                    print(f"Warning: Could not analyze batch: {e}")
                    continue

        if len(all_pred_corr) == 0:
            pred_corr_mean = 0.5
        else:
            pred_corr_mean = np.mean(all_pred_corr)

        diversity = 1 - pred_corr_mean

        self.history['prediction_correlation'].append(pred_corr_mean)
        self.history['diversity_score'].append(diversity)

        return {
            'prediction_correlation': pred_corr_mean,
            'diversity_score': diversity,
            'is_complementary': diversity > 0.3
        }


class TripleStreamCorrelationAnalyzer:
    def __init__(self, device):
        self.device = device
        self.history = {
            'img_cb_correlation': [],
            'img_sl_correlation': [],
            'cb_sl_correlation': [],
            'diversity_score': []
        }
    
    def compute_prediction_correlation(self, pred1, pred2):
        prob1 = torch.sigmoid(pred1)
        prob2 = torch.sigmoid(pred2)
        
        flat1 = prob1.view(prob1.shape[0], -1).detach().cpu().numpy()
        flat2 = prob2.view(prob2.shape[0], -1).detach().cpu().numpy()
        
        correlations = []
        for i in range(flat1.shape[0]):
            corr = np.corrcoef(flat1[i], flat2[i])[0, 1]
            if not np.isnan(corr):
                correlations.append(corr)
        
        return np.mean(correlations) if correlations else 0.5
    
    def compute_diversity_score(self, pred1, pred2):
        similarity = self.compute_prediction_correlation(pred1, pred2)
        return 1 - similarity
    
    def _get_stream_outputs(self, model, imgs, cbs, sls, config):
        target_size = imgs.shape[2:]
        
        if hasattr(model, 'aux_decoder_img') and model.aux_decoder_img is not None:
            if hasattr(model, 'extract_features'):
                feats_img, feats_cb, feats_sl = model.extract_features(imgs, cbs, sls)
            else:
                if model.share_backbone:
                    feats_img = model.backbone(imgs)
                    feats_cb = model.backbone(cbs)
                    feats_sl = model.backbone(sls)
                else:
                    feats_img = model.backbone1(imgs)
                    feats_cb = model.backbone2(cbs)
                    feats_sl = model.backbone3(sls)
            
            out_img = model.aux_decoder_img(feats_img, target_size=target_size)
            out_cb = model.aux_decoder_cb(feats_cb, target_size=target_size)
            out_sl = model.aux_decoder_sl(feats_sl, target_size=target_size)
            return out_img, out_cb, out_sl
        
        if hasattr(model, 'extract_features'):
            feats_img, feats_cb, feats_sl = model.extract_features(imgs, cbs, sls)
        else:
            if model.share_backbone:
                feats_img = model.backbone(imgs)
                feats_cb = model.backbone(cbs)
                feats_sl = model.backbone(sls)
            else:
                feats_img = model.backbone1(imgs)
                feats_cb = model.backbone2(cbs)
                feats_sl = model.backbone3(sls)
        
        out_img = model.decoder(feats_img, feats_img, feats_img, target_size=target_size)
        out_cb = model.decoder(feats_cb, feats_cb, feats_cb, target_size=target_size)
        out_sl = model.decoder(feats_sl, feats_sl, feats_sl, target_size=target_size)
        
        return out_img, out_cb, out_sl
    
    def analyze(self, model, loader, config, subset_size=100):
        model.eval()
        
        all_img_cb_corr = []
        all_img_sl_corr = []
        all_cb_sl_corr = []
        
        total_batches = 0
        
        with torch.no_grad():
            for batch_idx, (imgs, cbs, sls, gts, _) in enumerate(tqdm(loader, desc='Analyzing correlation', leave=False)):
                if batch_idx * config.batch_size >= subset_size:
                    break
                
                imgs = imgs.to(config.device)
                cbs = cbs.to(config.device)
                sls = sls.to(config.device)
                
                try:
                    out_img, out_cb, out_sl = self._get_stream_outputs(model, imgs, cbs, sls, config)
                    
                    corr_img_cb = self.compute_prediction_correlation(out_img, out_cb)
                    corr_img_sl = self.compute_prediction_correlation(out_img, out_sl)
                    corr_cb_sl = self.compute_prediction_correlation(out_cb, out_sl)
                    
                    all_img_cb_corr.append(corr_img_cb)
                    all_img_sl_corr.append(corr_img_sl)
                    all_cb_sl_corr.append(corr_cb_sl)
                    total_batches += 1
                    
                except Exception as e:
                    print(f"Warning: Could not analyze batch {batch_idx}: {e}")
                    continue
                
                del imgs, cbs, sls
        
        if total_batches > 0:
            img_cb_corr_mean = np.mean(all_img_cb_corr)
            img_sl_corr_mean = np.mean(all_img_sl_corr)
            cb_sl_corr_mean = np.mean(all_cb_sl_corr)
            diversity = 1 - (img_cb_corr_mean + img_sl_corr_mean + cb_sl_corr_mean) / 3
        
            self.history['img_cb_correlation'].append(img_cb_corr_mean)
            self.history['img_sl_correlation'].append(img_sl_corr_mean)
            self.history['cb_sl_correlation'].append(cb_sl_corr_mean)
            self.history['diversity_score'].append(diversity)
        
            return {
                'img_cb_correlation': img_cb_corr_mean,
                'img_sl_correlation': img_sl_corr_mean,
                'cb_sl_correlation': cb_sl_corr_mean,
                'diversity_score': diversity,
                'is_complementary': diversity > 0.3
            }
        return None

    def print_summary(self):
        if not self.history['diversity_score']:
            print("No correlation data available")
            return
        
        print(f"\n{'='*50}")
        print(f"Triple-Stream Correlation Analysis Summary")
        print(f"{'='*50}")
        
        final_img_cb = self.history['img_cb_correlation'][-1]
        final_img_sl = self.history['img_sl_correlation'][-1]
        final_cb_sl = self.history['cb_sl_correlation'][-1]
        final_diversity = self.history['diversity_score'][-1]
        
        print(f"Final Correlations:")
        print(f"  Image-CB:   {final_img_cb:.4f}")
        print(f"  Image-SL:   {final_img_sl:.4f}")
        print(f"  CB-SL:      {final_cb_sl:.4f}")
        print(f"\nFinal Diversity Score: {final_diversity:.4f}")
        
        if final_diversity > 0.3:
            print(f"\n✓ The three streams are complementary!")
        else:
            print(f"\n✗ The three streams have low complementarity.")
        print(f"{'='*50}")