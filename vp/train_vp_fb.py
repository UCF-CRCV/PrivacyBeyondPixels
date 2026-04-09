import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import copy
import numpy as np
from sklearn.metrics import average_precision_score, precision_recall_fscore_support
import time
import torch
from torch.utils.data import DataLoader
from torch.amp import autocast, GradScaler
import warnings
warnings.filterwarnings("ignore")

# from multi_dataset_dl import *
from feature_dl import VPHMDBFeaturesDataset, VPUCFFeaturesDataset, VISPRFeaturesDataset
from model_loaders import load_fa, load_ft, load_fb, load_backbone
from nt_xent_original import NTXentLoss


import config as cfg


# Training epoch.
def train_epoch(epoch, fa_model, fb_model, criterion_fb, optimizer_fb, scaler, train_loader, anon):
    # print(f'Train Epoch {epoch}')
    losses_fb = []
    fb_model.train()

    for batch_idx, (features, label, vlabels, _) in enumerate(train_loader):
        optimizer_fb.zero_grad()
        features = features.cuda()
        label = label.cuda()
        vlabels = vlabels.cuda()

        with autocast('cuda'):
            if anon:
                with torch.no_grad():
                    features = fa_model(features)
            output_fb = fb_model(features)
            loss_fb = criterion_fb(output_fb, vlabels)
        
        losses_fb.append(loss_fb.item())
        scaler.scale(loss_fb).backward()
        scaler.step(optimizer_fb)
        scaler.update()
    
    # print(f'Training Epoch {epoch}, Fb Loss: {np.mean(losses_fb):.4f}')
    return np.mean(losses_fb)


# Validation epoch.
def val_epoch(epoch, fa_model, fb_model, criterion_fb, test_loader, mode, anon):
    # print(f'\nVal Epoch {epoch}')
    fb_model.eval()
    losses_fb = [] 
    predictions_fb, gt_fb = [], []

    for batch_idx, (features, label, vlabels, _) in enumerate(test_loader):
        gt_fb.extend(vlabels.data.numpy())
        features = features.cuda()
        label = label.cuda()
        vlabels = vlabels.cuda()

        with torch.no_grad():
            if anon:
                features = fa_model(features)
            output_fb = fb_model(features)
            loss_fb = criterion_fb(output_fb, vlabels)
            losses_fb.append(loss_fb.item())

        predictions_fb.extend(output_fb.cpu().data.numpy())

    # print(f'Val Epoch {epoch}, Fb Loss: {np.mean(losses_fb):.4f}')

    ground_truth = np.asarray(gt_fb)
    predictions = np.asarray(predictions_fb)

    prec, recall, f1, _ = precision_recall_fscore_support(ground_truth, (np.array(predictions) > 0.5).astype(int))
    predictions = np.asarray(predictions)

    ap = average_precision_score(ground_truth, predictions, average=None)
    
    # print(f'Epoch {epoch}, mode {mode}')
    # print(f'Macro f1 is {np.mean(f1)}')
    # print(f'Macro prec is {np.mean(prec)}')
    # print(f'Macro recall is {np.mean(recall)}')
    # print(f'Classwise AP is {ap}')
    # print(f'Macro AP is {np.mean(ap)}')

    return np.mean(losses_fb), np.mean(ap)


def main(params):
    # Print relevant parameters.
    for k, v in params.__dict__.items():
        if '__' not in k:
            print(f'{k} : {v}')

    save_dir = os.path.join(cfg.saved_models_dir, params.run_id)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # Set up GPU.
    torch.set_float32_matmul_precision('high')
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True

    fa_model = None
    if params.anon:
        fa_model = load_fa(params)

    fb_model = load_fb(final_embedding_size=params.num_pa)

    criterion_fb = torch.nn.BCEWithLogitsLoss()
    scaler = GradScaler('cuda')
    
    if params.anon:
        fa_model.eval()
        for param in fa_model.parameters():
            param.requires_grad = False
        fa_model.cuda()

    fb_model.cuda()
    criterion_fb.cuda()
    optimizer_fb = torch.optim.AdamW(fb_model.parameters(), lr=params.learning_rate)

    train_dataset = VPUCFFeaturesDataset(params, split='train', shuffle=True) if params.dataset == 'ucf101' else VPHMDBFeaturesDataset(params, split='train', shuffle=True)
    train_loader = DataLoader(train_dataset, batch_size=params.batch_size, shuffle=True, num_workers=params.num_workers)

    num_epochs = params.num_epochs
    val_array = np.arange(0, num_epochs+1, params.val_freq)
    modes = np.arange(params.num_modes)
    best_score = 0
    best_map = 0
    map_path = None
    for epoch in range(1, num_epochs+1):
        start = time.time()
        train_epoch(epoch, fa_model, fb_model, criterion_fb, optimizer_fb, scaler, train_loader, params.anon)
        if epoch in val_array:
            val_losses_fb = []
            epoch_map = 0
            for val_iter, mode in enumerate(modes):
                test_dataset = VPUCFFeaturesDataset(params, split='test', mode=mode, shuffle=False) if params.dataset == 'ucf101' else VPHMDBFeaturesDataset(params, split='test', mode=mode, shuffle=False)
                test_loader = DataLoader(test_dataset, batch_size=params.v_batch_size, shuffle=False, num_workers=params.num_workers)
                val_loss_fb, m_ap = val_epoch(epoch, fa_model, fb_model, criterion_fb, test_loader, mode, params.anon)
                val_losses_fb.append(val_loss_fb)
                if epoch_map < m_ap:
                    epoch_map = m_ap

            # val_loss_fb = np.mean(val_losses_fb)
            # print(f'Fb Val loss for epoch {epoch} is {val_loss_fb:.4f}')

            if epoch_map > best_map:
                best_map = epoch_map
                print(f'Epoch {epoch} -- new best cMAP: {best_map*100:.2f}')
                # print('++++++++++++++++++++++++++++++')
                # print(f'Epoch {epoch} is the best model till now for {params.run_id}!')
                # print('++++++++++++++++++++++++++++++')
                map_path = f'model_{epoch}_map_{best_map*100:.4f}.pth'
                save_file_path = os.path.join(save_dir, map_path)
                states = {
                    'fb_model_state_dict': fb_model.state_dict(),
                    'fa_model_state_dict': fa_model.state_dict()
                } if params.anon else {
                    'fb_model_state_dict': fb_model.state_dict()
                }
                torch.save(states, save_file_path)
            torch.cuda.empty_cache()

        taken = time.time()-start
        # print(f'Time taken for Epoch-{epoch} is {taken:.3f} seconds.')
    print(f'Best cMAP for {params.run_id} is {best_map*100:.2f}')   


if __name__ == "__main__":
    import argparse, importlib
    parser = argparse.ArgumentParser(description='train vp model')
    parser.add_argument("--params", dest='params', type=str, required=False, default='vp/params_vp.py', help='params')
    args = parser.parse_args()
    if os.path.exists(args.params):
        params = importlib.import_module(args.params.replace('.py', '').replace('/', '.'))
        print(f'{args.params} is loaded as parameter file.')
    else:
        print(f'{args.params} does not exist, change to valid filename.')

    main(params)