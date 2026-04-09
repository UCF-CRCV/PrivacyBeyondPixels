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

# from multi_dataset_dl import *
from feature_dl import VPHMDBFeaturesDataset, VPUCFFeaturesDataset, VISPRFeaturesDataset
from model_loaders import load_fa, load_ft, load_fb, load_backbone
from nt_xent_original import NTXentLoss


import config as cfg


# Training epoch.
def train_epoch(epoch, fa_model, ft_model, fb_model, criterion_ft, criterion_fb, criterion_fa, optimizer_ft, optimizer_fb, optimizer_fa, scaler, train_loader, vispr_loader, anon, ssl):
    print(f'Train Epoch {epoch}')
    losses_ft, losses_fb, losses_fa = [], [], []
    ft_model.train()
    if anon:
        fa_model.train()

    for batch_idx, ((features, label, vlabels, _), (vispr_features, _, _)) in enumerate(zip(train_loader, vispr_loader)):
        optimizer_ft.zero_grad()
        if anon:
            optimizer_fa.zero_grad()
        features = features.cuda()
        label = label.cuda()
        vlabels = vlabels.cuda()
        if ssl:
            vispr_features = vispr_features.cuda()

        with autocast('cuda'):
            output_ft = ft_model(features) if not anon else ft_model(fa_model(features))
            loss_ft = criterion_ft(output_ft, label)

            if anon:
                loss_fa = criterion_fa(features, fa_model(features))
                output_fb = [fa_model(vispr_features[:, ii]) for ii in range(2)]
                criterion_fb = NTXentLoss(device='cuda', batch_size=output_fb[0].shape[0], temperature=0.1, use_cosine_similarity=True)
                loss_fb = criterion_fb(output_fb[0], output_fb[1])
        
        losses_ft.append(loss_ft.item())
        loss = loss_ft
        if anon:
            losses_fa.append(loss_fa.item())
            losses_fb.append(loss_fb.item())
            loss += params.fa_loss_weight*loss_fa
            loss -= params.fb_loss_weight*loss_fb
        scaler.scale(loss).backward()
        scaler.step(optimizer_ft)
        scaler.update()
        if anon:
            scaler.step(optimizer_fa)
    
    print(f'Training Epoch {epoch}, Ft Loss: {np.mean(losses_ft):.4f}, Fb Loss: {np.mean(losses_fb):.4f}, Fa Loss: {np.mean(losses_fa):.4f}')
    return np.mean(losses_ft), np.mean(losses_fb), np.mean(losses_fa)


# Validation epoch.
def val_epoch(epoch, fa_model, ft_model, criterion_ft, test_loader, pred_dict, label_dict, mode, anon):
    print(f'\nVal Epoch {epoch}')
    ft_model.eval()
    losses_ft = [] 
    predictions_ft, gt_ft = [], []
    vid_paths = []

    for batch_idx, (features, label, _, vid_path) in enumerate(test_loader):
        vid_paths.extend(vid_path)
        gt_ft.extend(label.data.numpy())
        features = features.cuda()
        label = label.cuda()

        with torch.no_grad():
            if anon:
                features = fa_model(features)
            output_ft = ft_model(features)
            loss_ft = criterion_ft(output_ft, label)
            losses_ft.append(loss_ft.item())

        predictions_ft.extend(output_ft.softmax(1).cpu().data.numpy())

    print(f'Val Epoch {epoch}, Ft Loss: {np.mean(losses_ft):.4f}')

    ground_truth = np.asarray(gt_ft)
    pred_array = np.flip(np.argsort(predictions_ft, axis=1), axis=1) 
    c_pred = pred_array[:, 0]

    for entry in range(len(vid_paths)):
        if str(vid_paths[entry].split('/')[-1]) not in pred_dict.keys():
            pred_dict[str(vid_paths[entry].split('/')[-1])] = []
            pred_dict[str(vid_paths[entry].split('/')[-1])].append(predictions_ft[entry])

        else:
            pred_dict[str(vid_paths[entry].split('/')[-1])].append(predictions_ft[entry])

    for entry in range(len(vid_paths)):
        if str(vid_paths[entry].split('/')[-1]) not in label_dict.keys():
            label_dict[str(vid_paths[entry].split('/')[-1])]= ground_truth[entry]

    correct_count = np.sum(c_pred==ground_truth)
    accuracy = float(correct_count)/len(c_pred)

    print(f'Epoch {epoch}, mode {mode} - Accuracy: {accuracy*100:.3f}%')
    return pred_dict, label_dict, accuracy, np.mean(losses_ft)


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

    ft_model = load_ft(params)
    fb_model = load_fb(final_embedding_size=128 if params.ssl else params.num_pa)

    criterion_ft = torch.nn.CrossEntropyLoss()
    criterion_fb = torch.nn.BCEWithLogitsLoss()
    criterion_fa = torch.nn.MSELoss()
    scaler = GradScaler('cuda')
    
    if params.anon:
        fa_model.eval()
        fa_model.cuda()

    ft_model.cuda()
    fb_model.cuda()
    criterion_ft.cuda()
    criterion_fb.cuda()
    criterion_fa.cuda()
    optimizer_ft = torch.optim.AdamW(ft_model.parameters(), lr=params.learning_rate)
    optimizer_fb = torch.optim.AdamW(fb_model.parameters(), lr=params.learning_rate)
    optimizer_fa = torch.optim.AdamW(fa_model.parameters(), lr=params.learning_rate) if params.anon else None

    train_dataset = VPUCFFeaturesDataset(params, split='train', shuffle=True) if params.dataset == 'ucf101' else VPHMDBFeaturesDataset(params, split='train', shuffle=True)
    train_loader = DataLoader(train_dataset, batch_size=params.batch_size, shuffle=True, num_workers=params.num_workers)
    vispr_dataset = VISPRFeaturesDataset(ssl=True, split='train', shuffle=True, model=params.ft_arch)
    vispr_loader = DataLoader(vispr_dataset, batch_size=params.batch_size, num_workers=params.num_workers, shuffle=True, pin_memory=False)

    num_epochs = params.num_epochs
    val_array = np.arange(0, num_epochs+1, params.val_freq)
    modes = np.arange(params.num_modes)
    best_score = 0
    map_path = None
    for epoch in range(1, num_epochs+1):
        start = time.time()
        train_epoch(epoch, fa_model, ft_model, fb_model, criterion_ft, criterion_fb, criterion_fa, optimizer_ft, optimizer_fb, optimizer_fa, scaler, train_loader, vispr_loader, params.anon, params.ssl)
        if epoch in val_array:
            pred_dict, label_dict = {}, {}
            val_losses_ft = []
            for val_iter, mode in enumerate(modes):
                test_dataset = VPUCFFeaturesDataset(params, split='test', mode=mode, shuffle=False) if params.dataset == 'ucf101' else VPHMDBFeaturesDataset(params, split='test', mode=mode, shuffle=False)
                test_loader = DataLoader(test_dataset, batch_size=params.v_batch_size, shuffle=False, num_workers=params.num_workers)
                pred_dict, label_dict, accuracy, val_loss_ft = val_epoch(epoch, fa_model, ft_model, criterion_ft, test_loader, pred_dict, label_dict, mode, params.anon)
                val_losses_ft.append(val_loss_ft)

                predictions = np.zeros((len(list(pred_dict.keys())), params.num_classes))
                ground_truth = []
                for entry, key in enumerate(pred_dict.keys()):
                    predictions[entry] = np.mean(pred_dict[key], axis=0)

                for key in label_dict.keys():
                    ground_truth.append(label_dict[key])

                pred_array = np.flip(np.argsort(predictions, axis=1), axis=1)  # Prediction with the most confidence is the first element here.
                c_pred = pred_array[:, 0]

                correct_count = np.sum(c_pred==ground_truth)
                accuracy_all = float(correct_count)/len(c_pred)
                print(f'Running Avg Accuracy for epoch {epoch}, mode {modes[val_iter]} is {accuracy_all*100:.3f}%')
            
            val_loss_ft = np.mean(val_losses_ft)
            predictions = np.zeros((len(list(pred_dict.keys())), params.num_classes))
            ground_truth = []

            for entry, key in enumerate(pred_dict.keys()):
                predictions[entry] = np.mean(pred_dict[key], axis=0)

            for key in label_dict.keys():
                ground_truth.append(label_dict[key])

            pred_array = np.flip(np.argsort(predictions, axis=1), axis=1)  # Prediction with the most confidence is the first element here.
            c_pred = pred_array[:,0]

            correct_count = np.sum(c_pred==ground_truth)
            accuracy = float(correct_count)/len(c_pred)
            print(f'Ft Val loss for epoch {epoch} is {val_loss_ft:.4f}')
            print(f'Correct Count is {correct_count} out of {len(c_pred)}')
            print(f'Overall Ft accuracy for epoch {epoch} is {accuracy*100:.3f}%')

            if accuracy > best_score and not params.anon:
                best_score = accuracy
                old_model_files = os.listdir(save_dir)
                for old_model_file in old_model_files:
                    if old_model_file == map_path:
                        continue
                    os.remove(os.path.join(save_dir, old_model_file))
                print('++++++++++++++++++++++++++++++')
                print(f'Epoch {epoch} is the best model till now for {params.run_id}!')
                print('++++++++++++++++++++++++++++++')
                save_file_path = os.path.join(save_dir, f'model_{epoch}_acc_{accuracy*100:.4f}.pth')
                states = {
                    'ft_model_state_dict': ft_model.state_dict(),
                }
                torch.save(states, save_file_path)
            elif params.anon:
                save_file_path = os.path.join(save_dir, f'model_{epoch}_acc_{accuracy*100:.4f}.pth')
                states = {
                    'ft_model_state_dict': ft_model.state_dict(),
                    'fa_model_state_dict': fa_model.state_dict(),
                }
                torch.save(states, save_file_path)
            torch.cuda.empty_cache()
            
        taken = time.time()-start
        print(f'Time taken for Epoch-{epoch} is {taken:.3f} seconds.')


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