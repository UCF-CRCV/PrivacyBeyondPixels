import os
import numpy as np
import time
import torch
from torch.utils.data import DataLoader

from feature_dl import KineticsFeaturesDataset, HMDBFeaturesDataset, UCFFeaturesDataset, TSHFeaturesDataset, NTUFeaturesDataset
from model_loaders import load_fa, load_ft

import config as cfg


# Training epoch.
def train_epoch(epoch, fa_model, ft_model, criterion, optimizer, train_loader, anon, fa_trainable=False):
    print(f'Train Epoch {epoch}')
    losses = []
    ft_model.train()
    if fa_trainable:
        fa_model.train()

    for batch_idx, (features, label, _) in enumerate(train_loader):
        optimizer.zero_grad()
        features = features.cuda()
        label = label.cuda()
        if anon:
            features = fa_model(features)
        output = ft_model(features)
        loss = criterion(output, label)
        losses.append(loss.item())
        loss.backward()
        optimizer.step()
    
    print(f'Training Epoch {epoch}, Loss: {np.mean(losses):.4f}')
    return np.mean(losses)


# Validation epoch.
def val_epoch(epoch, fa_model, ft_model, criterion, test_loader, pred_dict, label_dict, mode, anon):
    print(f'\nVal Epoch {epoch}')
    ft_model.eval()
    if anon:
        fa_model.eval()
    losses = []
    predictions, gt = [], []
    vid_paths = []

    for batch_idx, (features, label, vid_path) in enumerate(test_loader):
        vid_paths.extend(vid_path)
        gt.extend(label.data.numpy())
        features = features.cuda()
        label = label.cuda()

        with torch.no_grad():
            if anon:
                features = fa_model(features)
            output = ft_model(features)
            loss = criterion(output, label)
            losses.append(loss.item())

        predictions.extend(output.softmax(1).cpu().data.numpy())

    print(f'Val Epoch {epoch}, Loss: {np.mean(losses):.4f}')

    ground_truth = np.asarray(gt)
    pred_array = np.flip(np.argsort(predictions, axis=1), axis=1) 
    c_pred = pred_array[:, 0] 

    for entry in range(len(vid_paths)):
        if str(vid_paths[entry].split('/')[-1]) not in pred_dict.keys():
            pred_dict[str(vid_paths[entry].split('/')[-1])] = []
            pred_dict[str(vid_paths[entry].split('/')[-1])].append(predictions[entry])

        else:
            pred_dict[str(vid_paths[entry].split('/')[-1])].append(predictions[entry])

    for entry in range(len(vid_paths)):
        if str(vid_paths[entry].split('/')[-1]) not in label_dict.keys():
            label_dict[str(vid_paths[entry].split('/')[-1])]= ground_truth[entry]

    correct_count = np.sum(c_pred==ground_truth)
    accuracy = float(correct_count)/len(c_pred)

    print(f'Epoch {epoch}, mode {mode} - Accuracy: {accuracy*100:.3f}%')

    return pred_dict, label_dict, accuracy, np.mean(losses)


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
    criterion = torch.nn.CrossEntropyLoss()
    ft_model.cuda()
    criterion.cuda()

    if params.anon:
        fa_model.cuda()
        if not params.fa_trainable:
            for param in fa_model.parameters():
                param.requires_grad = False
            fa_model.eval()

    optimizer = torch.optim.AdamW(ft_model.parameters(), lr=params.learning_rate)
    if params.anon and params.fa_trainable:
        optimizer.add_param_group({'params': fa_model.parameters(), 'lr': params.learning_rate})

    if params.dataset == 'k400' or params.dataset == 'k200db':
        train_dataset = KineticsFeaturesDataset(params, split='train')
        train_loader = DataLoader(train_dataset, batch_size=params.batch_size, num_workers=params.num_workers, shuffle=True, pin_memory=True)
    elif params.dataset == 'ucf101':
        train_dataset = UCFFeaturesDataset(params, split='train')
        train_loader = DataLoader(train_dataset, batch_size=params.batch_size, num_workers=params.num_workers, shuffle=True, pin_memory=True)
    elif params.dataset == 'tsh':
        train_dataset = TSHFeaturesDataset(params, split='train')
        train_loader = DataLoader(train_dataset, batch_size=params.batch_size, num_workers=params.num_workers, shuffle=True, pin_memory=True)
    elif params.dataset == 'hmdb51':
        train_dataset = HMDBFeaturesDataset(params, split='train')
        train_loader = DataLoader(train_dataset, batch_size=params.batch_size, num_workers=params.num_workers, shuffle=True, pin_memory=True)
    elif params.dataset == 'ntu':
        train_dataset = NTUFeaturesDataset(params, split='train')
        train_loader = DataLoader(train_dataset, batch_size=params.batch_size, num_workers=params.num_workers, shuffle=True, pin_memory=True)
    num_epochs = params.num_epochs
    val_array = np.arange(0, num_epochs+1, params.val_freq)
    modes = np.arange(params.num_modes)
    best_score = 0

    for epoch in range(1, num_epochs+1):
        start = time.time()
        train_epoch(epoch, fa_model, ft_model, criterion, optimizer, train_loader, params.anon, params.fa_trainable)
        if epoch in val_array:
            pred_dict, label_dict = {}, {}
            val_losses = []
            for val_iter, mode in enumerate(modes):
                if params.dataset == 'k400' or params.dataset == 'k200db':
                    test_dataset = KineticsFeaturesDataset(params, split='val', mode=mode)
                    test_loader = DataLoader(test_dataset, batch_size=params.v_batch_size, shuffle=False, num_workers=params.num_workers)
                elif params.dataset == 'hmdb51':
                    test_dataset = HMDBFeaturesDataset(params, split='test', mode=mode)
                    test_loader = DataLoader(test_dataset, batch_size=params.v_batch_size, shuffle=False, num_workers=params.num_workers)
                elif params.dataset == 'ucf101':
                    test_dataset = UCFFeaturesDataset(params, split='test', mode=mode)
                    test_loader = DataLoader(test_dataset, batch_size=params.v_batch_size, shuffle=False, num_workers=params.num_workers)
                elif params.dataset == 'tsh':
                    test_dataset = TSHFeaturesDataset(params, split='test', mode=mode)
                    test_loader = DataLoader(test_dataset, batch_size=params.v_batch_size, shuffle=False, num_workers=params.num_workers)
                elif params.dataset == 'ntu':
                    test_dataset = NTUFeaturesDataset(params, split='test', mode=mode)
                    test_loader = DataLoader(test_dataset, batch_size=params.v_batch_size, shuffle=False, num_workers=params.num_workers)

                pred_dict, label_dict, accuracy, val_loss = val_epoch(epoch, fa_model, ft_model, criterion, test_loader, pred_dict, label_dict, mode, params.anon)
                val_losses.append(val_loss)

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
            
            val_loss = np.mean(val_losses)
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
            print(f'Val loss for epoch {epoch} is {val_loss}')
            print(f'Correct Count is {correct_count} out of {len(c_pred)}')
            # writer.add_scalar('Validation Loss', val_loss, epoch)
            # writer.add_scalar('Validation Accuracy', accuracy, epoch)
            print(f'Overall Ft accuracy for epoch {epoch} is {accuracy*100:.3f}%')
            if accuracy > best_score:
                best_score = accuracy
                old_model_files = os.listdir(save_dir)
                for old_model_file in old_model_files:
                    os.remove(os.path.join(save_dir, old_model_file))
                print('++++++++++++++++++++++++++++++')
                print(f'Epoch {epoch} is the best model till now for {params.run_id}!')
                print('++++++++++++++++++++++++++++++')
                save_file_path = os.path.join(save_dir, f'model_{epoch}_acc_{accuracy*100:.4f}.pth')
                if params.anon:
                    states = {
                        'ft_model_state_dict': ft_model.state_dict(),
                        'fa_model_state_dict': fa_model.state_dict()
                    }
                else:
                    states = {
                        'ft_model_state_dict': ft_model.state_dict()
                    }
                torch.save(states, save_file_path)
            
        taken = time.time()-start
        print(f'Time taken for Epoch-{epoch} is {taken:.3f} seconds.')


if __name__ == "__main__":
    import argparse, importlib
    parser = argparse.ArgumentParser(description='train ft model')
    parser.add_argument("--params", dest='params', type=str, required=False, default='params/params_ft.py', help='params')
    args = parser.parse_args()
    if os.path.exists(args.params):
        params = importlib.import_module(args.params.replace('.py', '').replace('/', '.'))
        print(f'{args.params} is loaded as parameter file.')
    else:
        print(f'{args.params} does not exist, change to valid filename.')

    main(params)