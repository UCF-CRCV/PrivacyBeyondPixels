import os
import numpy as np
from sklearn.metrics import precision_recall_fscore_support, average_precision_score
import time
import torch
from torch.utils.data import DataLoader
import warnings
warnings.filterwarnings("ignore")

from feature_dl import VISPRFeaturesDataset
from model_loaders import load_fa, load_fb

import config as cfg


# Training epoch.
def train_epoch(epoch, fa_model, fb_model, criterion, optimizer, train_loader, anon):
    # print(f'Train Epoch {epoch}')
    losses = []
    fb_model.train()

    for batch_idx, (features, label, _) in enumerate(train_loader):
        optimizer.zero_grad()

        features = features.cuda()
        label = label.cuda()
        if anon:
            output = fb_model(fa_model(features))
        else:
            output = fb_model(features)
        loss = criterion(output, label)
        losses.append(loss.item())
        loss.backward()
        optimizer.step()
    
    # print(f'Training Epoch {epoch}, Loss: {np.mean(losses):.4f}')
    return np.mean(losses)


# Validation epoch.
def val_epoch(epoch, fa_model, fb_model, criterion, test_loader, anon):
    # print(f'Val Epoch {epoch}')
    fb_model.eval()
    losses = []
    predictions, gt = [], []

    for batch_idx, (features, label, _) in enumerate(test_loader):
        gt.extend(label.data.numpy())
        features = features.cuda()
        label = label.cuda()

        with torch.no_grad():
            if anon:
                output = fb_model(fa_model(features))
            else:
                output = fb_model(features)
            loss = criterion(output, label)
            losses.append(loss.item())

        predictions.extend(output.cpu().data.numpy())

    # print(f'Val Epoch {epoch}, Loss: {np.mean(losses):.4f}')
    ground_truth = np.asarray(gt)
    predictions = np.asarray(predictions)

    # Print classwise proportions of ground truth.
    # print(f'Classwise proportions of ground truth: {np.sum(ground_truth, axis=0)/len(ground_truth)}')

    prec, recall, f1, _ = precision_recall_fscore_support(ground_truth, (np.array(predictions) > 0.5).astype(int))
    predictions = np.asarray(predictions)
    # try:
    #     print(f'GT shape before putting in ap: {ground_truth.shape}')
    #     print(f'pred shape before putting in ap: {predictions.shape}')
    # except:
    #     print(f'GT len before putting in ap: {len(ground_truth)}')
    #     print(f'pred len before putting in ap: {len(predictions)}')

    ap = average_precision_score(ground_truth, predictions, average=None)
    
    # print(f'Macro f1 is {np.mean(f1)}')
    # print(f'Macro prec is {np.mean(prec)}')
    # print(f'Macro recall is {np.mean(recall)}')
    # print(f'Classwise AP is {ap}')
    # print(f'Macro AP is {np.mean(ap)*100:.2f}')
    return np.mean(ap)


def main(params):
    # Print relevant parameters.
    for k, v in params.__dict__.items():
        if '__' not in k:
            print(f'{k} : {v}')

    save_dir = os.path.join(cfg.saved_models_dir, params.run_id)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    fa_model = None

    if params.anon:
        fa_model = load_fa(params)
        fa_model.eval()
        fa_model.cuda()
    fb_model = load_fb(initial_embedding_size=params.feature_dim, final_embedding_size=params.num_classes)
    criterion = torch.nn.BCEWithLogitsLoss()

    fb_model.cuda()
    criterion.cuda()

    optimizer = torch.optim.AdamW(fb_model.parameters(), lr=1e-3)

    train_dataset = VISPRFeaturesDataset(split='train', model=params.ft_arch)
    train_loader = DataLoader(train_dataset, batch_size=params.batch_size, shuffle=True)
    test_dataset = VISPRFeaturesDataset(split='test', model=params.ft_arch)
    test_loader = DataLoader(test_dataset, batch_size=params.v_batch_size, shuffle=False)

    num_epochs = params.num_epochs
    val_array = np.arange(0, num_epochs+1, 5)

    best_score = 0

    for epoch in range(1, num_epochs+1):
        start = time.time()
        train_epoch(epoch, fa_model, fb_model, criterion, optimizer, train_loader, params.anon)
        if epoch in val_array:
            macro_ap = val_epoch(epoch, fa_model, fb_model, criterion, test_loader, params.anon)
            if macro_ap > best_score:
                best_score = macro_ap
                old_model_files = os.listdir(save_dir)
                for old_model_file in old_model_files:
                    os.remove(os.path.join(save_dir, old_model_file))
                print(f'Epoch {epoch} -- new best cMAP: {best_score*100:.2f}')
                # print('++++++++++++++++++++++++++++++')
                # print(f'Epoch {epoch} is the best model till now for {params.run_id}!')
                # print('++++++++++++++++++++++++++++++')
                save_file_path = os.path.join(save_dir, f'model_{epoch}_map_{macro_ap*100:.2f}.pth')
                states = {
                    'fb_model_state_dict': fb_model.state_dict(),
                    'fa_model_state_dict': fa_model.state_dict()
                } if params.anon else {
                    'fb_model_state_dict': fb_model.state_dict()
                }
                torch.save(states, save_file_path)
            
        # taken = time.time()-start
        # print(f'Time taken for Epoch-{epoch} is {taken:.3f} seconds.')

    print(f'Best score for {params.run_id} is {best_score*100:.2f}')


if __name__ == "__main__":
    import argparse, importlib
    parser = argparse.ArgumentParser(description='train fb model')
    parser.add_argument("--params", dest='params', type=str, required=False, default='params/params_fb.py', help='params')
    args = parser.parse_args()
    if os.path.exists(args.params):
        params = importlib.import_module(args.params.replace('.py', '').replace('/', '.'))
        print(f'{args.params} is loaded as parameter file.')
    else:
        print(f'{args.params} does not exist, change to valid filename.')

    main(params)
