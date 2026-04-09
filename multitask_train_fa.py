import numpy as np
import os.path
from sklearn.metrics import auc, roc_curve, precision_recall_curve
import time
import torch
from torch.utils.data import DataLoader
import wandb

from feature_dl import KineticsFeaturesDataset, HMDBFeaturesDataset, UCFFeaturesDataset, NTUFeaturesDataset, TSHFeaturesDataset
from mgfn.datasets.dataset import Dataset
from mgfn.train import mgfn_loss, sparsity, smooth
from mgfn import option
# from tridet.tridet_utils import ANETdetection, load_model_tridet
from tridet.thumos14_dl import load_thumos
from tridet.utils import ANETdetection
from model_loaders import load_fa, load_ft, load_model_tridet, load_model_mgfn
from nt_xent_original import NTXentLoss

import config as cfg


# Training loop for fa_model.
def train_epoch(params, epoch, fa_model, ft_model_ar, ft_model_tad, ft_model_tad_ema, ft_model_ad, optimizer_fa, optimizer_ft_ar, optimizer_ft_tad, scheduler_ft_tad, optimizer_ad, train_loader, criterion_ft_ar, criterion_fa):
    # print(f'Train at epoch {epoch}') 
    losses_fa, losses_fb, losses_ft_ar, losses_ft_tad, losses_ft_ad = [], [], [], [], []
    
    # Group ad datasets together.
    train_loader[2] = zip(*train_loader[2])

    for batch_idx, (ar_video_list, tad_video_list, ad_video_list) in enumerate(zip(*train_loader)):
        optimizer_fa.zero_grad()
        features, labels, _, fb_features = ar_video_list
        fb_features = fb_features.cuda()

        if params.ft_ar_loss_weight > 0:
            features = features.cuda()
            labels = labels.cuda()
            optimizer_ft_ar.zero_grad()
            ft_model_ar.train()
        else:
            features = fb_features

        if params.ft_tad_loss_weight > 0:
            for video in tad_video_list:
                video['feats'] = video['feats'].cuda()
            optimizer_ft_tad.zero_grad()
            ft_model_tad.train()

        if params.ft_ad_loss_weight > 0:
            (ninput, nlabel), (ainput, alabel) = ad_video_list
            input_ad = torch.cat((ninput, ainput), 0)
            # Shave off feature norm.
            input_ad = input_ad[:, :, :, :-1].cuda()
            bs = params.batch_size_ad
            nlabel = nlabel[0:bs]
            alabel = alabel[0:bs]
            optimizer_ad.zero_grad()
            ft_model_ad.train()

        # Train all models.
        fa_model.train()

        # Privacy task.
        output_fb = [fa_model(fb_features[:, ii]) for ii in range(2)]
        # Contrastive loss function for SSL.
        criterion_fb = NTXentLoss(device='cuda', batch_size=output_fb[0].shape[0], temperature=0.1, use_cosine_similarity=True)
        loss_fb = criterion_fb(output_fb[0], output_fb[1])
        
        # Action Recognition task.
        if params.ft_ar_loss_weight > 0:
            output_ft_ar = ft_model_ar(fa_model(features))
            # Cross entropy loss function for classification.
            loss_ft_ar = criterion_ft_ar(output_ft_ar, labels)
        else:
            loss_ft_ar = torch.tensor(0.0).cuda()
            
        # Temporal Action Detection task.
        if params.ft_tad_loss_weight > 0:
            for idx, video in enumerate(tad_video_list):
                tad_video_list[idx]['feats'] = fa_model(video['feats'].permute(1, 0)).permute(1, 0)
            
            losses_tad = ft_model_tad(tad_video_list)
            loss_ft_tad = losses_tad['final_loss']
        else:
            loss_ft_tad = torch.tensor(0.0).cuda()

        # Anomaly Detection task.
        if params.ft_ad_loss_weight > 0:
            ad_input_shape = input_ad.shape
            input_ad = fa_model(input_ad.view(-1, ad_input_shape[-1])).view(ad_input_shape)
            feat_norms = []
            for feature in input_ad:
                feature = feature.squeeze()
                feat_norms.append(torch.norm(feature, p=2, dim=1))

            feat_norms = torch.stack(feat_norms).view(ad_input_shape[0], 1, 32, 1)
            # Concat with input_ad.
            input_ad = torch.cat((input_ad, feat_norms), -1)

            score_abnormal, score_normal, abn_feamagnitude, nor_feamagnitude, scores = ft_model_ad(input_ad)
            scores = scores.view(bs*32*2, -1)
            scores = scores.squeeze()
            abn_scores = scores[bs*32:]

            loss_criterion = mgfn_loss(0.0001)
            loss_sparse = sparsity(abn_scores, bs, 8e-3)
            loss_smooth = smooth(abn_scores, 8e-4)

            loss_ft_ad = loss_criterion(score_normal, score_abnormal, nlabel, alabel, nor_feamagnitude, abn_feamagnitude) + loss_smooth + loss_sparse
        else:
            loss_ft_ad = torch.tensor(0.0).cuda()

        loss_fa = criterion_fa(features, fa_model(features))
        loss_fa = params.fa_loss_weight*loss_fa + params.ft_ar_loss_weight*loss_ft_ar + params.ft_tad_loss_weight*loss_ft_tad + params.ft_ad_loss_weight*loss_ft_ad - params.fb_loss_weight*loss_fb

        losses_fa.append(loss_fa.item())
        losses_ft_ar.append(loss_ft_ar.item())
        losses_ft_tad.append(loss_ft_tad.item())
        losses_ft_ad.append(loss_ft_ad.item())
        losses_fb.append(loss_fb.item())

        loss_fa.backward()
        optimizer_fa.step()
        
        if params.ft_ar_loss_weight > 0:
            optimizer_ft_ar.step()
        if params.ft_tad_loss_weight > 0:
            torch.nn.utils.clip_grad_norm_(ft_model_tad.parameters(), 1.0)
            optimizer_ft_tad.step()
            scheduler_ft_tad.step()
            if ft_model_tad_ema is not None:
                ft_model_tad_ema.update(ft_model_tad)
        if params.ft_ad_loss_weight > 0:
            optimizer_ad.step()

    print(f'Training Epoch: {epoch}, loss_fa: {np.mean(losses_fa):.4f}, loss_fb: {np.mean(losses_fb):.4f}, loss_ft_ar: {np.mean(losses_ft_ar):.4f}, loss_ft_tad: {np.mean(losses_ft_tad):.4f}, loss_ft_ad: {np.mean(losses_ft_ad):.4f}', flush=True)
    if params.wandb:
        wandb.log({'loss_fa': np.mean(losses_fa), 'loss_fb': np.mean(losses_fb), 'loss_ft_ar': np.mean(losses_ft_ar), 'loss_ft_tad': np.mean(losses_ft_tad), 'loss_ft_ad': np.mean(losses_ft_ad)}, step=epoch)


# Validation loop for action recognition model.
def val_epoch_action_recognition(epoch, fa_model, ft_model, test_loader, criterion_ft, pred_dict, label_dict, mode):
    # print(f'Validation at epoch {epoch}.')
    fa_model.eval()
    ft_model.eval()
    losses = []
    predictions, ground_truth = [], []
    vid_paths = []

    for batch_idx, (features, labels, vid_path) in enumerate(test_loader):
        ground_truth.extend(labels)
        vid_paths.extend(vid_path)

        features = features.cuda()
        labels = labels.cuda()

        with torch.no_grad():
            output_ft = ft_model(fa_model(features))
            loss = criterion_ft(output_ft, labels)
            losses.append(loss.item())
            predictions.extend(output_ft.softmax(1).cpu().data.numpy())

    ground_truth = np.asarray(ground_truth)
    pred_array = np.flip(np.argsort(predictions,axis=1),axis=1) 
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

    # print(f'Epoch {epoch}, mode {mode} - Accuracy: {accuracy*100:.3f}%')

    return pred_dict, label_dict, accuracy, np.mean(losses)


# Validation loop for action detection model.
def val_epoch_action_detection(fa_model, tad_model, test_loader, evaluator):
    # print(f'Validation at epoch {epoch}.')
    fa_model.eval()
    tad_model.eval()

    results = {
        'video-id': [],
        't-start': [],
        't-end': [],
        'label': [],
        'score': []
    }

    for batch_idx, video_list in enumerate(test_loader):
        # forward the model (wo. grad)
        with torch.no_grad():
            for idx, video in enumerate(video_list):
                video_list[idx]['feats'] = fa_model(video['feats'].permute(1, 0).cuda()).permute(1, 0)
            output = tad_model(video_list)

            # upack the results into ANet format
            num_vids = len(output)
            for vid_idx in range(num_vids):
                if output[vid_idx]['segments'].shape[0] > 0:
                    results['video-id'].extend(
                        [output[vid_idx]['video_id']] *
                        output[vid_idx]['segments'].shape[0]
                    )
                    results['t-start'].append(output[vid_idx]['segments'][:, 0])
                    results['t-end'].append(output[vid_idx]['segments'][:, 1])
                    results['label'].append(output[vid_idx]['labels'])
                    results['score'].append(output[vid_idx]['scores'])

    # gather all stats and evaluate
    results['t-start'] = torch.cat(results['t-start']).numpy()
    results['t-end'] = torch.cat(results['t-end']).numpy()
    results['label'] = torch.cat(results['label']).numpy()
    results['score'] = torch.cat(results['score']).numpy()

    # call the evaluator
    _, mAP = evaluator.evaluate(results, verbose=False)

    return mAP

# Validation loop for anomaly detection model.
def val_epoch_anomaly_detection(fa_model, ad_model, test_loader):
    # print(f'Validation at epoch {epoch}.')
    fa_model.eval()
    ad_model.eval()

    with torch.no_grad():
        pred = torch.zeros(0).cuda()
        featurelen = []

        for i, inputs in enumerate(test_loader):
            input = inputs[0].cuda()
            input = input[:, :, :, :-1].cuda()
            input_shape = input.shape
            input = fa_model(input.view(-1, input_shape[-1])).view(input_shape)
            feat_norm = torch.norm(input.squeeze(), p=2, dim=1).view(input_shape[0], input.shape[1], 1, 1)
            # Concat with input_ad.
            input = torch.cat((input, feat_norm), -1)
            input = input.permute(0, 2, 1, 3)

            _, _, _, _, logits = ad_model(input)
            logits = torch.squeeze(logits, 1)
            logits = torch.mean(logits, 0)
            sig = logits
            featurelen.append(len(sig))
            pred = torch.cat((pred, sig))

        gt = np.load('mgfn/data/gt-ucf.npy')
        pred = list(pred.cpu().detach().numpy())
        pred = np.repeat(np.array(pred), 16)
        ratio = float(len(list(gt))) / float(len(pred))
        # In case size mismatch btwn predictions and gt.
        if ratio == 1.0:
            final_pred = pred
        else:
            print(f'Ground truth not exact shape: {ratio}')
            final_pred = np.zeros_like(gt, dtype='float32')
            for i in range(len(pred)):
                b = int(i * ratio + 0.5)
                e = int((i + 1) * ratio + 0.5)
                final_pred[b:e] = pred[i]

        fpr, tpr, _ = roc_curve(list(gt), list(final_pred), drop_intermediate=True)
        rec_auc = auc(fpr, tpr)
        precision, recall, th = precision_recall_curve(list(gt), list(final_pred))
        pr_auc = auc(recall, precision)

    return rec_auc, pr_auc


def main(params):
    # Print relevant parameters.
    for k, v in params.__dict__.items():
        if '__' not in k:
            print(f'{k} : {v}')

    save_dir = os.path.join(cfg.saved_models_dir, params.run_id)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # Init wandb.
    if params.wandb:
        os.environ["WANDB_START_METHOD"] = "thread"
        wandb.init(
        project='feature_anonymization',
        name=params.run_id,
        config={
            'learning_rate_fa': params.learning_rate_fa,
            'learning_rate_fb': params.learning_rate_fb,
            'learning_rate_ft': params.learning_rate_ft,
            'fb_loss_weight': params.fb_loss_weight,
            'ft_loss_weight': params.ft_loss_weight,
            'batch_size': params.batch_size,
            'epochs': params.num_epochs,
        })

    # Load datasets.
    if params.ar_dataset == 'k400' or params.ar_dataset == 'k200db':
        ar_train_dataset = KineticsFeaturesDataset(params, split='train', return_fb_frames=True)
    elif params.ar_dataset == 'ucf101':
        ar_train_dataset = UCFFeaturesDataset(params, split='train', return_fb_frames=True)
    elif params.ar_dataset == 'tsh':
        ar_train_dataset = TSHFeaturesDataset(params, split='train', return_fb_frames=True)
    elif params.ar_dataset == 'ntu':
        ar_train_dataset = NTUFeaturesDataset(params, split='train', return_fb_frames=True)
    elif params.ar_dataset == 'hmdb51':
        ar_train_dataset = HMDBFeaturesDataset(params, split='train', return_fb_frames=True)
    else:
        print('Dataset not implemented.')
        raise NotImplementedError
    ar_train_loader = DataLoader(ar_train_dataset, batch_size=params.batch_size_ar, num_workers=params.num_workers, shuffle=True, pin_memory=True)
    tad_train_loader = load_thumos(f'tridet/thumos_{params.ft_arch}.yaml', 'train')
    ad_args = option.parse_args()
    ad_args.fa_model = 'mae' if params.ft_arch == 'videoMAE' else params.ft_arch
    ad_args.feature_size = params.feature_dim
    ad_args.batch_size = params.batch_size_ad
    ad_train_nloader = DataLoader(Dataset(ad_args, test_mode=False, is_normal=True), batch_size=params.batch_size_ad, shuffle=False, num_workers=params.num_workers, pin_memory=False, drop_last=True)
    ad_train_aloader = DataLoader(Dataset(ad_args, test_mode=False, is_normal=False), batch_size=params.batch_size_ad, shuffle=False, num_workers=params.num_workers, pin_memory=False, drop_last=True)
    ad_train_loader = [ad_train_nloader, ad_train_aloader]

    # Load models.
    fa_model = load_fa(params)
    ft_model_ar = load_ft(params)
    ft_model_tad, ft_model_tad_ema, optimizer_ft_tad, scheduler_ft_tad = load_model_tridet(params.saved_ft_tad_model, tad_train_loader, params.ft_arch)
    ft_model_ad = load_model_mgfn(params.saved_ft_ad_model)

    criterion_ft_ar = torch.nn.CrossEntropyLoss()
    criterion_fa = torch.nn.MSELoss()

    fa_model.cuda()
    ft_model_ar.cuda()
    ft_model_ad.cuda()
    criterion_ft_ar.cuda()
    criterion_fa.cuda()

    optimizer_fa = torch.optim.AdamW(fa_model.parameters(), lr=params.learning_rate_fa)
    optimizer_ft_ar = torch.optim.AdamW(ft_model_ar.parameters(), lr=params.learning_rate_ft)
    optimizer_ft_ad = torch.optim.AdamW(ft_model_ad.parameters(), lr=params.learning_rate_ft_ad, weight_decay=0.0005)

    num_epochs = params.num_epochs
    modes = np.arange(params.num_modes)
    val_array = np.arange(0, num_epochs, params.val_freq)
    val_array = np.append(val_array, num_epochs)
    best_result_ar, best_result_tad, best_result_ad = -1, -1, -1
    best_epoch_ar, best_epoch_tad, best_epoch_ad = -1, -1, -1
    best_ar_fmap, best_ar_auc = -1, -1
    best_tad_acc, best_tad_auc = -1, -1
    best_ad_acc, best_ad_fmap = -1, -1
    accuracy, fmap, auc = -1, -1, -1

    for epoch in range(1, num_epochs+1):
        print(f'Epoch {epoch} started')
        start = time.time()
        train_loader = [ar_train_loader, tad_train_loader, ad_train_loader]

        train_epoch(params, epoch, fa_model, ft_model_ar, ft_model_tad, ft_model_tad_ema, ft_model_ad, optimizer_fa, optimizer_ft_ar, optimizer_ft_tad, scheduler_ft_tad, optimizer_ft_ad, train_loader, criterion_ft_ar, criterion_fa)

        if epoch in val_array:
            print(f'Validation at epoch {epoch}')
            if params.ft_ar_loss_weight > 0:
                pred_dict, label_dict = {}, {}
                val_losses = []
                for val_iter, mode in enumerate(modes):
                    if params.ar_dataset == 'k400' or params.ar_dataset == 'k200db':
                        test_dataset_ar = KineticsFeaturesDataset(params, split='test', mode=mode)
                    elif params.ar_dataset == 'ucf101':
                        test_dataset_ar = UCFFeaturesDataset(params, split='test', mode=mode)
                    elif params.ar_dataset == 'tsh':
                        test_dataset_ar = TSHFeaturesDataset(params, split='test', mode=mode)
                    elif params.ar_dataset == 'ntu':
                        test_dataset_ar = NTUFeaturesDataset(params, split='test', mode=mode)
                    elif params.ar_dataset == 'hmdb51':
                        test_dataset_ar = HMDBFeaturesDataset(params, split='test', mode=mode)
                    else:
                        print('Dataset not implemented.')
                        raise NotImplementedError
                    test_loader_ar = DataLoader(test_dataset_ar, batch_size=params.v_batch_size, num_workers=params.num_workers, shuffle=False)

                    pred_dict, label_dict, accuracy, val_loss = val_epoch_action_recognition(epoch, fa_model, ft_model_ar, test_loader_ar, criterion_ft_ar, pred_dict, label_dict, mode)
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
                    # print(f'Running Avg Accuracy for epoch {epoch}, mode {modes[val_iter]} is {accuracy_all*100:.3f}%')

                # End of epoch evaluation.
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
                # print(f'Correct Count is {correct_count} out of {len(c_pred)}')
                # print(f'Val loss for epoch {epoch} is {val_loss}')
                print(f'Overall Ft accuracy for epoch {epoch} is {accuracy*100:.3f}%')
                if params.wandb:
                    wandb.log({'val_loss': val_loss, 'val_accuracy': accuracy*100}, step=epoch)

            if params.ft_tad_loss_weight > 0:
                test_loader_tad = load_thumos(f'tridet/thumos_{params.ft_arch}.yaml', 'test')
                    
                test_db_vars = test_loader_tad.dataset.get_attributes()
                evaluator = ANETdetection(
                    test_loader_tad.dataset.json_file,
                    test_loader_tad.dataset.split[0],
                    tiou_thresholds=test_db_vars['tiou_thresholds']
                )
                fmap = val_epoch_action_detection(fa_model, ft_model_tad, test_loader_tad, evaluator)
                print(f'fMAP for epoch {epoch} is {fmap*100:.3f}%')
                if params.wandb:
                    wandb.log({'val_fmap': fmap*100}, step=epoch)

            if params.ft_ad_loss_weight > 0:
                test_loader_ad = DataLoader(Dataset(ad_args, test_mode=True), batch_size=1, shuffle=False, num_workers=0, pin_memory=False)

                auc, pr_auc = val_epoch_anomaly_detection(fa_model, ft_model_ad, test_loader_ad)
                print(f'AUC for epoch {epoch} is {auc*100:.3f}%')
                if params.wandb:
                    wandb.log({'val_auc': auc, 'val_pr_auc': pr_auc}, step=epoch)

            if accuracy > best_result_ar:
                best_result_ar = accuracy
                best_epoch_ar = epoch
                best_ar_fmap = fmap
                best_ar_auc = auc

            # These val scores are NOT useful for predicting the best model.
            # if fmap > best_result_tad:
            #     best_result_tad = fmap
            #     best_epoch_tad = epoch
            #     best_tad_acc = accuracy
            #     best_tad_auc = auc

            # if auc > best_result_ad:
            #     best_result_ad = auc
            #     best_epoch_ad = epoch
            #     best_ad_acc = accuracy
            #     best_ad_fmap = fmap

        # We will save optimizer weights for each temp model, not all saved models to reduce the storage.
        save_file_path = os.path.join(save_dir, 'model_temp.pth')
        states = {
            'epoch': epoch + 1,
            'fa_model_state_dict': fa_model.state_dict(),
            'ft_model_ar_state_dict': ft_model_ar.state_dict(),
            'ft_model_tad_state_dict': ft_model_tad.state_dict(),
            'ft_model_ad_state_dict': ft_model_ad.state_dict(),
            'optimizer_fa': optimizer_fa.state_dict(),
            'optimizer_ft_ar': optimizer_ft_ar.state_dict(),
            'optimizer_ft_tad': optimizer_ft_tad.state_dict(),
        }
        torch.save(states, save_file_path)

        # Save every 3 to save space.
        if epoch % params.val_freq == 0:
            save_file_path = os.path.join(save_dir, f'model_{epoch}_{accuracy*100:.2f}.pth')
            states = {
                'fa_model_state_dict': fa_model.state_dict(),
                'ft_model_ar_state_dict': ft_model_ar.state_dict(),
                # 'ft_model_tad_state_dict': ft_model_tad.state_dict(), # Best to start fresh...
                # 'ft_model_ad_state_dict': ft_model_ad.state_dict(),
            }
            torch.save(states, save_file_path)

        taken = time.time() - start
        print(f'Time taken for Epoch-{epoch} is {taken:.2f} seconds.')
        print()
    
    print()
    print()
    print('------------------------------------------')
    print(f'Best AR accuracy: {best_result_ar*100:.2f}% at epoch {best_epoch_ar}. (TAD fMAP {best_ar_fmap*100:.2f}%, AUC {best_ar_auc*100:.2f}%)')
    print(f'{best_result_ar*100:.2f}\t{best_ar_fmap*100:.2f}\t{best_ar_auc*100:.2f}')
    # print(f'Best TAD fMAP: {best_result_tad*100:.2f}% at epoch {best_epoch_tad}. (AR accuracy {best_tad_acc*100:.2f}%, AUC {best_tad_auc*100:.2f}%)')
    # print(f'{best_tad_acc*100:.2f}\t{best_result_tad*100:.2f}\t{best_tad_auc*100:.2f}')
    # print(f'Best AD AUC: {best_result_ad*100:.2f}% at epoch {best_epoch_ad}. (AR accuracy {best_ad_acc*100:.2f}%, TAD fMAP {best_ad_fmap*100:.2f}%)')
    # print(f'{best_ad_acc*100:.2f}\t{best_ad_fmap*100:.2f}\t{best_result_ad*100:.2f}')
    import glob
    print(glob.glob(save_dir + f'/model_{best_epoch_ar}_*.pth')[0])
    # print(glob.glob(save_dir + f'/model_{best_epoch_tad}_*.pth')[0])
    # print(glob.glob(save_dir + f'/model_{best_epoch_ad}_*.pth')[0])
    if params.wandb:
        wandb.finish()


if __name__ == "__main__":
    import argparse, importlib
    parser = argparse.ArgumentParser(description='train fa model')
    parser.add_argument("--params", dest='params', type=str, required=False, default='params/params_fa.py', help='params')
    args = parser.parse_args()
    if os.path.exists(args.params):
        params = importlib.import_module(args.params.replace('.py', '').replace('/', '.'))
        print(f'{args.params} is loaded as parameter file.')
    else:
        print(f'{args.params} does not exist, change to valid filename.')

    main(params)