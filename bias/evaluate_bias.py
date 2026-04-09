import os
import sys
import argparse
import importlib.util

_BIAS_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_BIAS_DIR)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from feature_dl import TSHFeaturesDataset, NTUFeaturesDataset
from model_loaders import load_fa, load_ft
import config as cfg


def eval_epoch(fa_model, ft_model, test_loader, pred_dict, label_dict, mode, anon):
    ft_model.eval()
    if anon and fa_model is not None:
        fa_model.eval()

    predictions, gt = [], []
    vid_paths = []

    for _, (features, label, vid_path) in enumerate(test_loader):
        vid_paths.extend(vid_path)
        gt.extend(label.data.numpy())
        features = features.cuda()
        label = label.cuda()

        with torch.no_grad():
            if anon and fa_model is not None:
                features = fa_model(features)
            output = ft_model(features)

        predictions.extend(output.softmax(1).cpu().data.numpy())

    ground_truth = np.asarray(gt)
    pred_array = np.flip(np.argsort(predictions, axis=1), axis=1)
    c_pred = pred_array[:, 0]

    for entry in range(len(vid_paths)):
        key = str(os.path.basename(vid_paths[entry]))
        if key not in pred_dict:
            pred_dict[key] = []
        pred_dict[key].append(predictions[entry])

    for entry in range(len(vid_paths)):
        key = str(os.path.basename(vid_paths[entry]))
        if key not in label_dict:
            label_dict[key] = ground_truth[entry]

    acc = float(np.sum(c_pred == ground_truth)) / len(c_pred)
    print(f"Mode {mode} - batch accuracy: {acc * 100:.3f}%")
    return pred_dict, label_dict


def aggregate_preds(pred_dict, label_dict, num_classes):
    predictions = np.zeros((len(pred_dict), num_classes))
    ground_truth = []
    keys = list(pred_dict.keys())
    for entry, key in enumerate(keys):
        predictions[entry] = np.mean(pred_dict[key], axis=0)
        ground_truth.append(label_dict[key])
    pred_array = np.flip(np.argsort(predictions, axis=1), axis=1)
    c_pred = pred_array[:, 0]
    ground_truth = np.asarray(ground_truth)
    return keys, predictions, ground_truth, c_pred


def preds_to_dataframe_fixed(keys, ground_truth, c_pred):
    rows = []
    for i, key in enumerate(keys):
        rows.append({"file_key": key, "label": int(ground_truth[i]), "pred_label": int(c_pred[i])})
    return pd.DataFrame(rows)


def attach_gender(df, dataset, split):
    if dataset == "tsh":
        labels = pd.read_csv(os.path.join(cfg.tsh_path, "tsh_labels.csv"))
        labels = labels[labels["split"] == split]
        labels = labels.copy()
        labels["_norm"] = labels["filename"].map(lambda x: os.path.basename(str(x)))
        df = df.copy()
        df["_norm"] = df["file_key"].map(lambda x: os.path.basename(str(x)))
        merged = df.merge(labels[["_norm", "gender"]], on="_norm", how="left")
        merged = merged.drop(columns=["_norm"])
        return merged
    if dataset == "ntu":
        gpath = os.path.join(cfg.ntu_feat_path, "ntu_val_labels.csv")
        labels = pd.read_csv(gpath)

        def to_csv_filename(name):
            base = os.path.basename(str(name))
            if not base.endswith(".avi"):
                base = base + ".avi"
            return base

        df = df.copy()
        df["filename"] = df["file_key"].map(to_csv_filename)
        merged = df.merge(labels[["filename", "gender"]], on="filename", how="left")
        return merged
    raise ValueError(f"Unknown dataset: {dataset}")


def report_bias_metrics(df):
    missing = df["gender"].isna().sum()
    if missing:
        print(f"Warning: {missing} rows without gender label after merge.")

    df = df.dropna(subset=["gender"])
    y_true = df["label"].astype(int).values
    y_pred = df["pred_label"].astype(int).values
    gen = df["gender"].astype(str).values

    overall = float(np.mean(y_true == y_pred))
    print(f"\nOverall accuracy: {overall * 100:.3f}%  (n={len(df)})")

    for g in sorted(np.unique(gen)):
        mask = gen == g
        acc_g = float(np.mean(y_true[mask] == y_pred[mask]))
        print(f"  Gender {g}: {acc_g * 100:.3f}%  (n={int(mask.sum())})")


def load_params_module(path):
    path = os.path.abspath(path)
    spec = importlib.util.spec_from_file_location("params_bias_loaded", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main(params):
    torch.set_float32_matmul_precision("high")
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True

    dataset = params.dataset
    if dataset not in ("tsh", "ntu"):
        raise ValueError("This script supports only dataset 'tsh' or 'ntu'.")

    split = getattr(params, "split", "test")

    fa_model = None
    if params.anon:
        fa_model = load_fa(params)

    ft_model = load_ft(params, saved_model=params.saved_ft_model)
    ft_model.cuda()

    if params.anon:
        fa_model.cuda()
        fa_model.eval()

    v_batch_size = getattr(params, "v_batch_size", 16)
    num_workers = getattr(params, "num_workers", 4)
    num_modes = getattr(params, "num_modes", 1)
    modes = np.arange(num_modes)

    pred_dict, label_dict = {}, {}

    for mode in modes:
        if dataset == "tsh":
            test_dataset = TSHFeaturesDataset(params, split=split, mode=mode)
        else:
            test_dataset = NTUFeaturesDataset(params, split=split, mode=mode)
        test_loader = DataLoader(
            test_dataset,
            batch_size=v_batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
        )

        pred_dict, label_dict = eval_epoch(
            fa_model, ft_model, test_loader, pred_dict, label_dict, mode, params.anon
        )

        keys, _, ground_truth, c_pred = aggregate_preds(
            pred_dict, label_dict, params.num_classes
        )
        correct_count = int(np.sum(c_pred == ground_truth))
        accuracy_all = correct_count / len(c_pred)
        print(f"Running avg accuracy: {accuracy_all * 100:.3f}%")

    keys, _, ground_truth, c_pred = aggregate_preds(pred_dict, label_dict, params.num_classes)
    correct_count = int(np.sum(c_pred == ground_truth))
    accuracy = correct_count / len(c_pred)
    print(f"\nCorrect: {correct_count} / {len(c_pred)}")
    print(f"Aggregated accuracy: {accuracy * 100:.3f}%")

    df = preds_to_dataframe_fixed(keys, ground_truth, c_pred)
    df = attach_gender(df, dataset, split)

    report_bias_metrics(df)


if __name__ == "__main__":
    import argparse, importlib
    parser = argparse.ArgumentParser(description='evaluate bias')
    parser.add_argument("--params", dest='params', type=str, required=False, default='bias/params_bias.py', help='params')
    args = parser.parse_args()
    if os.path.exists(args.params):
        params = importlib.import_module(args.params.replace('.py', '').replace('/', '.'))
        print(f'{args.params} is loaded as parameter file.')
    else:
        print(f'{args.params} does not exist, change to valid filename.')
    
    main(params)
