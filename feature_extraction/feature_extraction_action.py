import os.path
import sys

_FEATURE_EXTRACT_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_FEATURE_EXTRACT_DIR)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import h5py
import glob
import numpy as np
import time
import torch
from torch.utils.data import DataLoader
from transformers import VideoMAEForVideoClassification

from dataloaders.kinetics_dl import kinetics_frames_dataset
from dataloaders.vispr_dl import vispr_boring_dataset
from dataloaders.hmdb_dl import hmdb_frames_dataset
from dataloaders.ucf_dl import ucf_frames_dataset
from dataloaders.tsh_dl import tsh_frames_dataset
from dataloaders.ntu_dl import ntu_frames_dataset

import config as cfg
from models.i3d import InceptionI3d
from models.unet_model import UNet
from models.modeling_finetune import vit_huge_patch16_224, vit_giant_patch14_224
from models.load_vjepa import init_vjepa


# Run clips through model.
def extract_features(full_vid, vid_features, save_path, model, model_name, batch_size=16, device='cpu'):
    full_input = torch.stack(full_vid, dim=0)
    with torch.no_grad():
        for i in range(0, len(full_vid), batch_size):
            inputs = full_input[i:i+batch_size].to(device)
            
            if model_name == 'mae':
                output = model(inputs).logits
            else:
                inputs = inputs.permute(0, 2, 1, 3, 4)
                output = model(inputs)
            
            vid_features[i:i+batch_size] = output.squeeze().cpu().numpy()

    np.save(save_path, vid_features)


# Run images through model.
def extract_features_vispr_train(vids, save_path, model, model_name, device='cpu'):
    inputs = vids.to(device)

    with torch.no_grad():
        if model_name == 'mae':
            output = model(inputs).logits
        else:
            inputs = inputs.permute(0, 2, 1, 3, 4)
            output = model(inputs)
        vid_features = output.squeeze().cpu().numpy()

    np.save(save_path, vid_features)


# Run images through model.
def extract_features_vispr_test(vids, save_paths, model, model_name, device='cpu'):
    inputs = vids.to(device)

    with torch.no_grad():
        if model_name == 'mae':
            output = model(inputs).logits
        else:
            inputs = inputs.permute(0, 2, 1, 3, 4)
            output = model(inputs)
        vid_features = output.squeeze().cpu().numpy()

    for i, save_path in enumerate(save_paths):
        np.save(save_path, vid_features[i])


if __name__ == '__main__':
    model_name = 'mae' # 'i3d', 'mae', 'vjepa', 'maev2', 'internvid'
    dataset = 'hmdb'
    datasets = [f'{dataset}_train_{model_name}', f'{dataset}_test_{model_name}']
    reverse = False
    batch_size = 32
    num_fb_frames = 10

    if model_name == 'mae':
        model = VideoMAEForVideoClassification.from_pretrained('MCG-NJU/videomae-base-finetuned-kinetics')
        model.classifier = torch.nn.Identity()
        num_features = 768
    elif model_name == 'vjepa':
        # Make sure both 'vith16.pth.tar' and 'k400-probe.pth.tar' are in the weights directory.
        encoder, classifier = init_vjepa(weights_path='/home/jo869742/PythonProjects/weights/vjepa/vitl16-224')
        classifier.linear = torch.nn.Identity()
        num_features = encoder.embed_dim
        model = torch.nn.Sequential(encoder, classifier)
    elif model_name == 'i3d':
        model = InceptionI3d(num_classes=400)
        msg = model.load_state_dict(torch.load('saved_models/rgb_imagenet.pt'), strict=True)
        print(msg)
        model.logits = torch.nn.Identity()
        num_features = 1024
    elif model_name == 'maev2':
        model = vit_giant_patch14_224(pretrained=False, num_classes=710)
        state_dict = torch.load('saved_models/vit_g_hybrid_pt_1200e_k710_ft.pth')
        msg = model.load_state_dict(state_dict['module'])
        print(msg)
        model.head = torch.nn.Identity()
        num_features = 1408
    elif model_name == 'internvid':
        model = vit_huge_patch16_224(pretrained=False, num_classes=400)
        state_dict = torch.load('saved_models/ViT-H_f32_res384_89.54.pth')
        msg = model.load_state_dict(state_dict['module'])
        print(msg)
        model.head = torch.nn.Identity()
        num_features = 1280

    if torch.cuda.is_available():
        model = model.cuda()
        device = 'cuda:0'
    else:
        device = 'cpu'

    model.eval()

    # Loop through all datasets.
    for dataset in datasets:
        save_features_folder = f'my_features_{dataset}'
        # Make feature extraction folder.
        if not os.path.exists(save_features_folder):
            os.makedirs(save_features_folder)

        print(f'Features folder: {save_features_folder}', flush=True)
        if 'k400' in dataset:
            all_dataset = kinetics_frames_dataset(reverse=reverse, dataset=dataset)
        elif 'hmdb' in dataset:
            all_dataset = hmdb_frames_dataset(reverse=reverse, dataset=dataset)
        elif 'ucf' in dataset:
            all_dataset = ucf_frames_dataset(reverse=reverse, dataset=dataset)
        elif 'tsh' in dataset:
            all_dataset = tsh_frames_dataset(reverse=reverse, split=dataset)
        elif 'ntu' in dataset:
            all_dataset = ntu_frames_dataset(reverse=reverse, split='train' if 'train' in dataset else 'test')
        elif 'vispr' in dataset:
            all_dataset = vispr_boring_dataset(data_split='train' if 'train' in dataset else 'test')
        else:
            raise ValueError(f'Dataset {dataset} not supported.')
        
        print(f'Number of videos: {len(all_dataset)}', flush=True)

        # Different logic for processing VISPR test set.
        if 'vispr' in dataset and 'test' in dataset:
            all_dataloader = DataLoader(all_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
            for i, (full_vids, _, vid_paths) in enumerate(all_dataloader):
                save_paths = [os.path.join(save_features_folder, os.path.basename(vid_path).replace('.avi', '') + '.npy') for vid_path in vid_paths]
                print(f'Extracting features for {[os.path.basename(vid_path) for vid_path in vid_paths]}.', flush=True)
                extract_features_vispr_test(full_vids, save_paths, model, model_name, device=device)
            continue

        # Process all other datasets.
        for i, (full_vid, _, vid_path) in enumerate(all_dataset):
            save_path = os.path.join(save_features_folder, os.path.basename(vid_path).replace('.avi', '') + '.npy')
            if os.path.exists(save_path):
                continue
            print(f'Extracting features for {os.path.basename(vid_path)}.', flush=True)
            if 'vispr' in dataset:
                extract_features_vispr_train(full_vid, save_path, model, model_name, device=device)
                continue
            
            # Will fail if video length is 0.
            try:
                vid_features = np.zeros((len(full_vid), num_features)) # Pre-allocate feature tensor.
            except:
                print(f'Video {os.path.basename(vid_path)} could not process.')
                continue

            extract_features(full_vid, vid_features, save_path, model, model_name, batch_size, device=device)

        all_features = sorted(glob.glob(os.path.join(save_features_folder, '*.npy')))
        print(f'Number of features extracted: {len(all_features)}', flush=True)

        # Write features to h5 file.
        hf = h5py.File(f'features_{dataset}.h5', 'w')
        for feat in all_features:
            feature = np.load(feat)
            hf.create_dataset(os.path.basename(feat).replace('.npy', ''), data=np.float32(feature))

        hf.close()
        print(f'Features written to {f"features_{dataset}.h5"}', flush=True)

        # Extract static frames for budget loss.
        if num_fb_frames > 0 and 'train' in dataset:
            print(f'Starting static frame extraction ({num_fb_frames} per video): {len(all_dataset)} videos', flush=True)
            fb_dataset = hmdb_frames_dataset(reverse=reverse, dataset=dataset, fb_extract=True)
            for _, (fb_stack, _, vid_path) in enumerate(fb_dataset):
                if fb_stack is None:
                    continue
                save_path_fb = os.path.join(save_features_folder, os.path.basename(vid_path).replace('.avi', '') + f'_fb{num_fb_frames}.npy')
                if os.path.exists(save_path_fb):
                    continue
                print(f'Extracting FB frame features for {os.path.basename(vid_path)}.', flush=True)
                extract_features_vispr_train(fb_stack, save_path_fb, model, model_name, device=device)

            # Write features to h5 file.
            fb_paths = sorted(glob.glob(os.path.join(save_features_folder, f'*_fb{num_fb_frames}.npy')))
            hf_fb = h5py.File(f'features_{dataset}_fb{num_fb_frames}.h5', 'w')
            for feat in fb_paths:
                name = os.path.basename(feat).replace(f'_fb{num_fb_frames}.npy', '')
                hf_fb.create_dataset(name, data=np.float32(np.load(feat)))
            hf_fb.close()
            print(f'Features written to features_{dataset}_fb{num_fb_frames}.h5', flush=True)
