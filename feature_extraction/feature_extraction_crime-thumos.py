import os.path
import sys

_FEATURE_EXTRACT_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_FEATURE_EXTRACT_DIR)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import glob as glob
import numpy as np
from models.modeling_finetune import vit_huge_patch16_224, vit_giant_patch14_224
from nvidia.dali.plugin.pytorch import DALIGenericIterator
from nvidia.dali.pipeline import Pipeline
from nvidia.dali import fn, types
import time
import torch
import torchvision.transforms.functional as F
import torchvision.transforms.v2 as v2
from transformers import VideoMAEForVideoClassification

import params_feature_ex as params
import config as cfg
from models.load_vjepa import init_vjepa
from models.i3d import InceptionI3d


# Pytorch DataLoader.
class DALIDataloader(DALIGenericIterator):
    def __init__(self, pipeline, params, reader_name, batch_size, output_map=["data", "label"], auto_reset=False):
        self.params = params
        self.output_map = output_map
        super().__init__(pipelines=pipeline, reader_name=reader_name, auto_reset=auto_reset, output_map=output_map)
        self.iter_batch_size = batch_size

    def __len__(self):
        if self.size % self.iter_batch_size == 0:
            return self.size//self.iter_batch_size
        else:
            return self.size//self.iter_batch_size+1

    def __next__(self):
        data = super().__next__()[0]
        video, label = data[self.output_map[0]], data[self.output_map[1]]
        video = self.val_augmentations(video)
        return video, label

    def val_augmentations(self, video):
        augmentation = v2.Compose([
            v2.Resize(size=256, antialias=True),
            v2.CenterCrop(size=(224, 224)),
            v2.ToDtype(torch.float32, scale=True),  # Normalize expects float input
            v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        video = video / 255.
        video = video.permute(0, 1, 4, 2, 3)
        video = augmentation(video).permute(0, 2, 1, 3, 4)
        return video


class HybridValPipe(Pipeline):
    def __init__(self, data_dir, params, cropping_factor=0.8, num_threads=4, device_id=0, num_gpus=1):
        super(HybridValPipe, self).__init__(1, num_threads, device_id)
        dali_device = 'gpu'
        self.params = params
        self.input = fn.readers.video(
            filenames=data_dir,
            labels=[],
            sequence_length=params.num_frames,
            num_shards=num_gpus,
            shard_id=device_id,
            device=dali_device,
            random_shuffle=False,
            pad_sequences=True,
            stride=params.fix_skip,
            step=params.num_frames*params.fix_skip,
            file_list_include_preceding_frame=True,
            dtype=types.DALIDataType.FLOAT,
            name='reader')

        self.min_size = min(params.reso_h, params.reso_w)


    def define_graph(self):
        videos, labels = self.input
        return [videos, labels]


# This code is used for feature extraction from long videos.
if __name__ == '__main__':
    model_name = 'i3d' # 'i3d', 'mae', 'vjepa', 'maev2', 'internvid'
    batch_size = 1
    dataset = 'crime' # 'crime', 'thumos'
    save_features_folder = f'my_features_{dataset}_{model_name}'
    
    print(f'Model: {model_name}', flush=True)
    print(f'Features folder: {save_features_folder}', flush=True)

    # Make feature extraction folder.
    if not os.path.exists(save_features_folder):
        os.makedirs(save_features_folder)

    # If files are set up differently, change this. This code should work for any glob of filenames.
    if dataset == 'crime':
        filenames = sorted(glob.glob(os.path.join(cfg.ucf_crime_path, 'Videos', '*', '*')))
    elif dataset == 'thumos':
        filenames = sorted(glob.glob(os.path.join(cfg.thumos_path, 'videos', '*')))
    else:
        raise ValueError(f'Invalid dataset: {dataset}')

    if len(filenames) == 0:
        raise ValueError(f'No filenames found for {dataset}.')

    # Remove all existing filenames.
    filenames = [x for x in filenames if not os.path.exists(os.path.join(save_features_folder, os.path.basename(x).replace('.mp4', '') + '.npy'))]
    

    if model_name == 'mae':
        model = VideoMAEForVideoClassification.from_pretrained('MCG-NJU/videomae-base-finetuned-kinetics')
        model.classifier = torch.nn.Identity()
        num_features = 768
    elif model_name == 'vjepa':
        # Make sure both 'vith16.pth.tar' and 'k400-probe.pth.tar' are in the weights directory.
        encoder, classifier = init_vjepa(weights_path='/path/to/vjepa/weights')
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
    model.eval()

    # Create DALI pipeline.
    pipes = [HybridValPipe(data_dir=filenames, params=params, num_gpus=1, device_id=0)]
    all_dataset = DALIDataloader(pipeline=pipes, params=params, reader_name='reader', batch_size=batch_size)

    prev_label = -1
    first_vid = True

    for _, (inputs, label) in enumerate(all_dataset):
        label = label.item()
        # Processing is a bit awkward to work with DALI pipeline. But it works.
        if label != prev_label:
            if not first_vid:
                if count != -1:
                    with torch.no_grad():
                        output = model(long_inputs[:count+1])
                    vid_features = np.vstack([vid_features, output.squeeze().detach().cpu().numpy()])
                np.save(save_path, vid_features[1:])
                print(save_path, vid_features.shape, flush=True)

            prev_label = label
            vid_path = filenames[label]
            save_path = os.path.join(save_features_folder, os.path.basename(vid_path).replace('.mp4', '') + '.npy')
            if not os.path.exists(save_path):
                print(f'Extracting features for {os.path.basename(vid_path)}.', flush=True)
            vid_features = np.zeros(num_features)
            long_inputs = torch.zeros((batch_size, 3, 16, 224, 224), device='cuda')
            count = -1
            first_vid = False
            if os.path.exists(save_path):
                continue

        count += 1
        with torch.no_grad():
            long_inputs[count] = inputs[0]
            if count == batch_size-1:
                output = model(long_inputs)
                vid_features = np.vstack([vid_features, output.squeeze().detach().cpu().numpy()])
                count = -1
                long_inputs = torch.zeros((batch_size, 3, 16, 224, 224), device='cuda')

    np.save(save_path, vid_features[1:])

