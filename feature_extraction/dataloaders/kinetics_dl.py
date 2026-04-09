import glob
import os.path
import torch
import decord
decord.bridge.set_bridge('torch')
import numpy as np
from torch.utils.data import Dataset
import torchvision.transforms.v2 as v2
from torchvision.io import read_image
import random

import params_feature_ex as params
import config as cfg


# Dataloader for feature extraction.
class kinetics_frames_dataset(Dataset):
    def __init__(self, reverse=False, dataset='kinetics_mae_train', mode=0, num_modes=1, fb_extract=False):
        split = 'train' if 'train' in dataset else 'test'
        video_list = sorted(glob.glob(os.path.join(cfg.kinetics_path, split, '*')))
        mode, num_modes = float(mode), float(num_modes)
        num_vids = len(video_list)
        # Slice up dataset for parallel processing.
        start_idx = int((mode/num_modes)*num_vids)
        end_idx = int(((mode+1)/num_modes)*num_vids)
        video_list = video_list[start_idx:end_idx]
        self.fb_extract = fb_extract
        save_dir = f'my_features_{dataset}'
        existing_features = os.listdir(save_dir) if os.path.isdir(save_dir) else []
        if self.fb_extract:
            fb_tag = f'_fb{params.num_fb_frames}.npy'
            self.video_list = [x for x in self.video_list if os.path.basename(x).replace('.mp4', fb_tag) not in existing_features]
        else:
            self.video_list = [x for x in self.video_list if os.path.basename(x).replace('.mp4', '.npy') not in existing_features]
        if reverse:
            self.video_list.reverse()

        self.augmentation = v2.Compose([
            v2.Resize(size=256, antialias=True),
            v2.CenterCrop(size=(224, 224)),
            v2.ToDtype(torch.float32, scale=True),  # Normalize expects float input
            v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        self.train_augmentation = v2.Compose([
            v2.RandomResizedCrop(size=(224, 224), scale=(0.8, 1.0), antialias=True),
            v2.RandomHorizontalFlip(p=0.5),
            v2.ToDtype(torch.float32, scale=True),  # Normalize expects float input
            v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def __len__(self):
        return len(self.video_list)

    def __getitem__(self, idx):
        if self.fb_extract:
            return self.read_video_fb_frames(self.video_list[idx])
        return self.read_video(self.video_list[idx])

    def read_video_fb_frames(self, vid_path):
        try:
            vr = decord.VideoReader(vid_path)
            total_frames = len(vr)
            if total_frames == 0:
                return None, None, vid_path
            n = params.num_fb_frames
            frame_idx = np.linspace(0, total_frames - 1, n, dtype=int)
            all_frames = vr.get_batch(frame_idx.tolist())
            clips = []
            for frame in all_frames:
                frame = frame.permute(2, 0, 1)
                clip = frame.unsqueeze(0).repeat(params.num_frames, 1, 1, 1)
                clips.append(self.train_augmentation(clip))
            return torch.stack(clips, dim=0), None, vid_path
        except:
            import traceback
            traceback.print_exc()
            return None, None, vid_path

    # Clip builder, reads video frames from custom start and end times (if necessary), stacks them.
    def read_video(self, vid_path):
        try:
            vr = decord.VideoReader(vid_path)
            total_frames = len(vr)

            full_vid = []
            full_frame_pos = []
            frame_pos = []
            clip_frames = []

            count = 0
            repeat = False
            if total_frames < params.fix_skip*params.num_frames:
                fix_skip = 1
            else:
                fix_skip = params.fix_skip

            if total_frames < params.num_frames:
                repeat = True

            frame_idx = np.arange(0, total_frames, fix_skip)
            all_frames = vr.get_batch(frame_idx.tolist())

            count = 0
            for frame in all_frames:
                count += 1
                if repeat and count == total_frames:
                    keep_frame = frame
                clip_frames.append(frame.permute(2, 0, 1))
                frame_pos.append(count)
                if count % (params.num_frames*fix_skip) == 0:
                    full_vid.append(self.augmentation(torch.stack(clip_frames, dim=0)))
                    full_frame_pos.append(frame_pos)
                    frame_pos = []
                    clip_frames = []

            if repeat:
                count -= 1
                last_frame = count
                # In case of size mismatch, stack last frame.
                while count % params.num_frames != 0:
                    count += 1
                    clip_frames.append(keep_frame)
                    frame_pos.append(last_frame)
                    if count % 16 == 0:
                        full_vid.append(self.augmentation(torch.stack(clip_frames, dim=0)))
                        full_frame_pos.append(frame_pos)

            return full_vid, full_frame_pos, vid_path
        except:
            # import traceback
            # traceback.print_exc()
            return None, None, vid_path
