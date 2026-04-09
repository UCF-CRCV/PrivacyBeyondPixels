import glob
import json
import os.path
import pandas as pd
import torch
import decord
decord.bridge.set_bridge('torch')
import numpy as np
from torch.utils.data import Dataset
import torchvision.transforms.v2 as v2

import params_feature_ex as params
import config as cfg


# Dataloader for feature extraction.
class ntu_frames_dataset(Dataset):
    def __init__(self, reverse=False, dataset='ntu_mae_train', fb_extract=False):
        all_paths = glob.glob(os.path.join(cfg.ntu_path, '*'))
        video_list = []
        split = 'train' if 'train' in dataset else 'test'

        if split == 'test':
            test_subjects = ['P002', 'P005', 'P006', 'P011', 'P012', 'P013', 'P014', 'P022', 'P023', 'P024', 'P026', 'P029', 'P030', 'P031', 'P032', 'P033', 'P034', 'P035', 'P036', 'P038']
            for path in all_paths:
                video = os.path.basename(path)
                s_num, cam_id, sub_id, rep_num, act_id = video[0:4], video[4:8], video[8:12], video[12:16], video[16:20]
                if int(act_id[1:]) > 60: # or sub_id not in ntu_60_subjects:
                    continue

                if sub_id in test_subjects:
                    video_list.append(path)
        else:
            ntu_60_subjects = range(1, 41)
            ntu_60_subjects = [f'P{i:03d}' for i in ntu_60_subjects]
            banned_subjects = ['P003', 'P004', 'P009', 'P019', 'P021', 'P040']
            ntu_60_subjects = [x for x in ntu_60_subjects if x not in banned_subjects]
            test_subjects = ['P002', 'P005', 'P006', 'P011', 'P012', 'P013', 'P014', 'P022', 'P023', 'P024', 'P026', 'P029', 'P030', 'P031', 'P032', 'P033', 'P034', 'P035', 'P036', 'P038']
            train_subjects = [x for x in ntu_60_subjects if x not in test_subjects]
            
            for path in all_paths:
                video = os.path.basename(path)
                s_num, cam_id, sub_id, rep_num, act_id = video[0:4], video[4:8], video[8:12], video[12:16], video[16:20]
                if int(act_id[1:]) > 60: # or sub_id not in ntu_60_subjects:
                    continue
                if sub_id not in train_subjects:
                    continue
                
                video_list.append(path)
        save_dir = f'my_features_{dataset}'
        existing_features = os.listdir(save_dir) if os.path.isdir(save_dir) else []
        if self.fb_extract:
            fb_tag = f'_fb{params.num_fb_frames}.npy'
            self.video_list = [x for x in self.video_list if os.path.basename(x).replace('.mp4', fb_tag) not in existing_features]
        else:
            self.video_list = [x for x in self.video_list if os.path.basename(x).replace('.mp4', '.npy') not in existing_features]

        print(len(existing_features), len(video_list))
        self.video_list = [x for x in video_list if os.path.basename(x).replace('.mp4', '.npy') not in existing_features]
        # print(len(self.video_list))
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
            import traceback
            traceback.print_exc()
            return None, None, vid_path
