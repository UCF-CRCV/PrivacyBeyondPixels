import h5py
import json
import numpy as np
import os.path
import pandas as pd
import pickle
import random
import torch
from torch.utils.data import Dataset, DataLoader

import config as cfg


def _sample_fb_frames(fb_feature_list, fb_key, fb_frame_sample):
    fb_full = torch.from_numpy(fb_feature_list[fb_key][...])
    perm = torch.randperm(fb_full.shape[0])[:fb_frame_sample]
    return fb_full[perm]


class KineticsFeaturesDataset(Dataset):
    def __init__(self, params, split='train', dataset=None, data_percentage=1.0, shuffle=False, mode=-1, return_fb_frames=False):
        self.params = params
        self.dataset = params.ar_dataset if dataset is None else dataset
        self.split = split
        self.model = params.ft_arch
        if self.model == 'videoMAE':
            self.model = 'mae'
        self.mode = mode
        self.num_modes = self.params.num_modes
        if self.num_modes == 1:
            self.num_modes = 5
            self.mode = 2
        # TEMP
        if self.split == 'val':
            self.split = 'test'
        self.return_fb_frames = return_fb_frames
        self.fb_frame_pool = getattr(params, 'fb_frame_pool', 10)
        self.fb_frame_sample = getattr(params, 'fb_frame_sample', 2)
        self.feature_list = h5py.File(os.path.join(cfg.kinetics_feat_path, f'features_kinetics_{self.split}_{self.model}.h5'), 'r')
        if self.return_fb_frames:
            self.fb_feature_list = h5py.File(os.path.join(cfg.kinetics_feat_path, f'features_kinetics_{self.split}_{self.model}_fb{self.fb_frame_pool}.h5'), 'r')
        self.all_paths = list(self.feature_list.keys())
        self.label_dict = json.load(open(os.path.join(cfg.kinetics_feat_path, f'k400_{self.split}_labels.json'), 'r'))
        if self.model == 'i3d' or self.model == 'vjepa':
            self.all_paths = [x[:-4] for x in self.all_paths]
        for key in self.all_paths.copy():
            if key not in self.label_dict.keys():
                # print(f'{key} not in {self.split}')
                # self.label_dict.pop(key)
                self.all_paths.remove(key)
        
        self.shuffle = shuffle
        if self.shuffle:
            random.shuffle(self.all_paths)

        self.data_percentage = data_percentage
        self.data_limit = int(len(self.all_paths)*self.data_percentage)
        self.data = self.all_paths[0: self.data_limit]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        feature_name = self.data[idx]
        label = int(self.label_dict[feature_name]) - 1
        features = self.get_data(feature_name)
        if self.return_fb_frames:
            fb_pair = _sample_fb_frames(self.fb_feature_list, feature_name, self.fb_frame_sample)
            return features, label, feature_name, fb_pair
        return features, label, feature_name
    

    def get_data(self, feature_name):
        features = torch.from_numpy(self.feature_list[feature_name][...])
        if features.shape[0] == 0:
            return torch.zeros(features.shape[1])
        if self.mode == -1:
            # rand_idx = torch.randint(0, features.shape[0], (2,))
            rand_idx = torch.randint(0, features.shape[0], (1,))
        else:
            rand_idx = int(torch.linspace(0, features.shape[0]-1, self.num_modes)[self.mode])
        return features[rand_idx].squeeze()
    

class HMDBFeaturesDataset(Dataset):
    def __init__(self, params, split='train', dataset=None, data_percentage=1.0, shuffle=False, mode=-1, num_modes=-1, return_fb_frames=False):
        self.params = params
        # self.dataset = params.dataset if dataset is None else dataset
        self.split = split
        self.model = params.ft_arch
        if self.model == 'videoMAE':
            self.model = 'mae'
        self.mode = mode
        self.num_modes = num_modes if num_modes != -1 else self.params.num_modes
        if self.num_modes == 1 and self.mode != -1:
            self.num_modes = 5
            self.mode = 2
        self.return_fb_frames = return_fb_frames
        self.fb_frame_pool = getattr(params, 'fb_frame_pool', 10)
        self.fb_frame_sample = getattr(params, 'fb_frame_sample', 2)
        self.feature_list = h5py.File(os.path.join(cfg.hmdb_feat_path, f'features_hmdb_{self.split}_{self.model}.h5'), 'r')
        if self.return_fb_frames:
            self.fb_feature_list = h5py.File(os.path.join(cfg.hmdb_feat_path, f'features_hmdb_{self.split}_{self.model}_fb{self.fb_frame_pool}.h5'), 'r')

        anno_file = pd.read_csv(os.path.join(cfg.hmdb_path, 'hmdb51_labels.csv'), index_col=None)
        self.classes = {os.path.basename(f)[:-4]: l for f, l in zip(anno_file['filename'].to_list(), anno_file['label'].to_list())}
        self.all_paths = sorted(list(self.feature_list.keys()))
        for key in self.all_paths.copy():
            if key not in self.classes.keys():
                # print(f'{key} not in {self.split}')
                # self.label_dict.pop(key)
                self.all_paths.remove(key)

        self.shuffle = shuffle

        if self.shuffle:
            random.shuffle(self.all_paths)

        self.data_percentage = data_percentage
        self.data_limit = int(len(self.all_paths)*self.data_percentage)
        self.data = self.all_paths[0: self.data_limit]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        feature_name = self.data[idx]
        key = os.path.basename(feature_name).replace('.avi', '')
        label = self.classes[key]
        features = self.get_data(feature_name)
        if self.return_fb_frames:
            fb_pair = _sample_fb_frames(self.fb_feature_list, key, self.fb_frame_sample)
            return features, label, feature_name, fb_pair
        return features, label, feature_name
    
    def get_data(self, feature_name):
        features = torch.from_numpy(self.feature_list[feature_name][...])
        if self.mode == -1:
            rand_idx = torch.randint(0, features.shape[0], (1,))
        else:
            rand_idx = int(torch.linspace(0, features.shape[0]-1, self.num_modes)[self.mode])
        return features[rand_idx].squeeze()
    

class UCFFeaturesDataset(Dataset):
    def __init__(self, params, split='train', dataset=None, data_percentage=1.0, shuffle=False, mode=-1, num_modes=-1, return_fb_frames=False):
        self.params = params
        self.split = split
        self.model = params.ft_arch
        if self.model == 'videoMAE':
            self.model = 'mae'
        self.mode = mode
        self.num_modes = num_modes if num_modes != -1 else self.params.num_modes
        if self.num_modes == 1 and self.mode != -1:
            self.num_modes = 5
            self.mode = 2
        self.return_fb_frames = return_fb_frames
        self.fb_frame_pool = getattr(params, 'fb_frame_pool', 10)
        self.fb_frame_sample = getattr(params, 'fb_frame_sample', 2)
        self.feature_list = h5py.File(os.path.join(cfg.ucf101_feat_path, f'features_ucf101_{self.split}_{self.model}.h5'), 'r')
        if self.return_fb_frames:
            self.fb_feature_list = h5py.File(os.path.join(cfg.ucf101_feat_path, f'features_ucf101_{self.split}_{self.model}_fb{self.fb_frame_pool}.h5'), 'r')

        if split == 'train':
            all_paths = open(os.path.join(cfg.ucf101_path, 'ucfTrainTestlist', f'trainlist01.txt'),'r').read().splitlines()
        else:
            all_paths = open(os.path.join(cfg.ucf101_path, 'ucfTrainTestlist', f'testlist01.txt'),'r').read().splitlines()
        self.all_paths = [x.replace('/', os.sep) for x in all_paths]
        self.classes = json.load(open(os.path.join(cfg.ucf101_path, 'ucfTrainTestlist', 'action_classes.json')))['classes']

        self.shuffle = shuffle

        if self.shuffle:
            random.shuffle(self.all_paths)

        self.data_percentage = data_percentage
        self.data_limit = int(len(self.all_paths)*self.data_percentage)
        self.data = self.all_paths[0: self.data_limit]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        feature_name = self.data[idx]
        label = int(self.classes[feature_name.split(os.sep)[-2]]) - 1
        ucf_key = os.path.basename(feature_name).split(' ')[0][:-4]
        features = self.get_data(ucf_key)
        if self.return_fb_frames:
            fb_pair = _sample_fb_frames(self.fb_feature_list, ucf_key, self.fb_frame_sample)
            return features, label, feature_name, fb_pair
        return features, label, feature_name
    
    def get_data(self, feature_name):
        features = torch.from_numpy(self.feature_list[feature_name][...])
        if self.mode == -1:
            # rand_idx = torch.randint(0, features.shape[0], (2,))
            rand_idx = torch.randint(0, features.shape[0], (1,))
        else:
            rand_idx = int(torch.linspace(0, features.shape[0]-1, self.num_modes)[self.mode])
        return features[rand_idx].squeeze()


class VPHMDBFeaturesDataset(Dataset):
    def __init__(self, params, split='train', dataset=None, data_percentage=1.0, shuffle=False, mode=-1, num_modes=-1):
        self.params = params
        self.split = split
        self.model = params.ft_arch
        if self.model == 'videoMAE':
            self.model = 'mae'
        self.mode = mode
        self.num_modes = num_modes if num_modes != -1 else self.params.num_modes
        if self.num_modes == 1 and self.mode != -1:
            self.num_modes = 5
            self.mode = 2
        self.feature_list = h5py.File(os.path.join(cfg.hmdb_feat_path, f'features_vp_hmdb_{self.split}_{self.model}.h5'), 'r')

        test_file = pd.read_csv(os.path.join(cfg.hmdb_path, f'hmdb51_{self.split}_labels.csv'), index_col=None)
        labels = {k: v for k, v in zip(test_file['filename'].to_list(), test_file['label'].to_list())}
        anno_file = pd.read_csv(os.path.join(cfg.hmdb_path, 'vphmdb51_labels_num.csv'), index_col=None)
        classes = {k: v for k, v in zip(anno_file['filename'].to_list(), anno_file['label'].to_list()) if os.path.basename(k) in test_file['filename'].to_list()}
        self.classes = {k: (v, labels[os.path.basename(k)]) for k, v in classes.items()}
        self.all_paths = list(self.classes.keys())

        self.shuffle = shuffle

        if self.shuffle:
            random.shuffle(self.all_paths)

        self.data_percentage = data_percentage
        self.data_limit = int(len(self.all_paths)*self.data_percentage)
        self.data = self.all_paths[0: self.data_limit]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        feature_name = self.data[idx]
        vlabels, label = self.classes[feature_name] 
        vlabels = [int(x.replace('[', '').replace(']', '').replace('.', '')) for x in vlabels.split(' ')]
        vlabels = torch.tensor(vlabels, dtype=torch.float32)
        features = self.get_data(feature_name)
        return features, label, vlabels, feature_name
    

    def get_data(self, feature_name):
        features = torch.from_numpy(self.feature_list[feature_name][...])

        if self.mode == -1:
            rand_idx = torch.randint(0, features.shape[0], (1,))
        else:
            rand_idx = int(torch.linspace(0, features.shape[0]-1, self.num_modes)[self.mode])
        return features[rand_idx].squeeze()
    

class VPUCFFeaturesDataset(Dataset):
    def __init__(self, params, split='train', dataset=None, data_percentage=1.0, shuffle=False, mode=-1, num_modes=-1):
        self.params = params
        self.split = split
        self.model = params.ft_arch
        if self.model == 'videoMAE':
            self.model = 'mae'
        self.mode = mode
        self.num_modes = num_modes if num_modes != -1 else self.params.num_modes
        if self.num_modes == 1 and self.mode != -1:
            self.num_modes = 5
            self.mode = 2
        self.feature_list = h5py.File(os.path.join(cfg.ucf101_feat_path, f'features_ucf101_{self.split}_{self.model}.h5'), 'r')

        classes = json.load(open(os.path.join(cfg.ucf101_path, 'ucfTrainTestlist', 'action_classes.json')))['classes']
        all_paths = open(os.path.join(cfg.ucf101_path, 'ucfTrainTestlist', f'{self.split}list01.txt'),'r').read().splitlines()
        all_paths = [x.replace('/', os.sep).split(' ')[0] for x in all_paths] if self.split == 'train' else [x.replace('/', os.sep) for x in all_paths]
        anno_file = pd.read_csv(os.path.join(cfg.ucf101_path, 'vpucf101_labels_num.csv'), index_col=None)
        self.classes = {k: v for k, v in zip(anno_file['filename'].to_list(), anno_file['label'].to_list()) if k in all_paths}
        self.classes = {k: (v, classes[k.split(os.sep)[0]]) for k, v in self.classes.items()}
        self.all_paths = list(self.classes.keys())

        self.shuffle = shuffle

        if self.shuffle:
            random.shuffle(self.all_paths)

        self.data_percentage = data_percentage
        self.data_limit = int(len(self.all_paths)*self.data_percentage)
        self.data = self.all_paths[0: self.data_limit]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        feature_name = self.data[idx]
        vlabels, label = self.classes[feature_name]
        label -= 1
        vlabels = [int(x.replace('[', '').replace(']', '').replace('.', '')) for x in vlabels.split(' ')]
        vlabels = torch.tensor(vlabels, dtype=torch.float32)
        features = self.get_data(os.path.basename(feature_name).split(' ')[0][:-4])
        return features, label, vlabels, feature_name
    

    def get_data(self, feature_name):
        features = torch.from_numpy(self.feature_list[feature_name][...])
        if self.mode == -1:
            rand_idx = torch.randint(0, features.shape[0], (1,))
        else:
            rand_idx = int(torch.linspace(0, features.shape[0]-1, self.num_modes)[self.mode])
        return features[rand_idx].squeeze()
    

class TSHFeaturesDataset(Dataset):
    def __init__(self, params, split='train', dataset=None, data_percentage=1.0, shuffle=False, mode=-1, num_modes=-1, task='action', return_fb_frames=False):
        self.params = params
        self.split = split
        self.model = params.ft_arch
        if self.model == 'videoMAE':
            self.model = 'mae'
        self.mode = mode
        self.num_modes = num_modes if num_modes != -1 else self.params.num_modes
        if self.num_modes == 1 and self.mode != -1:
            self.num_modes = 5
            self.mode = 2
        self.return_fb_frames = return_fb_frames
        self.fb_frame_pool = getattr(params, 'fb_frame_pool', 10)
        self.fb_frame_sample = getattr(params, 'fb_frame_sample', 2)
        self.feature_list = h5py.File(os.path.join(cfg.tsh_feat_path, f'features_tsh_{self.split}_{self.model}.h5'), 'r')
        if self.return_fb_frames:
            self.fb_feature_list = h5py.File(os.path.join(cfg.tsh_feat_path, f'features_tsh_{self.split}_{self.model}_fb{self.fb_frame_pool}.h5'), 'r')
        self.task = task

        anno_file = pd.read_csv(os.path.join(cfg.tsh_path, 'tsh_labels.csv'), index_col=None)
        anno_file = anno_file[anno_file['split'] == split]

        self.all_paths = anno_file['filename'].to_list()
        self.classes = json.load(open(os.path.join(cfg.tsh_path, 'tsh_class_mapping.json'), 'r'))

        self.data_percentage = data_percentage
        self.data_limit = int(len(self.all_paths)*self.data_percentage)
        self.data = self.all_paths[0: self.data_limit]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        feature_name = self.data[idx]
        label = self.classes[os.path.basename(feature_name).split('_')[0]] - 1
        bn = os.path.basename(feature_name)
        features = self.get_data(bn)
        if self.return_fb_frames:
            fb_pair = _sample_fb_frames(self.fb_feature_list, bn, self.fb_frame_sample)
            return features, label, feature_name, fb_pair
        return features, label, feature_name
    

    def get_data(self, feature_name):
        features = torch.from_numpy(self.feature_list[feature_name][...])
        if self.mode == -1:
            rand_idx = torch.randint(0, features.shape[0], (1,))
        else:
            rand_idx = int(torch.linspace(0, features.shape[0]-1, self.num_modes)[self.mode])
        return features[rand_idx].squeeze()
    

class NTUFeaturesDataset(Dataset):
    def __init__(self, params, split='train', dataset=None, data_percentage=1.0, shuffle=False, mode=-1, num_modes=-1, task='action', return_fb_frames=False):
        self.params = params
        self.split = split
        self.model = params.ft_arch
        if self.model == 'videoMAE':
            self.model = 'mae'
        self.mode = mode
        self.num_modes = num_modes if num_modes != -1 else self.params.num_modes
        if self.num_modes == 1 and self.mode != -1:
            self.num_modes = 5
            self.mode = 2
        self.task = task
        self.return_fb_frames = return_fb_frames
        self.fb_frame_pool = getattr(params, 'fb_frame_pool', 10)
        self.fb_frame_sample = getattr(params, 'fb_frame_sample', 2)
        self.feature_list = h5py.File(os.path.join(cfg.ntu_feat_path, f'features_ntu_{self.split}_{self.model}.h5'), 'r')
        if self.return_fb_frames:
            self.fb_feature_list = h5py.File(os.path.join(cfg.ntu_feat_path, f'features_ntu_{self.split}_{self.model}_fb{self.fb_frame_pool}.h5'), 'r')
        
        test_subjects = ['P002', 'P005', 'P006', 'P011', 'P012', 'P013', 'P014', 'P022', 'P023', 'P024', 'P026', 'P029', 'P030', 'P031', 'P032', 'P033', 'P034', 'P035', 'P036', 'P038']
        all_paths = list(self.feature_list.keys())
        # print(f'All paths: {len(all_paths)}')
        self.all_paths = []
        if split == 'test':
            for path in all_paths:
                video = os.path.basename(path)
                s_num, cam_id, sub_id, rep_num, act_id = video[0:4], video[4:8], video[8:12], video[12:16], video[16:20]
                if int(act_id[1:]) > 60: # or sub_id not in ntu_60_subjects:
                    continue
                if sub_id in test_subjects:
                    self.all_paths.append(path)
        else:
            ntu_60_subjects = range(1, 41)
            ntu_60_subjects = [f'P{i:03d}' for i in ntu_60_subjects]
            banned_subjects = ['P003', 'P004', 'P009', 'P019', 'P021', 'P040']
            ntu_60_subjects = [x for x in ntu_60_subjects if x not in banned_subjects]
            train_subjects = [x for x in ntu_60_subjects if x not in test_subjects]
            
            male_balance_subject = 'P010'
            female_balance_subject = 'P039'
            gender_file = json.load(open(os.path.join(cfg.ntu_path, 'ntu_subject_gender_labels.json')))

            fem_actions = []
            male_actions = []

            if self.params.split == 'femact3':
                fem_actions = ['A001', 'A002', 'A004']
            elif self.params.split == 'femact1':
                fem_actions = ['A004']
                # fem_actions = ['A025']
            elif self.params.split == 'maleact1':
                male_actions = ['A004']
                # male_actions = ['A025']
            elif self.params.split == 'maleact3':
                male_actions = ['A001', 'A002', 'A004']

            for path in all_paths:
                video = os.path.basename(path)
                s_num, cam_id, sub_id, rep_num, act_id = video[0:4], video[4:8], video[8:12], video[12:16], video[16:20]
                if int(act_id[1:]) > 60: # or sub_id not in ntu_60_subjects:
                    continue
                if sub_id not in train_subjects:
                    continue

                if len(fem_actions) > 0:
                    # Specific actions mainly female.
                    if act_id in fem_actions:
                        if sub_id in gender_file['female'] or sub_id == male_balance_subject:
                            if sub_id == female_balance_subject:
                                continue
                            self.all_paths.append(path)
                        else:
                            continue
                    
                    if sub_id in train_subjects:
                        if sub_id in gender_file['male'] or sub_id == female_balance_subject:
                            if sub_id == male_balance_subject:
                                continue
                            self.all_paths.append(path)
                elif len(male_actions) > 0:
                    # Specific actions mainly male.
                    if act_id in male_actions:
                        if sub_id in gender_file['male'] or sub_id == female_balance_subject:
                            if sub_id == male_balance_subject:
                                continue
                            self.all_paths.append(path)
                        else:
                            continue
                    
                    if sub_id in train_subjects:
                        if sub_id in gender_file['female'] or sub_id == male_balance_subject:
                            if sub_id == female_balance_subject:
                                continue
                            self.all_paths.append(path)

                elif self.params.split == 'male':
                    if sub_id in train_subjects and sub_id in gender_file['male']:
                        self.all_paths.append(path)
                
                elif self.params.split == 'female':
                    if sub_id in train_subjects and sub_id in gender_file['female']:
                        self.all_paths.append(path)

                else:
                    if sub_id in train_subjects:
                        self.all_paths.append(path)

        self.shuffle = shuffle

        if self.shuffle:
            random.shuffle(self.all_paths)

        self.data_percentage = data_percentage
        self.data_limit = int(len(self.all_paths)*self.data_percentage)
        self.data = self.all_paths[0: self.data_limit]

        if self.split == 'gallery':
            self.load_gallery()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        feature_name = self.data[idx]
        label = int(os.path.basename(feature_name)[17:20]) - 1
        bn = os.path.basename(feature_name)
        features = self.get_data(bn)
        if self.return_fb_frames:
            fb_pair = _sample_fb_frames(self.fb_feature_list, bn, self.fb_frame_sample)
            return features, label, feature_name, fb_pair
        return features, label, feature_name
    

    def get_data(self, feature_name):
        features = torch.from_numpy(self.feature_list[feature_name][...])
        if self.mode == -1:
            rand_idx = torch.randint(0, features.shape[0], (1,))
        else:
            rand_idx = int(torch.linspace(0, features.shape[0]-1, self.num_modes)[self.mode])
        return features[rand_idx].squeeze()
    

class VISPRFeaturesDataset(Dataset):
    def __init__(self, split='train', data_percentage=1.0, shuffle=False, model='videoMAE'):
        self.split = split
        self.model = model
        if model == 'videoMAE':
            self.model = 'mae'

        self.feature_list = h5py.File(os.path.join(cfg.vispr_feat_path, f'features_vispr_{self.split}_{self.model}.h5'), 'r')
        self.all_paths = list(self.feature_list.keys())
        self.label_dict = pickle.load(open(os.path.join(cfg.vispr_path , f'{self.split}_labels.pkl'), 'rb'))
        for key in self.all_paths.copy():
            key = key.split('.')[0]
            if key not in self.label_dict.keys():
                print(f'{key} not in {self.split}')
                # self.label_dict.pop(key)
                self.all_paths.remove(key)

        self.shuffle = shuffle

        if self.shuffle:
            random.shuffle(self.all_paths)

        self.data_percentage = data_percentage
        self.data_limit = int(len(self.all_paths)*self.data_percentage)
        self.data = self.all_paths[0: self.data_limit]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        feature_name = self.data[idx]
        label = self.label_dict[feature_name.split('.')[0]]
        features = self.get_data(feature_name)

        if self.split == 'train':
            rand_idx = np.random.randint(0, features.shape[0], size=(1,))[0]
            features = features[rand_idx]
        return features, label, feature_name

    def get_data(self, feature_name):
        features = torch.from_numpy(self.feature_list[feature_name][...])
        return features


if __name__ == '__main__':
    import params.params_fa as params

    train_dataset = HMDBFeaturesDataset(params, split='train')
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

    for batch_idx, (features, label, name) in enumerate(train_loader):
        print(batch_idx, features.shape, name)
        break

    vispr_dataset = VISPRFeaturesDataset(split='train', shuffle=False, data_percentage=0.01)
    vispr_loader = DataLoader(vispr_dataset, batch_size=32, shuffle=True)

    for batch_idx, (features, label, name) in enumerate(vispr_loader):
        print(batch_idx, features.shape, name)
        break