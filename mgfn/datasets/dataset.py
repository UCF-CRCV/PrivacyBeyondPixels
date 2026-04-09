import torch.utils.data as data
import numpy as np
from mgfn.utils.utils import process_feat
import torch
torch.set_default_dtype(torch.float32)
# torch.set_default_device('cuda')
# from mgfn import option
# args=option.parse_args()

class Dataset(data.Dataset):
    def __init__(self, args, is_normal=True, transform=None, test_mode=False):
        self.args = args
        self.modality = self.args.modality
        self.is_normal = is_normal
        if test_mode:
            self.rgb_list_file = self.args.test_rgb_list
        else:
            self.rgb_list_file = self.args.rgb_list
        self.tranform = transform
        self.test_mode = test_mode
        self._parse_list()
        self.num_frame = 0
        self.labels = None
        self.fa_model = self.args.fa_model

    def _parse_list(self):
        self.list = list(open(self.rgb_list_file))
        if self.test_mode is False:
            if self.args.datasetname == 'UCF':
                if self.is_normal:
                    self.list = self.list[810:]#ucf 810; sht63; xd 9525
                    # print('normal list')
                    # print(self.list)
                else:
                    self.list = self.list[:810]#ucf 810; sht 63; 9525
                    # print('abnormal list')
                    # print(self.list)
            elif self.args.datasetname == 'XD':
                if self.is_normal:
                    self.list = self.list[1905:]
                    # print('normal list')
                    # print(self.list)
                else:
                    self.list = self.list[:1905]
                    # print('abnormal list')
                    # print(self.list)
            elif self.args.datasetname == 'ST':
                if self.is_normal:
                    self.list = self.list[63:]
                else:
                    self.list = self.list[:63]


    def __getitem__(self, index):
        label = self.get_label(index)  # get video level label 0/1
        if self.args.datasetname == 'UCF':
            features = np.load(self.list[index].strip('\n').replace('_mgfn', f'_{self.fa_model}'), allow_pickle=True)
            # features = np.random.rand(32, 768)
            features = np.array(features, dtype=np.float32)
            name = self.list[index].split('/')[-1].strip('\n')[:-4]
        elif self.args.datasetname == 'XD':
            features = np.load(self.list[index].strip('\n').replace('_mgfn', '_ours'), allow_pickle=True)
            features = np.array(features, dtype=np.float32)
            name = self.list[index].split('/')[-1].strip('\n')[:-4]
        elif self.args.datasetname == 'ST':
            features = np.load(self.list[index].strip('\n').replace('_mgfn', '_ours'), allow_pickle=True)
            features = np.array(features, dtype=np.float32)
            name = self.list[index].split('/')[-1].strip('\n')[:-4]
        
        if self.tranform is not None:
            features = self.tranform(features)
        if self.test_mode:
            if self.args.datasetname == 'UCF':
                if len(features.shape) < 3:
                    features = np.expand_dims(features, axis=1)
                mag = np.linalg.norm(features, axis=2)[:,:, np.newaxis]
                features = np.concatenate((features,mag),axis = 2)
            elif self.args.datasetname == 'XD':
                if len(features.shape) < 3:
                    features = np.expand_dims(features, axis=1)
                mag = np.linalg.norm(features, axis=2)[:,:, np.newaxis]
                features = np.concatenate((features, mag), axis=2)
            elif self.args.datasetname == 'ST':
                if len(features.shape) < 3:
                    features = np.expand_dims(features, axis=1)
                mag = np.linalg.norm(features, axis=2)[:,:, np.newaxis]
                features = np.concatenate((features,mag), axis=2)
            return features, name
        else:
            if self.args.datasetname == 'UCF':
                if len(features.shape) < 3:
                    features = np.expand_dims(features, axis=1)
                features = features.transpose(1, 0, 2)  # [10, T, F]
                divided_features = []

                divided_mag = []
                for feature in features:
                    feature = process_feat(feature, self.args.seg_length) #ucf(32,2048)
                    divided_features.append(feature)
                    divided_mag.append(np.linalg.norm(feature, axis=1)[:, np.newaxis])
                divided_features = np.array(divided_features, dtype=np.float32)
                divided_mag = np.array(divided_mag, dtype=np.float32)
                divided_features = np.concatenate((divided_features,divided_mag),axis = 2)
                return divided_features, label

            elif self.args.datasetname == 'XD':
                if len(features.shape) < 3:

                    features = np.expand_dims(features, axis=1)
                features = features.transpose(1, 0, 2)
                divided_features = []
                divided_mag = []
                for feature in features:
                    feature = process_feat(feature, self.args.seg_length)
                    divided_features.append(feature)
                    divided_mag.append(np.linalg.norm(feature, axis=1)[:, np.newaxis])
                divided_features = np.array(divided_features, dtype=np.float32)
                divided_mag = np.array(divided_mag, dtype=np.float32)
                divided_features = np.concatenate((divided_features,divided_mag),axis = 2)
                return divided_features, label

            elif self.args.datasetname == 'ST':
                if len(features.shape) < 3:
                    features = np.expand_dims(features, axis=1)
                features = features.transpose(1, 0, 2)  # [10, T, F]
                divided_features = []

                divided_mag = []
                for feature in features:
                    feature = process_feat(feature, self.args.seg_length) #ucf(32,2048)
                    divided_features.append(feature)
                    divided_mag.append(np.linalg.norm(feature, axis=1)[:, np.newaxis])
                divided_features = np.array(divided_features, dtype=np.float32)
                divided_mag = np.array(divided_mag, dtype=np.float32)
                divided_features = np.concatenate((divided_features,divided_mag),axis = 2)
                return divided_features, label



    def get_label(self, index):
        if self.is_normal:
            # label[0] = 1
            label = torch.tensor(0.0)
        else:
            label = torch.tensor(1.0)
            # label[1] = 1
        return label

    def __len__(self):

        return len(self.list)


    def get_num_frames(self):
        return self.num_frame
