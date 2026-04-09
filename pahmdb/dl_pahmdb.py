import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import config as cfg
import random
import pickle, traceback
# import parameters_BL as params
import params_pahmdb as params
import json
import math
import cv2
from tqdm import tqdm
import time
import torchvision.transforms as trans
# from decord import VideoReader

class pahmdb_loader(Dataset):
    def __init__(self, data_split = 0, shuffle = True, data_percentage = 1.0, mode = 0, skip = 1, \
            hflip=0, crop_size=None, blur= False, black = False):

        self.annots = pickle.load(open('./pahmdb/pahmdb_one_hot_parsed.pkl','rb'))
        self.all_paths = list(self.annots.keys())
        # print(self.all_paths)

        self.shuffle = shuffle

        if self.shuffle:
            random.shuffle(self.all_paths)
        
        self.data_percentage = data_percentage
        self.data_limit = int(len(self.all_paths)*self.data_percentage)
        self.data = self.all_paths[0: self.data_limit]
        self.PIL = trans.ToPILImage()
        self.TENSOR = trans.ToTensor()
        self.mode = mode
        self.skip = skip
        self.hflip = hflip
        self.crop_size = crop_size
        self.blur = blur
        self.black = black

    def __len__(self):
        return len(self.data)
            
    def __getitem__(self,index):        
        clip, label, vid_path = self.process_data(index)
        return clip, label, vid_path

    def process_data(self, idx):
    
        # label_building
        vid_path = cfg.hmdb_vid_path + '/' + self.annots[self.all_paths[idx]]["act"] + '/' + self.all_paths[idx]
        # clip_building
        clip, frame_list = self.build_clip(vid_path)
        try:
            label = self.annots[self.all_paths[idx]]["privacy"][frame_list]
        except:
            print(vid_path, self.annots[self.all_paths[idx]]["privacy"].shape[0], frame_list)
            label = [0]
        if label.shape[0] != len(clip):
            print(label.shape[0], len(clip))
        return clip, label, vid_path

    def build_clip(self, vid_path):

        try:
            cap = cv2.VideoCapture(vid_path)
            cap.set(1, 0)
            frame_count = cap.get(7)
            frame_h = cap.get(4)
            frame_w = cap.get(3)

            ############################# frame_list maker start here#################################
            # skip_max = frame_count/(params.num_frames)
            # skip_frames_full = int(skip_max/params.num_skips*self.skip)
            
            ################################ frame list maker finishes here ###########################

            ################################ actual clip builder starts here ##########################
            full_clip = []
            list_full = []
            count = -1

        
            if self.crop_size == None:
                random_size = int((frame_h + frame_w)/2)
            else:
                random_size = np.linspace(frame_h, frame_w, params.num_crops)[self.crop_size]
            
            x0 = int((frame_h-random_size)/2)
            y0 = int((frame_w-random_size)/2)

            while(cap.isOpened()):
                count += 1
                if frame_count - count <10:
                    break
                ret, frame = cap.read()
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                # if ((count not in frames_full)) and (ret == True): 
                #     continue
                if ret == True:
                    # if (count in frames_full):
                    frame = self.augmentation(frame,x0,y0, random_size)
                    frame = frame.unsqueeze(0).repeat(params.num_frames, 1, 1, 1)
                    full_clip.append(frame)
                    list_full.append(count)
                else:
                    break
            return full_clip, list_full

        except:
            print(f'Clip {vid_path} Failed')
            traceback.print_exc()
            return None, None

    def augmentation(self, image, x0,y0, random_size):
        image = self.PIL(image)
        image = trans.functional.resized_crop(image,x0,y0, random_size, random_size,(params.reso_h,params.reso_w))
        image = trans.functional.to_tensor(image)
        return image

def collate_fn1(batch):
    clip, label, vid_path = [], [], []
    for item in batch:
        if not (item[0] == None):
            clip.append(torch.stack(item[0],dim=0)) 

            label.append(item[1])
            vid_path.append(item[2])

    clip = torch.stack(clip, dim=0)

    return clip, label, vid_path
