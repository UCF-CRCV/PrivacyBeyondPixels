import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import *
import time, random
import numpy as np
from model_loaders import load_fa, load_fb, load_backbone
import params_pahmdb as params
import config as cfg
from dl_pahmdb import *
import traceback
from sklearn.metrics import precision_recall_fscore_support, average_precision_score
from tensorboardX import SummaryWriter
import cv2
from torch.utils.data import DataLoader
import math
import argparse
import itertools
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning) 


def val_epoch(run_id, epoch, validation_dataloader, fa_model, fb_model, backbone, criterion, writer, use_cuda, setting):
    
    fa_model.eval()
    fb_model.eval()
    losses = []
    predictions, ground_truth = [], []
    vid_paths = []
    label_dict, pred_dict = {}, {}

    for i, (inputs, label, vid_path) in enumerate(validation_dataloader):
        # inputs = inputs.permute(0,4,1,2,3)
        if len(inputs.shape) != 1:

            # inputs = inputs.permute(0, 2, 1, 3, 4)

            if use_cuda:
                inputs = inputs.cuda().squeeze(0)
                label = torch.from_numpy(np.asarray(label)).float().cuda().squeeze(0)
        
            # print(inputs.shape)
            with torch.no_grad():
                # Process one video at a time.
                all_input = []
                for input in inputs:
                    all_input.append(backbone(input.unsqueeze(0)).logits)
                inputs = torch.cat(all_input, dim=0)
                if params.anon:
                    inputs = fa_model(inputs)
                output = fb_model(inputs)[:,:-1]

                loss = criterion(output,label)
                losses.append(loss.item())


            predictions.extend(output.cpu().data.numpy())
            vid_paths.extend(vid_path)
            ground_truth.extend(label.cpu().data.numpy())

            # print(len(predictions))


            if i % 100 == 0:
                print("Validation Epoch ", epoch, " Batch ", i, "- Loss : ", np.mean(losses))
        
    del inputs, output, label, loss 
    print("Validation Epoch ", epoch, "- Loss : ", np.mean(losses))

    ground_truth = np.asarray(ground_truth)

    prec, recall, f1, _ = precision_recall_fscore_support(ground_truth, (np.array(predictions) > 0.5).astype(int))
    predictions = np.asarray(predictions)

    # ap = average_precision_score(ground_truth, predictions)
    try:
        print('gt shape ', ground_truth.shape)
        print('pred shape ',predictions.shape)
    except:
        print('gt length ', len(ground_truth))
        print('pred length ', len(predictions))

    ap = average_precision_score(ground_truth, predictions, average=None)

    print(f'Macro f1 is {np.mean(f1)}')
    print(f'Macro prec is {np.mean(prec)}')
    print(f'Macro recall is {np.mean(recall)}')
    print(f'Classwise AP is {ap}')
    print(f'Macro AP is {np.mean(ap)}')


    for entry in range(len(vid_paths)):
        if str(vid_paths[entry].split('/')[-1]) not in pred_dict.keys():
            pred_dict[str(vid_paths[entry].split('/')[-1])] = []
            pred_dict[str(vid_paths[entry].split('/')[-1])].append(predictions[entry])

        else:
            # print('yes')
            pred_dict[str(vid_paths[entry].split('/')[-1])].append(predictions[entry])

    for entry in range(len(vid_paths)):
        if str(vid_paths[entry].split('/')[-1]) not in label_dict.keys():
            label_dict[str(vid_paths[entry].split('/')[-1])]= ground_truth[entry]

    return pred_dict, label_dict, np.mean(ap)
    
def train_classifier(run_id, restart, saved_model,fb_arch, setting, downsample, just_eval_raw, imagenet_pretrained_fb):
    print('fb architecture: ', fb_arch)
    print(f'Input reso: {params.reso_h}')
    print(f'Setting: {setting}')

    use_cuda = True
    best_score = 0
    writer = SummaryWriter(os.path.join(cfg.logs, str(run_id)))
    blur = False
    black = False
    save_dir = os.path.join(cfg.saved_models_dir, run_id)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    fa_model = load_fa(params)
    backbone = load_backbone(params)
    fb_model = load_fb(initial_embedding_size=params.feature_dim, final_embedding_size=params.num_pa, saved_model=params.saved_fb_model)
    

    for param in fa_model.parameters():
        param.requires_grad = False
    

    epoch0 = 0        

    learning_rate1 = params.learning_rate_fb
    
    criterion = nn.BCEWithLogitsLoss()
    if torch.cuda.device_count()>1:
        print(f'Multiple GPUS found!')
        fa_model=nn.DataParallel(fa_model)
        fb_model=nn.DataParallel(fb_model)
        criterion.cuda()
        fb_model.cuda()
        fa_model.cuda()
    else:
        print('Only 1 GPU is available')
        criterion.cuda()
        backbone.cuda()
        fa_model.cuda()
        fb_model.cuda()


    optimizer = optim.Adam(list(fb_model.parameters()),lr=params.learning_rate_fb)
    val_array = [0,5, 10,15, 20, 25, 30, 35, 40, 45] + [40+ x for x in range(100)]


    print(f'Base learning rate {params.learning_rate_fb}')
    
    accuracy = 0
    lr_flag1 = 0
    lr_counter = 0
    train_loss_prev = 1000
    # lr_array = [0.001, 0.01, 0.1, 1] + [1 for x in range(15)] + [0.5 for x in range(15)] +  [0.1 for x in range(30)] + [0.05 for x in range(20)] + [0.01 for x in range(25)]
    # lr_array = np.asarray(lr_array)*learning_rate1
    learning_rate1 = params.learning_rate_fb
    train_loss_best = 1000

    for epoch in range(1):
       
        print(f'Epoch {epoch} started')
        start=time.time()
        try:
            
            if epoch in val_array:
                pred_dict = {}
                label_dict = {}
                val_losses =[]

                validation_dataset = pahmdb_loader(blur= blur, black =black)

                validation_dataloader = DataLoader(validation_dataset, batch_size=1, shuffle=True, num_workers=params.num_workers, collate_fn=collate_fn1)
                print(f'Validation dataset length: {len(validation_dataset)}')
                print(f'Validation dataset steps per epoch: {len(validation_dataset)/params.v_batch_size}') 
                pred_dict, label_dict, macro_ap = val_epoch(run_id, epoch, validation_dataloader, fa_model, fb_model, backbone, criterion, writer, use_cuda, setting)
                
               
                # file_name = f'RunID_{run_id}_Acc_{accuracy1*100 :.3f}_cf_{len(cropping_fac1)}_m_{params.num_modes}_s_{params.num_skips}.pkl'     
                # pickle.dump(pred_dict, open(file_name,'wb'))
            if macro_ap > best_score:
                print('++++++++++++++++++++++++++++++')
                print(f'Epoch {epoch} is the best model till now for {run_id}!')
                print('++++++++++++++++++++++++++++++')
                save_dir = os.path.join(cfg.saved_models_dir, run_id)
                if not os.path.exists(save_dir):
                    os.makedirs(save_dir)
                save_file_path = os.path.join(save_dir, 'model_{}_bestAP_{}.pth'.format(epoch, str(macro_ap)[:6]))
                states = {
                    'epoch': epoch + 1,
                    'lr_counter' : lr_counter,
                    # 'arch': params.arch,
                    'fb_model_state_dict': fb_model.state_dict(),
                    'pred_dict': pred_dict,
                    'label_dict': label_dict,
                }
                torch.save(states, save_file_path)
                best_score = macro_ap
            # else:
            save_dir = os.path.join(cfg.saved_models_dir, run_id)
            save_file_path = os.path.join(save_dir, 'model_temp.pth')
            states = {
                'epoch': epoch + 1,
                'lr_counter' : lr_counter,
                # 'arch': params.arch,
                'fb_model_state_dict': fb_model.state_dict(),
                'pred_dict': pred_dict,
                'label_dict': label_dict,
                }
            torch.save(states, save_file_path)
        except:
            print("Epoch ", epoch, " failed")
            print('-'*60)
            traceback.print_exc(file=sys.stdout)
            print('-'*60)
            continue
        taken = time.time()-start
        print(f'Time taken for Epoch-{epoch} is {taken}')
        print()


        

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Script to train baseline')

    parser.add_argument("--run_id", dest='run_id', type=str, required=False, default= "dummy_privacy",
                        help='run_id')
    parser.add_argument("--restart", action='store_true')
    parser.add_argument("--saved_model", dest='saved_model', type=str, required=False, default= None,
                        help='run_id')
    parser.add_argument("--fb_arch", dest='fb_arch', type=str, required=False, default='mv1',
                        help='fb_arch: mv1 or r50')
    parser.add_argument("--setting", dest='setting', type=str, required=False, default= None,
                        help='ours, wu, bl')  
    parser.add_argument("--downsample", dest='downsample', type=int, required=False, default= 1,
                        help='1,2,4,8') 
    parser.add_argument("--just_eval_raw", action='store_true', help='evaluation on raw data')
    parser.add_argument("--imagenet_pretrained_fb", action='store_true', help='ImageNet pretrained and VISPR1 finetuned raw data powerful classifier')
                            
    args = parser.parse_args()
    print(f'Restart {args.restart}')

    run_id = args.run_id
    fb_arch = args.fb_arch
    setting = args.setting
    downsample = args.downsample
    print(f'downsample {args.downsample}')
    print(f'imagenet_pretrained_fb {args.imagenet_pretrained_fb}')

    saved_model = args.saved_model
    if args.just_eval_raw:
        print('privacy evluation on raw pretrained classifier')
    train_classifier(str(run_id), args.restart, saved_model, fb_arch, setting, 1/downsample, args.just_eval_raw, args.imagenet_pretrained_fb)


        


