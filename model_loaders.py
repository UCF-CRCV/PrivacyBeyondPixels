import os.path
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import VideoMAEForVideoClassification

from models.anonymizer import TransformerAnonymizer
from models.i3d import InceptionI3d

from tridet.core import load_config
from tridet.modeling import make_meta_arch
from tridet.utils import ModelEma, make_optimizer, make_scheduler

from mgfn.models.mgfn import mgfn


# Load full model.
def load_full(params, saved_model=None, arch=None):
    if arch is None:
        arch = params.ft_arch
    if saved_model is None:
        saved_model = params.saved_ft_model
    if arch == 'videoMAE':
        backbone = VideoMAEForVideoClassification.from_pretrained('MCG-NJU/videomae-base-finetuned-kinetics')
        head = backbone.classifier
        if params.num_classes != 400:
            head = nn.Linear(backbone.config.hidden_size, params.num_classes, bias=True)
        backbone.classifier = nn.Identity()
        model = FullModelHF(backbone, head, params.proj_dim)
    elif arch == 'i3d':
        backbone = InceptionI3d(num_classes=400)
        msg = backbone.load_state_dict(torch.load('saved_models/rgb_imagenet.pt'), strict=True)
        print(msg)
        if params.num_classes != 400:
            backbone.replace_logits(params.num_classes)
        head = backbone.logits
        backbone.logits = nn.Identity()
        model = FullModelHF(backbone, head, params.proj_dim)
    else:
        raise ValueError(f'Unknown architecture: {arch}')
    
    if saved_model is not None:
        state_dict = torch.load(saved_model)
        model.head.load_state_dict(state_dict['ft_model_state_dict'], strict=True)
        print(f'Ft model loaded from: {saved_model}')
    return model


# Load full HuggingFace model.
class FullModelHF(nn.Module):
    def __init__(self, backbone, head, proj_dim):
        super(FullModelHF, self).__init__()
        self.backbone = backbone
        self.head = head
        self.proj = MLP(backbone.config.hidden_size, proj_dim, use_normalization=True) if proj_dim > 0 else nn.Identity()

    def forward(self, x, fa_model=None, just_feats=False):
        f = self.backbone(x).logits
        f = fa_model(f) if fa_model is not None else f
        if just_feats:
            return f
        pred = self.head(f)
        f_proj = self.proj(f)
        return pred, f_proj, f
    

# Load backbone model.
def load_backbone(params):
    if params.ft_arch == 'videoMAE':
        backbone = VideoMAEForVideoClassification.from_pretrained('MCG-NJU/videomae-base-finetuned-kinetics', image_size=params.reso_h, output_hidden_states=True)
        backbone.classifier = nn.Identity()
    elif params.ft_arch == 'i3d':
        backbone = InceptionI3d(num_classes=400)
        msg = backbone.load_state_dict(torch.load('saved_models/rgb_imagenet.pt'), strict=True)
        print(msg)
        backbone.logits = nn.Identity()
    else:
        raise ValueError(f'Unknown architecture: {params.ft_arch}')
    return backbone

# Load MLP annonimization model.
def load_fa(params, saved_model=None, arch=None):
    if saved_model is None or saved_model == 'none':
        saved_model = params.saved_fa_model
    if arch is None:
        arch = params.fa_arch
    
    if arch == 'mlp':
        model = MLP(params.feature_dim)
    elif arch == 'transformer':
        model = TransformerAnonymizer(params.feature_dim, params.num_heads, params.num_layers_fa)
    else:
        raise ValueError(f'Unknown architecture: {arch}')
    
    if saved_model is not None:
        state_dict = torch.load(saved_model)
        model.load_state_dict(state_dict['fa_model_state_dict'], strict=True)
        print(f'Fa model loaded from: {saved_model}')
    return model


# Load fb privacy prediction model.
def load_fb(saved_model=None, arch='mlp', initial_embedding_size=768, final_embedding_size=128):
    # MLP layer.
    class MLP(nn.Module):
        def __init__(self, initial_embedding_size=768, final_embedding_size=128, use_normalization=True):
            super(MLP, self).__init__()
            self.initial_embedding_size = initial_embedding_size
            self.final_embedding_size = final_embedding_size
            self.use_normalization = use_normalization
            self.fc1 = nn.Linear(self.initial_embedding_size, self.initial_embedding_size, bias=True)
            self.relu = nn.ReLU(inplace=True)
            self.fc2 = nn.Linear(self.initial_embedding_size, self.final_embedding_size, bias=True)

        def forward(self, x):
            x = self.relu(self.fc1(x))
            x = nn.functional.normalize(self.fc2(x), p=2, dim=1)
            return x

    if arch == 'mlp':
        model = MLP(initial_embedding_size=initial_embedding_size, final_embedding_size=final_embedding_size)
    else:
        raise ValueError(f'Unknown architecture: {arch}')
    
    if saved_model is not None:
        state_dict = torch.load(saved_model)
        model.load_state_dict(state_dict['fb_model_state_dict'], strict=True)
        print(f'Fb model loaded from: {saved_model}')
    return model


# Load video classifier.
def load_ft(params, saved_model=None, arch=None):
    if arch is None:
        arch = params.ft_arch
    if saved_model is None:
        saved_model = params.saved_ft_model
    if arch == 'videoMAE':
        load_model = VideoMAEForVideoClassification.from_pretrained('MCG-NJU/videomae-base-finetuned-kinetics')
        model = load_model.classifier
        if params.num_classes != 400:
            model = nn.Linear(load_model.config.hidden_size, params.num_classes, bias=True)
    elif arch == 'vjepa':
        model = nn.Linear(1280, params.num_classes, bias=True)
    elif arch == 'maev2':
        model = nn.Linear(params.feature_dim, params.num_classes, bias=True)
    elif arch == 'i3d':
        model = nn.Linear(1024, params.num_classes, bias=True)
    elif arch == 'largei3d':
        model = nn.Linear(2048, params.num_classes, bias=True)
    else:
        raise ValueError(f'Unknown architecture: {arch}')
    
    if saved_model is not None:
        state_dict = torch.load(saved_model)
        model.load_state_dict(state_dict['ft_model_state_dict'], strict=True)
        print(f'Ft model loaded from: {saved_model}')
    return model


# MLP layer.
class MLP(nn.Module):
    def __init__(self, initial_embedding_size=768, final_embedding_size=128, use_normalization=True):
        super(MLP, self).__init__()
        self.initial_embedding_size = initial_embedding_size
        self.final_embedding_size = final_embedding_size
        self.use_normalization = use_normalization
        self.fc1 = nn.Linear(self.initial_embedding_size, self.initial_embedding_size, bias=True)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(self.initial_embedding_size, self.final_embedding_size, bias=True)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = nn.functional.normalize(self.fc2(x), p=2, dim=1)
        return x
    

def load_model_tridet(checkpoint, train_loader, arch):
    cfg = load_config(f'tridet/thumos_{arch}.yaml')
    # model
    model = make_meta_arch(cfg['model_name'], **cfg['model'])
    # not ideal for multi GPU training, ok for now
    model = torch.nn.DataParallel(model, device_ids=cfg['devices'])
    # optimizer
    optimizer = make_optimizer(model, cfg['opt'])
    # schedule
    num_iters_per_epoch = len(train_loader)
    scheduler = make_scheduler(optimizer, cfg['opt'], num_iters_per_epoch)

    # enable model EMA
    model_ema = ModelEma(model)

    """4. Resume from model / Misc"""
    # resume from a checkpoint?
    if checkpoint:
        if os.path.isfile(checkpoint):
            device = torch.device("cuda:0")
            # load ckpt, reset epoch / best rmse
            # ckpt = torch.load(checkpoint,
            #                         map_location=lambda storage, loc: storage.cuda(
            #                             cfg['devices'][0]))
            ckpt = torch.load(checkpoint, map_location=device)
            start_epoch = ckpt['epoch'] + 1
            model.load_state_dict(ckpt['state_dict'])
            model_ema.module.load_state_dict(ckpt['state_dict_ema'])
            # also load the optimizer / scheduler if necessary
            optimizer.load_state_dict(ckpt['optimizer'])
            scheduler.load_state_dict(ckpt['scheduler'])
            # print("=> loaded checkpoint '{:s}' (epoch {:d}".format(
            #     checkpoint, ckpt['epoch']
            # ))
            del ckpt
        else:
            print("=> no checkpoint found at '{}'".format(checkpoint))
            return
        
    return model, model_ema, optimizer, scheduler


def load_model_mgfn(checkpoint):
    model = mgfn()
    if checkpoint is not None:
        model_ckpt = torch.load(checkpoint)
        model.load_state_dict(model_ckpt)
        print(f'Loaded MGFN model from checkpoint: {os.path.basename(checkpoint)}')

    return model
