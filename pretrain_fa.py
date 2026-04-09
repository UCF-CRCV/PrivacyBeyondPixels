import numpy as np
import os.path
import torch
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler

from feature_dl import KineticsFeaturesDataset, UCFFeaturesDataset, VISPRFeaturesDataset, HMDBFeaturesDataset
from model_loaders import load_fa, load_backbone


if __name__ == '__main__':
    import params.params_fa as params
    params.ar_dataset = 'hmdb51'
    params.saved_fa_model = None
    params.fa_arch = 'transformer'
    params.num_layers_fa = 3
    params.feature_dim = 2048
    params.num_heads = 8
    params.ft_arch = 'largei3d'
    fa_model = load_fa(params=params)
    # backbone = load_backbone(params)
    # criterion = torch.nn.L1Loss()
    criterion = torch.nn.MSELoss()
    fa_model.cuda()
    criterion.cuda()

    print(f'Pretraining: {params.fa_arch} with {params.num_layers_fa} layers and {params.num_heads} heads.')
    # backbone.cuda()
    # for param in backbone.parameters():
    #     param.requires_grad = False

    optimizer = torch.optim.AdamW(fa_model.parameters(), lr=1e-4)
    scaler = GradScaler()

    # train_dataset = KineticsFeaturesDataset(params, split='train')
    # test_dataset = KineticsFeaturesDataset(params, split='test')
    
    # train_dataset = UCFFeaturesDataset(params, split='train')
    # test_dataset = UCFFeaturesDataset(params, split='test')
    
    train_dataset = HMDBFeaturesDataset(params, split='train')
    test_dataset = HMDBFeaturesDataset(params, split='test')
    
    train_loader = DataLoader(train_dataset, batch_size=2048, shuffle=True, num_workers=8)
    test_loader = DataLoader(test_dataset, batch_size=2048, shuffle=False, num_workers=8, pin_memory=True)
    
    val_loss_best = 1000

    for epoch in range(500):
        # backbone.train()
        fa_model.train()
        losses = []
        with autocast():
            # for batch_idx, (inputs, _, _, _) in enumerate(train_loader):
            for batch_idx, (inputs, _, _) in enumerate(train_loader):
                inputs = inputs.cuda()
                optimizer.zero_grad()
                # features = backbone(inputs).logits
                features = inputs
                output = fa_model(features)
                loss = criterion(output, features)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                # loss.backward()
                # optimizer.step()
        #     losses.append(loss.item())
        # train_loss = np.mean(losses)

        if epoch % 5 == 0:
            fa_model.eval()
            # backbone.eval()
            losses = []
            # for batch_idx, (inputs, _, _, _) in enumerate(test_loader):
            for batch_idx, (inputs, _, _) in enumerate(test_loader):
                inputs = inputs.cuda()
                with torch.no_grad():
                    # features = backbone(inputs).logits
                    features = inputs
                    output = fa_model(features)
                    loss = criterion(output, features)
                    losses.append(loss.item())
            val_loss = np.mean(losses)
            print(f'Epoch: {epoch}, Loss: {val_loss:.5f}')

            if val_loss < val_loss_best:
                val_loss_best = val_loss
                torch.save({'fa_model_state_dict': fa_model.state_dict()}, os.path.join('saved_models', f'{params.ft_arch}_fa_{params.fa_arch}_recon_{params.num_layers_fa}_layers.pth'))  
        

    print(f'Best validation loss: {val_loss_best:.5f}')
