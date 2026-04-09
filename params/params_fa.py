run_id = 'fa_mae_hmdb51_fa100fb1ft1-0-0'
ar_dataset = 'hmdb51'
fa_arch = 'transformer'
ft_arch = 'videoMAE'
feature_dim = 768
num_classes = 51
num_layers_fa = 3
num_heads = 8
saved_fa_model = f'saved_models/mae_fa_{fa_arch}_recon_{num_layers_fa}_layers.pth'
saved_ft_model = None
saved_ft_tad_model = 'tridet/mae_thumos_baseline.pth.tar'
saved_ft_ad_model = 'mgfn/mae_crimes_baseline.pkl'

num_epochs = 100
batch_size = 32
batch_size_ar = 512
batch_size_ad = 8
v_batch_size = 512
val_freq = 2
num_workers = 4

learning_rate_fa = 1e-4
learning_rate_fb = 1e-4
learning_rate_ft = 1e-4
learning_rate_ft_ad = 1e-3

fa_loss_weight = 100.0
ft_loss_weight = 1.0
ft_ar_loss_weight = 1.0
ft_tad_loss_weight = 1.0
ft_ad_loss_weight = 1.0
fb_loss_weight = 1.0
fb_frame_pool = 10
fb_frame_sample = 2

num_modes = 5

num_frames = 16
reso_h = 224
reso_w = 224

wandb = False
