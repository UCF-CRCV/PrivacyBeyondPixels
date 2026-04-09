fa_arch = 'transformer'
ft_arch = 'videoMAE'
anon = False
num_pa = 7
num_layers_fa = 3
num_heads = 8
saved_fa_model = None # 'saved_models/fa_mae_pahmdb_fa100fb1ft1-0-0/model_16_0.7314.pth'
saved_fb_model = None

feature_dim = 768

learning_rate_fb = 1e-4
num_workers = 0
v_batch_size = 512

num_frames = 16
reso_h = 224
reso_w = 224
num_crops = 5
