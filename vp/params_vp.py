# Run parameters.
run_id = 'vp_ucf101_mae_fa100ft1fb1'
dataset = 'ucf101'
fa_arch = 'transformer'
ft_arch = 'videoMAE'
num_classes = 101
num_pa = 5
anon = False
ssl = False
saved_fa_model = None # 'saved_models/vp_hmdb51_mae_fa0ft1fb100_ssl/model_35_acc_71.6340.pth' #'/media/jo869742/SEAGATE/FeatureAnon_saved_models/multitask_save/fa_mae_multitask_fa100fb1ft1-1-1_t3/model_58_74.58_56.14_69.00.pth'
saved_ft_model = None # 'saved_models/vp_hmdb51_mae_fa0ft1fb100_ssl/model_35_acc_71.6340.pth'
fa_trainable = False

# Training parameters.
num_epochs = 100
batch_size = 512
v_batch_size = 512
val_freq = 5
learning_rate = 1e-3
num_workers = 4
feature_dim = 768
num_layers_fa = 3
num_heads = 8

fb_loss_weight = 5.0
fa_loss_weight = 0.0

# Validation parameters.
num_modes = 5
