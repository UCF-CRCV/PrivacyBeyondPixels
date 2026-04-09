anon = False
ft_arch = 'videoMAE'
saved_ft_model = '../../action_recognition/FeatureAnonymizer/saved_models/ft_ntu_mae_baseline/model_250_acc_51.4931.pth'
num_layers_fa = 3
feature_dim = 768
num_heads = 8
fa_arch = 'transformer'
saved_fa_model = None

dataset = 'ntu'
split = 'test'
num_classes = 60
num_modes = 1
v_batch_size = 16
num_workers = 4
