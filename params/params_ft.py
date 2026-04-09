run_id = 'ft_k400_mae_baseline'
dataset = 'k400'
ar_dataset = 'k400'
fa_arch = 'transformer'
ft_arch = 'videoMAE'
feature_dim = 768
num_classes = 400
anon = False
fa_trainable = False
saved_fa_model = None
saved_ft_model = None

num_epochs = 250
batch_size = 1024
v_batch_size = 1024
val_freq = 2
learning_rate = 1e-3
num_workers = 4
num_layers_fa = 3
num_heads = 8

num_modes = 5
