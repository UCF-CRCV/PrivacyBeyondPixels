import os
from pprint import pprint

from tridet.core import load_config
from tridet.datasets import make_dataset, make_data_loader
from tridet.utils import fix_random_seed


def load_thumos(config, split='train'):
    if os.path.isfile(config):
        cfg = load_config(config)
    else:
        raise ValueError("Config file does not exist.")
    # pprint(cfg)

    if split == 'train':
        """1. fix random seeds"""
        # fix the random seeds (this will fix everything)
        rng_generator = fix_random_seed(cfg['init_rand_seed'], include_cuda=True)

        # re-scale learning rate / # workers based on number of GPUs
        # cfg['opt']["learning_rate"] *= len(cfg['devices'])
        # cfg['loader']['num_workers'] *= len(cfg['devices'])
        """2. create dataset / dataloader"""
        train_dataset = make_dataset(
            cfg['dataset_name'], True, cfg['train_split'], **cfg['dataset']
        )
        # update cfg based on dataset attributes (fix to epic-kitchens)
        train_db_vars = train_dataset.get_attributes()
        cfg['model']['train_cfg']['head_empty_cls'] = train_db_vars['empty_label_ids']

        # data loaders
        train_loader = make_data_loader(
            train_dataset, True, rng_generator, **cfg['loader'])
        
        return train_loader
    else:
        # fix the random seeds (this will fix everything)
        _ = fix_random_seed(0, include_cuda=True)

        """2. create dataset / dataloader"""
        val_dataset = make_dataset(
            cfg['dataset_name'], False, cfg['val_split'], **cfg['dataset']
        )
        # set bs = 1, and disable shuffle
        val_loader = make_data_loader(
            val_dataset, False, None, 1, cfg['loader']['num_workers']
        )

        return val_loader

    

if __name__ == '__main__':
    config = 'tridet/thumos_vjepa.yaml'
    load_thumos(config)
