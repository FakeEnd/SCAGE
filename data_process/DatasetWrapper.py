from .loader import MoleculeDataset
import numpy as np
from torch.utils.data.sampler import SubsetRandomSampler
# from torch_geometric.loader import DataLoader
from torch.utils.data import DataLoader
from .split import create_splitter
from data_process.data_transform import TransformFn
from data_process.data_collator import collator


class PreTrainDatasetWrapper(object):
    def __init__(self, args, config_2d):
        self.args = args
        self.config_2d = config_2d

    def get_data_loaders(self):
        dataset = MoleculeDataset(self.args.root + self.args.dataset, dataset=self.args.dataset,
                                  transform=TransformFn(self.config_2d))
        splitter = create_splitter(self.args.split_type)
        train_dataset, val_dataset, _ = splitter.split(dataset, frac_train=0.8, frac_valid=0.2, frac_test=0)
        train_loader = DataLoader(train_dataset,
                                  batch_size=self.args.batch_size,
                                  shuffle=True,
                                  num_workers=0,
                                  collate_fn=lambda x: collator(x, self.config_2d),
                                  drop_last=True)
        valid_loader = DataLoader(val_dataset,
                                  batch_size=self.args.batch_size,
                                  shuffle=False,
                                  num_workers=0,
                                  collate_fn=lambda x: collator(x, self.config_2d),
                                  drop_last=True)

        return train_loader, valid_loader


class FinetuneDatasetWrapper(object):
    def __init__(self, config):
        self.config = config

    def get_data_loaders(self):
        dataset = MoleculeDataset(self.config['root']+self.config['task_name'], is_pretrain=self.config['is_pretrain'],
                                  target=self.config['target'],
                                  transform=TransformFn(self.config['model_2d']))
        splitter = create_splitter(self.config['split_type'])
        train_dataset, val_dataset, test_dataset = splitter.split(dataset, frac_train=0.8, frac_valid=0.1, frac_test=0.1)

        train_loader = DataLoader(train_dataset,
                                  batch_size=self.config['batch_size'],
                                  shuffle=True,
                                  num_workers=0,
                                  collate_fn=lambda x: collator(x, self.config['model_2d']),
                                  drop_last=True)

        valid_loader = DataLoader(val_dataset,
                                  batch_size=self.config['batch_size'],
                                  shuffle=False,
                                  num_workers=0,
                                  collate_fn=lambda x: collator(x, self.config['model_2d']),
                                  drop_last=True)

        test_loader = DataLoader(val_dataset,
                                 batch_size=self.config['batch_size'],
                                 shuffle=False,
                                 num_workers=0,
                                 collate_fn=lambda x: collator(x, self.config['model_2d']),
                                 drop_last=True)

        return train_loader, valid_loader, test_loader
