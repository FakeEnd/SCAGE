import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3'
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
import yaml
import shutil
import numpy as np
from datetime import datetime
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from utils.public_util import set_seed
from utils.loss_util import NTXentLoss
from utils.scheduler_util import *

from data_process.loader import PretrainDataset
from data_process.data_transform import MaskTransformFn
from data_process.data_collator import pretrain_collator
from models.pretrain_model import MolGraphCL
import warnings
warnings.filterwarnings("ignore")


def write_record(path, message):
    file_obj = open(path, 'a')
    file_obj.write('{}\n'.format(message))
    file_obj.close()


def copyfile(srcfile, path):
    if not os.path.isfile(srcfile):
        print(f"{srcfile} not exist!")
    else:
        fpath, fname = os.path.split(srcfile)
        if not os.path.exists(path):
            os.makedirs(path)
        shutil.copy(srcfile, os.path.join(path, fname))


class PreTrainer(object):
    def __init__(self, config, file_path):
        self.config = config

        self.loader = self.get_data_loaders()
        self.model = self._get_net()
        self.criterion = self._get_loss_fn()
        self.model_optim = self._get_optim()
        self.lr_scheduler = self._get_lr_scheduler()

        if config['checkpoint']:
            self.load_ckpt(self.config['checkpoint'])
        else:
            self.start_epoch = 1
            self.optim_steps = 0
            self.best_loss = np.inf
            self.writer = SummaryWriter('{}/{}/{}_{}_{}'.format(
                'pretrain_result', 'cl_result', config['task_name'], config['seed'],
                datetime.now().strftime('%b%d_%H:%M')))
        self.txtfile = os.path.join(self.writer.log_dir, 'record.txt')
        copyfile(file_path, self.writer.log_dir)

    def get_data_loaders(self):
        dataset = PretrainDataset(self.config['root'] + self.config['task_name'],
                                transform=MaskTransformFn(self.config))
        print('all_dataset_num:', len(dataset))
        loader = DataLoader(dataset,
                            batch_size=self.config['batch_size'],
                            shuffle=True,
                            num_workers=24,
                            collate_fn=lambda x: pretrain_collator(x, self.config['model']),
                            drop_last=False)

        return loader

    def _get_net(self):
        model = MolGraphCL(self.config['model'])

        model = model.to('cuda')
        if self.config['DP']:
            model = torch.nn.DataParallel(model)

        return model

    def _get_loss_fn(self):
        loss_type = self.config['loss']['type']
        if loss_type == 'NTXentLoss':
            return NTXentLoss(self.config['batch_size'], **self.config['loss']['param'])

    def _get_optim(self):
        optim_type = self.config['optim']['type']
        lr = self.config['optim']['init_lr']
        weight_decay = eval(self.config['optim']['weight_decay'])

        if optim_type == 'adam':
            model_optimizer = torch.optim.Adam(self.model.parameters(), lr=lr, weight_decay=weight_decay)
            return model_optimizer
        elif optim_type == 'rms':
            model_optimizer = torch.optim.RMSprop(self.model.parameters(), lr=lr, weight_decay=weight_decay)
            return model_optimizer
        elif optim_type == 'sgd':
            model_optimizer = torch.optim.SGD(self.model.parameters(), lr=lr, weight_decay=weight_decay)
            return model_optimizer
        else:
            raise ValueError('not supported optimizer!')

    def _get_lr_scheduler(self):
        scheduler_type = self.config['lr_scheduler']['type']
        init_lr = self.config['lr_scheduler']['start_lr']
        warm_up_epoch = self.config['lr_scheduler']['warm_up_epoch']

        if scheduler_type == 'linear':
            return LinearSche(self.config['epochs'], warm_up_end_lr=self.config['optim']['init_lr'], init_lr=init_lr,
                              warm_up_epoch=warm_up_epoch)
        elif scheduler_type == 'square':
            return SquareSche(self.config['epochs'], warm_up_end_lr=self.config['optim']['init_lr'], init_lr=init_lr,
                              warm_up_epoch=warm_up_epoch)
        elif scheduler_type == 'cos':
            return CosSche(self.config['epochs'], warm_up_end_lr=self.config['optim']['init_lr'], init_lr=init_lr,
                           warm_up_epoch=warm_up_epoch)
        elif scheduler_type == 'None':
            return None
        else:
            raise ValueError('not supported learning rate scheduler!')

    def _train_step(self):
        self.model.train()
        model_loss_all = 0
        print('batch_all_num:', len(self.loader))
        for batch_raw, batch_mask in tqdm(self.loader, leave=False, ascii=True):
            self.model_optim.zero_grad()
            batch_raw = {key: value.to('cuda') for key, value in batch_raw.items()
                         if value is not None and not isinstance(value, list)}
            batch_mask = {key: value.to('cuda') for key, value in batch_mask.items()
                          if value is not None and not isinstance(value, list)}
            batch_raw['edge_weight'], batch_mask['edge_weight'] = None, None

            mol_rep_raw = self.model(batch_raw)
            mol_rep_aug = self.model(batch_mask)
            if self.config['DP']:
                model_loss = self.model.module.loss_cl(mol_rep_raw, mol_rep_aug)
            else:
                model_loss = self.model.loss_cl(mol_rep_raw, mol_rep_aug)

            model_loss_all += model_loss.item()
            model_loss.backward()
            self.model_optim.step()
            self.writer.add_scalar('model_loss', model_loss, global_step=self.optim_steps)
            self.optim_steps += 1

        model_loss_mean = model_loss_all / len(self.loader)
        return model_loss_mean

    def save_ckpt(self, epoch):
        model_dict = {
            'model': self.model.state_dict()
        }
        checkpoint = {
            "net": model_dict,
            'model_optim': self.model_optim.state_dict(),
            "epoch": epoch,
            'best_loss': self.best_loss,
            'optim_steps': self.optim_steps
        }
        path = os.path.join(self.writer.log_dir, 'checkpoint')
        os.makedirs(path, exist_ok=True)
        torch.save(checkpoint, os.path.join(path, 'model_{}.pth'.format(epoch)))

    def load_ckpt(self, load_pth):
        checkpoint = torch.load(load_pth, map_location='cuda')
        self.writer = SummaryWriter(os.path.dirname(load_pth))
        self.model.load_state_dict(checkpoint['net']['model'])
        self.model_optim.load_state_dict(checkpoint['model_optim'])
        self.start_epoch = checkpoint['epoch'] + 1
        self.best_loss = checkpoint['best_loss']
        self.optim_steps = checkpoint['optim_steps']

    def train(self):
        # print(self.config)
        write_record(self.txtfile, self.config)
        for i in range(self.start_epoch, self.config['epochs'] + 1):
            if self.config['lr_scheduler']['type'] in ['cos', 'square', 'linear']:
                self.lr_scheduler.adjust_lr(self.model_optim, i)

            model_loss = self._train_step()
            # 保存模型
            if model_loss < self.best_loss:
                self.best_loss = model_loss
                if self.config['DP']:
                    model_dict = {'model': self.model.module.state_dict()}
                    torch.save(model_dict, os.path.join(self.writer.log_dir, 'model.pth'))
                else:
                    model_dict = {'model': self.model.state_dict()}
                    torch.save(model_dict, os.path.join(self.writer.log_dir, 'model.pth'))
            # 每间隔i个epoch进行保存
            if i % self.config['save_ckpt'] == 0:
                self.save_ckpt(i)

            print(f'Epoch:{i} model_loss:{model_loss}')
            write_record(self.txtfile, f'Epoch:{i} model_loss:{model_loss}')


if __name__ == '__main__':
    path = "config/config_pretrain_cl.yaml"
    config = yaml.load(open(path, "r"), Loader=yaml.FullLoader)
    print(config['task_name'])
    set_seed(config['seed'])
    pre_trainer = PreTrainer(config, path)
    pre_trainer.train()
