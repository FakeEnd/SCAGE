# !/usr/bin/python3
# --coding:utf-8--
# @File: pretrain.py
# @Author:junru jin
# @Time: 2023年11月12日06
# @description:

import argparse
import os
import time

import yaml
import shutil
import numpy as np
from datetime import datetime
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torchmetrics

import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')

from utils.public_util import set_seed
from utils.loss_util import NTXentLoss
from utils.training_util import write_record, copyfile
from utils.scheduler_util import *

from data_process.loader import PretrainDataset
from data_process.data_transform import MaskTransformFn
from data_process.data_collator import pretrain_collator
from models.pretrain_model import MolGraphCL, ViewLearner, reparame_trick, regular_trick
import warnings


os.environ['CUDA_VISIBLE_DEVICES'] = '1'
# os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
warnings.filterwarnings("ignore")


class PreTrainer(object):
    def __init__(self, config, file_path):
        self.config = config

        self.loader = self.get_data_loaders()
        self.model, self.view = self._get_net()
        self.criterion = self._get_loss_fn()
        self.model_optim, self.view_optim = self._get_optim()
        self.lr_scheduler = self._get_lr_scheduler()

        if config['checkpoint']:
            print('loading check point')
            self.load_ckpt(self.config['checkpoint'])
        else:
            self.start_epoch = 1
            self.optim_steps = 0
            self.best_loss = np.inf
            self.writer = SummaryWriter('{}/{}/{}/{}_{}_{}'.format(
                'pretrain_model', self.config['pretrain_task'], self.config['pretrain_mode'], config['task_name'], config['seed'],
                datetime.now().strftime('%b%d_%H_%M')))
        self.txtfile = os.path.join(self.writer.log_dir, 'record.txt')
        copyfile(file_path, self.writer.log_dir)

    def get_data_loaders(self):
        # dataset = PretrainDataset(self.config['root'] + self.config['task_name'],
        #                         transform=MaskTransformFn(self.config))
        dataset = PretrainDataset(self.config['root'] + self.config['task_name'])
        print('all_dataset_num:', len(dataset))
        loader = DataLoader(dataset,
                            batch_size=self.config['batch_size'],
                            shuffle=True,
                            prefetch_factor=16,
                            pin_memory=True,
                            num_workers=16,
                            collate_fn=lambda x: pretrain_collator(x, self.config['model']),
                            drop_last=False)

        return loader

    def _get_net(self):
        model = MolGraphCL(self.config['model'])
        view_learner = ViewLearner(self.config['model'])

        model = model.to('cuda')
        view_learner = view_learner.to('cuda')
        if self.config['DP']:
            model = torch.nn.DataParallel(model)
            view_learner = torch.nn.DataParallel(view_learner)

        return model, view_learner

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
            view_optimizer = torch.optim.Adam(self.view.parameters(), lr=lr, weight_decay=weight_decay)
            return model_optimizer, view_optimizer
        elif optim_type == 'rms':
            model_optimizer = torch.optim.RMSprop(self.model.parameters(), lr=lr, weight_decay=weight_decay)
            view_optimizer = torch.optim.RMSprop(self.view.parameters(), lr=lr, weight_decay=weight_decay)
            return model_optimizer, view_optimizer
        elif optim_type == 'sgd':
            model_optimizer = torch.optim.SGD(self.model.parameters(), lr=lr, weight_decay=weight_decay)
            view_optimizer = torch.optim.SGD(self.view.parameters(), lr=lr, weight_decay=weight_decay)
            return model_optimizer, view_optimizer
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
        view_loss_all = 0
        model_loss_all = 0
        reg_all = 0
        batch_index = 0

        for batch_raw, batch_mask in tqdm(self.loader, leave=False, ascii=True):
            batch_raw = {key: value.to('cuda') for key, value in batch_raw.items()
                         if value is not None and not isinstance(value, list)}
            batch_mask = {key: value.to('cuda') for key, value in batch_mask.items()
                          if value is not None and not isinstance(value, list)}

            if self.config['pretrain_mode'] == 'ad':
                self.view.train()
                self.view_optim.zero_grad()
                self.model.eval()

                batch_raw['edge_weight'] = None
                mol_rep_raw, atom_rep, fg_out = self.model(batch_raw)
                edge_logit = self.view(batch_raw, atom_rep)  # 计算边的概率
                batch_aug_edge_weight = reparame_trick(edge_logit)  # 重参数化

                batch_mask['edge_weight'] = batch_aug_edge_weight
                mol_rep_aug, _, _ = self.model(batch_mask)

                reg = regular_trick(batch_raw, batch_aug_edge_weight)  # 正则化

                # print('mol_rep_raw.size=', mol_rep_raw.size())
                # print('mol_rep_aug.size=', mol_rep_aug.size())
                if self.config['pretrain_task'] == 'cl':
                    view_loss = self.model.loss_cl(mol_rep_raw, mol_rep_aug) - (self.config['reg_lambda'] * reg)
                elif self.config['pretrain_task'] == 'fg':
                    view_loss = self.model.loss_cl(mol_rep_raw, mol_rep_aug) - (self.config['reg_lambda'] * reg) + self.model.loss_fg(fg_out, batch_raw['fg'])
                else:
                    raise ValueError('not supported pretrain task!')

                view_loss_all += view_loss.item()
                reg_all += reg.item()
                # gradient ascent formulation
                (-view_loss).backward()
                self.view_optim.step()

            # train model to minimize contrastive loss
            self.model.train()
            self.view.eval()
            self.model_optim.zero_grad()

            if self.config['pretrain_mode'] == 'cl':
                batch_raw['edge_weight'], batch_mask['edge_weight'] = None, None

            mol_rep_raw, atom_rep, fg_out = self.model(batch_raw)

            if self.config['pretrain_mode'] == 'ad':
                edge_logit = self.view(batch_raw, atom_rep)  # 计算边的概率
                batch_aug_edge_weight = reparame_trick(edge_logit)  # 重参数化
                batch_mask['edge_weight'] = batch_aug_edge_weight.detach()

            mol_rep_aug, _, _ = self.model(batch_mask)

            # print('batch_raw.fg.size=', batch_raw['fg'].size())
            # fg_out, fg_loss = self.fg_pred(batch_raw)

            if self.config['pretrain_task'] == 'cl':
                model_loss = self.model.loss_cl(mol_rep_raw, mol_rep_aug)
            elif self.config['pretrain_task'] == 'fg':
                model_loss = self.model.loss_cl(mol_rep_raw, mol_rep_aug) + self.model.loss_fg(fg_out, batch_raw['fg'])
            else:
                raise ValueError('not supported pretrain task!')

            model_loss_all += model_loss.item()
            # standard gradient descent formulation
            model_loss.backward()
            self.model_optim.step()

            self.writer.add_scalar('model_loss', model_loss, global_step=self.optim_steps)

            if self.config['pretrain_task'] == 'ad':
                self.writer.add_scalar('view_loss', view_loss, global_step=self.optim_steps)

            self.optim_steps += 1

            # if (batch_index + 1) % 100 == 0 or batch_index == total_step - 1:
            #     acc = test_acc(fg_out.softmax(dim=-1), batch_raw['fg'])
            #     print()
            #     print(f'Accuracy on batch {batch_index}: {acc}')

            batch_index += 1

        model_loss_mean = model_loss_all / len(self.loader)
        view_loss_mean = view_loss_all / len(self.loader)
        reg_mean = reg_all / len(self.loader)
        return model_loss_mean, view_loss_mean, reg_mean

    def save_ckpt(self, epoch):
        checkpoint = {
            "model": self.model.state_dict(),
            'model_optim': self.model_optim.state_dict(),
            'view_optim': self.view_optim.state_dict(),
            "epoch": epoch,
            'best_loss': self.best_loss,
            'optim_steps': self.optim_steps
        }
        path = os.path.join(self.writer.log_dir, 'checkpoint')
        os.makedirs(path, exist_ok=True)
        torch.save(checkpoint, os.path.join(path, 'model_{}.pth'.format(epoch)))

    def load_ckpt(self, load_pth):
        checkpoint = torch.load(load_pth, map_location='cuda')
        # print(checkpoint)
        self.writer = SummaryWriter(os.path.dirname(load_pth))
        self.model.load_state_dict(checkpoint['model'])
        self.model_optim.load_state_dict(checkpoint['model_optim'])
        self.view_optim.load_state_dict(checkpoint['view_optim'])
        self.start_epoch = checkpoint['epoch'] + 1
        self.best_loss = checkpoint['best_loss']
        self.optim_steps = checkpoint['optim_steps']

    def train(self):
        write_record(self.txtfile, self.config)
        for i in range(self.start_epoch, self.config['epochs'] + 1):
            if self.config['lr_scheduler']['type'] in ['cos', 'square', 'linear']:
                self.lr_scheduler.adjust_lr(self.model_optim, i)
                self.lr_scheduler.adjust_lr(self.view_optim, i)

            model_loss, view_loss, reg = self._train_step()

            if self.config['save_model'] == 'best_valid':
                if model_loss < self.best_loss:
                    self.best_loss = model_loss

                    if self.config['DP']:
                        model_dict = {
                            'model': self.model.module.state_dict()
                        }
                        torch.save(model_dict, os.path.join(self.writer.log_dir, 'model.pth'))
                    else:
                        model_dict = {
                            'model': self.model.state_dict()
                        }
                        torch.save(model_dict, os.path.join(self.writer.log_dir, 'model.pth'))
            if i % self.config['save_ckpt'] == 0:
                self.save_ckpt(i)

            print(f'Epoch:{i} model_loss:{model_loss} view_loss:{view_loss}')
            write_record(self.txtfile, f'Epoch:{i} model_loss:{model_loss} view_loss:{view_loss}')


if __name__ == '__main__':
    path = "config/config_pretrain.yaml"
    config = yaml.load(open(path, "r"), Loader=yaml.FullLoader)
    print(config['task_name'])
    set_seed(config['seed'])
    pre_trainer = PreTrainer(config, path)
    pre_trainer.train()
