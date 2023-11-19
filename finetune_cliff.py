import warnings
warnings.filterwarnings("ignore")

import os
import sys
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
import yaml
import shutil
import numpy as np
from datetime import datetime

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from utils.public_util import set_seed
from utils.metric_util import calc_rmse, calc_cliff_rmse
from utils.scheduler_util import *

from data_process.loader import CliffDataset
from data_process.data_transform import TransformFn
from data_process.data_collator import collator
from models.finetune_model import CliffModel



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


class Trainer(object):
    def __init__(self, config, file_path):
        self.imbalance_ratio = None
        self.config = config

        self.train_loader, self.test_loader = self.get_data_loaders()
        self.net = self._get_net()
        self.reg_loss, self.cls_loss = self._get_loss_fn()
        self.optim = self._get_optim()
        self.lr_scheduler = self._get_lr_scheduler()

        self.start_epoch = 1
        self.optim_steps = 0
        self.best_metric = np.inf
        self.writer = SummaryWriter('{}/{}_{}_{}_{}_{}'.format(
            'finetune_result_cliff', config['task_name'], config['seed'],
            config['optim']['init_lr'],
            config['batch_size'], datetime.now().strftime('%b%d_%H:%M')
        ))
        self.txtfile = os.path.join(self.writer.log_dir, 'record.txt')
        copyfile(file_path, self.writer.log_dir)

    def get_data_loaders(self):
        train_dataset = CliffDataset(root=self.config['root'] + self.config['task_name'],
                                     split_type='train',
                                     transform=TransformFn(self.config))
        test_dataset = CliffDataset(root=self.config['root'] + self.config['task_name'],
                                   split_type='test',
                                    transform=TransformFn(self.config))

        train_loader = DataLoader(train_dataset,
                                  batch_size=self.config['batch_size'],
                                  shuffle=True,
                                  num_workers=0,
                                  collate_fn=lambda x: collator(x, self.config['model']),
                                  drop_last=False)

        test_loader = DataLoader(test_dataset,
                                 batch_size=self.config['batch_size'],
                                 shuffle=False,
                                 num_workers=0,
                                 collate_fn=lambda x: collator(x, self.config['model']),
                                 drop_last=False)

        return train_loader, test_loader

    def _get_net(self):
        model = CliffModel(self.config)
        if self.config['pretrain_model_path'] != 'None':
            state_dict = torch.load(os.path.join(self.config['pretrain_model_path'], 'model_motif.pth'),
                                    map_location='cuda')
            model.model.load_model_state_dict(state_dict['model'])
            print("Loading pretrain model from", os.path.join(self.config['pretrain_model_path'], 'model_motif.pth'))
        return model

    def _get_loss_fn(self):
        if self.config['add_cliff_pred']:
            return nn.MSELoss(), nn.BCEWithLogitsLoss()
        else:
            return nn.MSELoss(), None

    def _get_optim(self):

        optim_type = self.config['optim']['type']
        lr = self.config['optim']['init_lr']
        weight_decay = self.config['optim']['weight_decay']

        layer_list = []
        for name, param in self.net.named_parameters():
            if 'mlp_proj' in name:
                layer_list.append(name)
        params = list(
            map(lambda x: x[1], list(filter(lambda kv: kv[0] in layer_list, self.net.named_parameters()))))
        base_params = list(
            map(lambda x: x[1], list(filter(lambda kv: kv[0] not in layer_list, self.net.named_parameters()))))
        model_params = [{'params': base_params, 'lr': self.config['optim']['init_base_lr']}, {'params': params}]
        if optim_type == 'adam':
            return torch.optim.Adam(model_params, lr=lr, weight_decay=weight_decay)
        elif optim_type == 'rms':
            return torch.optim.RMSprop(model_params, lr=lr, weight_decay=weight_decay)
        elif optim_type == 'sgd':
            return torch.optim.SGD(model_params, lr=lr, weight_decay=weight_decay)
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
        self.net.train()
        num_data = 0
        train_loss = 0
        y_pred = []
        y_true = []
        cliff_list = []
        for _, batch in enumerate(self.train_loader):
            self.optim.zero_grad()
            batch = {key: value.to('cuda') for key, value in batch.items()
                     if value is not None and not isinstance(value, list)}
            batch['edge_weight'] = None
            pred_y, pred_cliff = self.net(batch)
            reg_loss = self.reg_loss(pred_y, batch['label_y'].float())
            if self.config['add_cliff_pred']:
                cls_loss = self.cls_loss(pred_cliff, batch['label_cliff'].float())
                loss = reg_loss + cls_loss * 0.1
            else:
                loss = reg_loss

            train_loss += loss.item()
            self.writer.add_scalar('train_loss', loss, global_step=self.optim_steps)

            y_pred.extend(pred_y.cpu().detach().numpy())
            y_true.extend(batch['label_y'].cpu().numpy())
            cliff_list.extend(batch['label_cliff'].cpu().numpy())
            loss.backward()

            self.optim.step()
            num_data += 1
            self.optim_steps += 1

        train_loss /= num_data
        rmse = calc_rmse(y_true, y_pred)
        rmse_cliff = calc_cliff_rmse(y_test_pred=y_pred, y_test=y_true, cliff_mols_test=cliff_list)

        return train_loss, rmse, rmse_cliff

    def _test_step(self):
        self.net.eval()
        y_pred = []
        y_true = []
        cliff_list = []
        test_loss = 0
        num_data = 0
        for batch in self.test_loader:
            batch = {key: value.to('cuda') for key, value in batch.items()
                     if value is not None and not isinstance(value, list)}
            batch['edge_weight'] = None
            with torch.no_grad():
                pred_y, pred_cliff = self.net(batch)
                reg_loss = self.reg_loss(pred_y, batch['label_y'].float())
                if self.config['add_cliff_pred']:
                    cls_loss = self.cls_loss(pred_cliff, batch['label_cliff'].float())
                    loss = reg_loss + cls_loss * 0.1
                else:
                    loss = reg_loss

            test_loss += loss.item()
            num_data += 1
            y_pred.extend(pred_y.cpu().detach().numpy())
            y_true.extend(batch['label_y'].cpu().numpy())
            cliff_list.extend(batch['label_cliff'].cpu().numpy())

        test_loss /= num_data
        rmse = calc_rmse(y_true, y_pred)
        rmse_cliff = calc_cliff_rmse(y_test_pred=y_pred, y_test=y_true, cliff_mols_test=cliff_list)

        return test_loss, rmse, rmse_cliff

    def train(self):
        print(self.config)
        write_record(self.txtfile, self.config)
        self.net = self.net.to('cuda')

        test_rmse_list = []
        test_rmse_cliff_list = []
        for i in range(self.start_epoch, self.config['epochs'] + 1):
            if self.config['lr_scheduler']['type'] in ['cos', 'square', 'linear']:
                self.lr_scheduler.adjust_lr(self.optim, i)
            print("Epoch {} cur_lr {}".format(i, self.optim.param_groups[1]['lr']))
            train_loss, train_rmse, train_rmse_cliff = self._train_step()
            test_loss, test_rmse, test_rmse_cliff = self._test_step()
            test_rmse_list.append(test_rmse)
            test_rmse_cliff_list.append(test_rmse_cliff)

            if self.best_metric > test_rmse:
                self.best_metric = test_rmse

                model_dict = {'model': self.net.state_dict()}
                torch.save(model_dict, os.path.join(self.writer.log_dir, 'model.pth'))

            print(f'train_loss:{train_loss} train_rmse:{train_rmse} train_rmse_cliff:{train_rmse_cliff}\t'
                  f'test_loss:{test_loss} test_rmse:{test_rmse} test_rmse_cliff:{test_rmse_cliff}')

            write_record(self.txtfile,
                         f'train_loss:{train_loss} train_rmse:{train_rmse} train_rmse_cliff:{train_rmse_cliff}\t'
                         f'test_loss:{test_loss} test_rmse:{test_rmse} test_rmse_cliff:{test_rmse_cliff}')

        best_test_rmse = np.min(test_rmse_list)
        best_test_rmse_cliff = np.min(test_rmse_cliff_list)
        print(f'best_test_rmse:{best_test_rmse} best_test_rmse_cliff:{best_test_rmse_cliff}')
        write_record(self.txtfile, f'best_test_rmse:{best_test_rmse} best_test_rmse_cliff:{best_test_rmse_cliff}')


if __name__ == '__main__':

    path = "config/config_finetune_cliff.yaml"
    config = yaml.load(open(path, "r"), Loader=yaml.FullLoader)
    print(config['task_name'])
    set_seed(config['seed'])
    trainer = Trainer(config, path)
    trainer.train()
