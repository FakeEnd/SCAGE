import warnings
warnings.filterwarnings("ignore")

import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '0'
# os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
import yaml
import shutil
import numpy as np
from datetime import datetime
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from utils.public_util import set_seed, EarlyStopping
from utils.data_util import get_downstream_task_names
from utils.metric_util import compute_cls_metric, compute_reg_metric, compute_cliff_metric
from utils.loss_util import bce_loss
from utils.scheduler_util import *

from data_process.loader import MoleculeDataset
from data_process.data_transform import TransformFn
from data_process.data_collator import collator, collator_finetune
from data_process.split import create_splitter
from models.finetune_model import DownstreamModel

import nni
params_nni = {
    'seed': 1,
    'batch_size': 128,
    'task': 'lipophilicity',

    'init_lr': 0.0002,
    'init_base_lr': 0.0001,
    'weight_decay': 2e-5,

    'scheduler_type': None,
    'warm_up_epoch': 5,
    'start_lr': 5e-4,

    'num_layers': 3,
    'hidden_dim': 256,
    'dropout': 0.08
}
tuner_params = nni.get_next_parameter() # 获得下一组搜索空间中的参数
params_nni.update(tuner_params) # 更新参数


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

        self.train_loader, self.val_loader, self.test_loader = self.get_data_loaders()
        self.net = self._get_net()
        self.criterion = self._get_loss_fn()
        self.optim = self._get_optim()
        self.lr_scheduler = self._get_lr_scheduler()

        if config['checkpoint']:
            self.load_ckpt(self.config['checkpoint'])
        else:
            self.start_epoch = 1
            self.optim_steps = 0
            self.best_metric = -np.inf if config['task'] == 'classification' else np.inf
            self.writer = SummaryWriter('{}/{}_{}_{}_{}_{}_{}'.format(
                'finetune_result', config['task_name'], config['seed'], config['split_type'],
                config['optim']['init_lr'],
                config['batch_size'], datetime.now().strftime('%b%d_%H:%M')
            ))
        self.txtfile = os.path.join(self.writer.log_dir, 'record.txt')
        copyfile(file_path, self.writer.log_dir)

    def get_data_loaders(self):

        dataset = MoleculeDataset(root=self.config['root'] + self.config['task_name'],
                                  is_pretrain=self.config['is_pretrain'],
                                  target=self.config['target'],
                                  transform=TransformFn(self.config))
        splitter = create_splitter(self.config['split_type'])
        train_dataset, val_dataset, test_dataset = splitter.split(dataset, frac_train=0.8, frac_valid=0.1,
                                                                  frac_test=0.1)

        self.imbalance_ratio = ((dataset.data.label == -1).sum()) / ((dataset.data.label == 1).sum())

        train_loader = DataLoader(train_dataset,
                                  batch_size=self.config['batch_size'],
                                  shuffle=True,
                                  num_workers=4,
                                  collate_fn=lambda x: collator_finetune(x, self.config['model']),
                                  drop_last=True)
        val_loader = DataLoader(val_dataset,
                                batch_size=self.config['batch_size'],
                                shuffle=False,
                                num_workers=4,
                                collate_fn=lambda x: collator_finetune(x, self.config['model']),
                                drop_last=True)
        test_loader = DataLoader(test_dataset,
                                 batch_size=self.config['batch_size'],
                                 shuffle=False,
                                 num_workers=4,
                                 collate_fn=lambda x: collator_finetune(x, self.config['model']),
                                 drop_last=True)

        return train_loader, val_loader, test_loader

    def _get_net(self):
        model = DownstreamModel(self.config)
        if self.config['pretrain_model_path'] != 'None':
            state_dict = torch.load(self.config['pretrain_model_path'],
                                    map_location='cuda')
            # model.model.load_model_state_dict(state_dict['model'])
            model.model.load_model_state_dict(state_dict['model'])
            print("Loading pretrain model from", self.config['pretrain_model_path'])
        return model

    def _get_loss_fn(self):
        loss_type = self.config['loss_type']
        if loss_type == 'bce':
            return bce_loss()
        elif loss_type == 'wb_bce':
            ratio = self.imbalance_ratio
            return bce_loss(weights=[1.0, ratio])
        elif loss_type == 'mse':
            return nn.MSELoss()
        elif loss_type == 'l1':
            return nn.L1Loss()
        else:
            raise ValueError('not supported loss function!')

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

    def _step(self, model, batch):

        pred = model(batch)

        if self.config['task'] == 'classification':
            label = batch['label']
            is_valid = label ** 2 > 0
            label = ((label + 1.0) / 2).view(pred.shape)
            loss = self.criterion(pred, label)
            loss = torch.where(is_valid, loss, torch.zeros(loss.shape).to(loss.device).to(loss.dtype))
            loss = torch.sum(loss) / torch.sum(is_valid)
        else:
            loss = self.criterion(pred, batch['label'].float())

        return loss, pred

    def _train_step(self):
        self.net.train()
        num_data = 0
        train_loss = 0
        y_pred = []
        y_true = []
        for _, batch in enumerate(self.train_loader):
            self.optim.zero_grad()
            batch = {key: value.to('cuda') for key, value in batch.items()
                     if value is not None and not isinstance(value, list)}
            batch['edge_weight'] = None

            loss, pred = self._step(self.net, batch)
            train_loss += loss.item()
            self.writer.add_scalar('train_loss', loss, global_step=self.optim_steps)

            y_pred.extend(pred.cpu().detach().numpy())
            y_true.extend(batch['label'].cpu().numpy())

            loss.backward()

            self.optim.step()
            num_data += 1
            self.optim_steps += 1

        train_loss /= num_data
        if self.config['task'] == 'regression':
            if self.config['task_name'] in ['qm7', 'qm8', 'qm9']:
                mae, _ = compute_reg_metric(y_true, y_pred)
                return train_loss, mae
            else:
                _, rmse = compute_reg_metric(y_true, y_pred)
                return train_loss, rmse
        elif self.config['task'] == 'classification':
            roc_auc = compute_cls_metric(y_true, y_pred)
            return train_loss, roc_auc

    def _valid_step(self, valid_loader):
        self.net.eval()
        y_pred = []
        y_true = []
        valid_loss = 0
        num_data = 0
        for batch in valid_loader:
            batch = {key: value.to('cuda') for key, value in batch.items()
                     if value is not None and not isinstance(value, list)}
            batch['edge_weight'] = None
            with torch.no_grad():
                loss, pred = self._step(self.net, batch)
            valid_loss += loss.item()
            num_data += 1
            y_pred.extend(pred.cpu().detach().numpy())
            y_true.extend(batch['label'].cpu().numpy())

        valid_loss /= num_data

        if self.config['task'] == 'regression':
            if self.config['task_name'] in ['qm7', 'qm8', 'qm9']:
                mae, _ = compute_reg_metric(y_true, y_pred)
                return valid_loss, mae
            else:
                _, rmse = compute_reg_metric(y_true, y_pred)
                return valid_loss, rmse
        elif self.config['task'] == 'classification':
            roc_auc = compute_cls_metric(y_true, y_pred)
            return valid_loss, roc_auc

    def save_ckpt(self, epoch):
        checkpoint = {
            "net": self.net.state_dict(),
            'optimizer': self.optim.state_dict(),
            "epoch": epoch,
            'best_metric': self.best_metric,
            'optim_steps': self.optim_steps
        }
        path = os.path.join(self.writer.log_dir, 'checkpoint')
        os.makedirs(path, exist_ok=True)
        torch.save(checkpoint, os.path.join(self.writer.log_dir, 'checkpoint', 'model_{}.pth'.format(epoch)))

    def load_ckpt(self, load_pth):
        checkpoint = torch.load(load_pth, map_location='cuda')
        self.writer = SummaryWriter(os.path.dirname(load_pth))
        self.net.load_state_dict(checkpoint['net'])
        self.optim.load_state_dict(checkpoint['optimizer'])
        self.start_epoch = checkpoint['epoch'] + 1
        self.best_metric = checkpoint['best_metric']
        self.optim_steps = checkpoint['optim_steps']

    def train(self):
        print(self.config)
        write_record(self.txtfile, self.config)
        self.net = self.net.to('cuda')
        # 设置模型并行
        if self.config['DP']:
            self.net = torch.nn.DataParallel(self.net)
        # 设置早停
        mode = 'lower' if self.config['task'] == 'regression' else 'higher'
        stopper = EarlyStopping(mode=mode, patience=self.config['patience'],
                                filename=os.path.join(self.writer.log_dir, 'model.pth'))

        val_metric_list, test_metric_list = [], []
        for i in range(self.start_epoch, self.config['epochs'] + 1):
            if self.config['lr_scheduler']['type'] in ['cos', 'square', 'linear']:
                self.lr_scheduler.adjust_lr(self.optim, i)
            print("Epoch {} cur_lr {}".format(i, self.optim.param_groups[1]['lr']))
            train_loss, train_metric = self._train_step()
            valid_loss, valid_metric = self._valid_step(self.val_loader)
            test_loss, test_metric = self._valid_step(self.test_loader)
            val_metric_list.append(valid_metric)
            test_metric_list.append(test_metric)
            if stopper.step(valid_metric, self.net, test_score=test_metric):
                stopper.report_final_results(i_epoch=i)
                break
            if self.config['save_model'] == 'best_valid':
                if (self.config['task'] == 'regression' and (self.best_metric > train_metric)) or (
                        self.config['task'] == 'classification' and (self.best_metric < train_metric)):
                    self.best_metric = train_metric
                    if self.config['DP']:
                        torch.save(self.net.module.state_dict(), os.path.join(self.writer.log_dir, 'model.pth'))
                    else:
                        model_dict = {'model': self.net.state_dict()}
                        torch.save(model_dict, os.path.join(self.writer.log_dir, 'model.pth'))
            elif self.config['save_model'] == 'each':
                if self.config['DP']:
                    torch.save(self.net.module.state_dict(),
                               os.path.join(self.writer.log_dir, 'model_{}.pth'.format(i)))
                else:
                    torch.save(self.net.state_dict(), os.path.join(self.writer.log_dir, 'model_{}.pth'.format(i)))
            self.writer.add_scalar('valid_loss', valid_loss, global_step=i)
            self.writer.add_scalar('test_loss', test_loss, global_step=i)

            if config['task'] == 'classification':
                print(f'train_loss:{train_loss} valid_loss:{valid_loss} test_loss:{test_loss}\n'
                      f'train_auc:{train_metric} valid_auc:{valid_metric} test_auc:{test_metric}')
            else:
                if self.config["task_name"] in ['qm7', 'qm8', 'qm9']:
                    print(f'train_loss:{train_loss} valid_loss:{valid_loss} test_loss:{test_loss}\n'
                          f'train_mae:{train_metric} valid_mae:{valid_metric} test_mae:{test_metric}')
                else:
                    print(f'train_loss:{train_loss} valid_loss:{valid_loss} test_loss:{test_loss}\n'
                          f'train_rmse:{train_metric} valid_rmse:{valid_metric} test_rmse:{test_metric}')
            write_record(self.txtfile,
                         f'epoch:{i}\n'
                         f'train_loss:{train_loss} valid_loss:{valid_loss} test_loss:{test_loss}\n'
                         f'train_metric:{train_metric} valid_metric:{valid_metric} test_metric:{test_metric}')
            if i % self.config['save_ckpt'] == 0:
                self.save_ckpt(i)
            nni.report_intermediate_result(test_metric)

        if config['task'] == 'classification':
            best_val_metric = np.max(val_metric_list)
            best_test_metric = np.max(test_metric_list)
            true_test_metric = test_metric_list[np.argmax(val_metric_list)]

        elif config['task'] == 'regression':
            best_val_metric = np.min(val_metric_list)
            best_test_metric = np.min(test_metric_list)
            true_test_metric = test_metric_list[np.argmin(val_metric_list)]
        else:
            raise ValueError('only supported classification or regression!')

        print(f'best_val_metric:{best_val_metric}\t'
              f'best_test_metric:{best_test_metric}\t'
              f'true_test_metric:{true_test_metric}')

        write_record(self.txtfile, f'best_val_metric:{best_val_metric}\t'
                                   f'best_test_metric:{best_test_metric}\t'
                                   f'true_test_metric:{true_test_metric}')
        nni.report_final_result(true_test_metric)


if __name__ == '__main__':
    path = f"config/config_finetune.yaml"
    config = yaml.load(open(path, "r"), Loader=yaml.FullLoader)

    config['task_name'] = params_nni['task']
    config['batch_size'] = params_nni['batch_size']
    config['seed'] = params_nni['seed']

    config['optim']['init_lr'] = params_nni['init_lr']
    config['optim']['init_base_lr'] = params_nni['init_base_lr']
    config['optim']['weight_decay'] = params_nni['weight_decay']

    config['lr_scheduler']['type'] = params_nni['scheduler_type']
    config['lr_scheduler']['warm_up_epoch'] = params_nni['warm_up_epoch']
    config['lr_scheduler']['start_lr'] = params_nni['start_lr']

    config['DownstreamModel']['num_layers'] = params_nni['num_layers']
    config['DownstreamModel']['hidden_dim'] = params_nni['hidden_dim']
    config['DownstreamModel']['dropout'] = params_nni['dropout']

    print(config['task_name'])
    set_seed(params_nni['seed'])
    get_downstream_task_names(config)

    trainer = Trainer(config, path)
    trainer.train()
