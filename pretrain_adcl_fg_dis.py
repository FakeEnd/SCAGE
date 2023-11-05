import argparse
import os
import time

from torchmetrics import Accuracy

os.environ['CUDA_VISIBLE_DEVICES'] = '3,2,0,1'
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
# os.environ['RANK'] = '0'                     # 当前进程的rank设置为0
# os.environ['WORLD_SIZE'] = '2'               # 总共有4个进程参与训练
# os.environ['MASTER_ADDR'] = '127.0.0.1'      # master节点的地址，设置为本地IP地址或其它
# os.environ['MASTER_PORT'] = '1234'           # 端口号设置，自己定

import yaml
import shutil
import numpy as np
from datetime import datetime
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch import distributed as dist
import torchmetrics

import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')
# torch.autograd.set_detect_anomaly(True)

from utils.public_util import set_seed
from utils.loss_util import NTXentLoss
from utils.scheduler_util import *

from data_process.loader import PretrainDataset
from data_process.data_transform import MaskTransformFn
from data_process.data_collator import pretrain_collator
from models.pretrain_model import MolGraphCL, ViewLearner, reparame_trick, regular_trick
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

        self.loader, self.data_sampler = self.get_data_loaders()
        self.model, self.view = self._get_net()
        self.criterion = self._get_loss_fn()
        self.model_optim, self.view_optim = self._get_optim()
        self.lr_scheduler = self._get_lr_scheduler()
        self.training_mode = "original"
        # self.training_mode = "fg_enhance"

        if config['checkpoint']:
            print('loading check point')
            self.load_ckpt(self.config['checkpoint'])
        else:
            self.start_epoch = 1
            self.optim_steps = 0
            self.best_loss = np.inf
            self.writer = SummaryWriter('{}/{}/{}_{}_{}'.format(
                'pretrain_model', self.training_mode, config['task_name'], config['seed'],
                datetime.now().strftime('%b%d_%H_%M')))
        self.txtfile = os.path.join(self.writer.log_dir, 'record.txt')
        copyfile(file_path, self.writer.log_dir)

    def get_data_loaders(self):
        # dataset = PretrainDataset(self.config['root'] + self.config['task_name'],
        #                         transform=MaskTransformFn(self.config))
        # print('all_dataset_num:', len(dataset))

        global data_sampler
        # dataset = PretrainDataset(self.config['root'] + self.config['task_name'],
        #                         transform=MaskTransformFn(self.config))
        dataset = PretrainDataset(self.config['root'] + self.config['task_name'])
        print('all_dataset_num:', len(dataset))
        if self.config['DDP']:
            data_sampler = torch.utils.data.distributed.DistributedSampler(dataset)
            loader = DataLoader(dataset,
                                batch_size=self.config['batch_size'],
                                shuffle=(data_sampler is None),
                                num_workers=40,
                                collate_fn=lambda x: pretrain_collator(x, self.config['model']),
                                drop_last=True,
                                sampler=data_sampler,
                                pin_memory=True)
        else:
            loader = DataLoader(dataset,
                                batch_size=self.config['batch_size'],
                                shuffle=True,
                                num_workers=40,
                                collate_fn=lambda x: pretrain_collator(x, self.config['model']),
                                drop_last=True)

        return loader, data_sampler

    def _get_net(self):
        model = MolGraphCL(self.config['model'])
        view_learner = ViewLearner(self.config['model'])

        model = model.to(device)
        view_learner = view_learner.to(device)
        if self.config['DP']:
            model = torch.nn.DataParallel(model)
            view_learner = torch.nn.DataParallel(view_learner)
        if self.config['DDP']:
            model = torch.nn.parallel.DistributedDataParallel(model,
                                                              device_ids=[args.local_rank],
                                                              output_device=args.local_rank,
                                                              broadcast_buffers=False,
                                                              find_unused_parameters=True)

            view_learner = torch.nn.parallel.DistributedDataParallel(view_learner,
                                                              device_ids=[args.local_rank],
                                                              output_device=args.local_rank,
                                                              broadcast_buffers=False,
                                                              find_unused_parameters=True)

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
        self.model.train()
        view_loss_all = 0
        model_loss_all = 0
        reg_all = 0
        batch_index = 0
        total_step = len(self.loader)
        test_acc = Accuracy(num_classes=83, average="micro").to(device)
        # if args.local_rank == 0:
        #     print('开始构造Loader')
        #     st = time.time()
        #     pre_time = st
        #     index = 0
        for batch_raw, batch_mask in tqdm(self.loader, leave=False, ascii=True):
            # if args.local_rank == 0:
            #     now_time = time.time()
            #     cost = now_time - pre_time
            #     print(f'构造第{index}个batch花费的时间: {cost:.5f}')
            #     st1 = time.time()
            # print('args.local_rank=', args.local_rank)
            self.view.train()
            self.view_optim.zero_grad()
            self.model.eval()

            batch_raw = {key: value.cuda(args.local_rank, non_blocking=True) for key, value in batch_raw.items()
                         if value is not None and not isinstance(value, list)}
            batch_mask = {key: value.cuda(args.local_rank, non_blocking=True) for key, value in batch_mask.items()
                          if value is not None and not isinstance(value, list)}
            # print(batch_raw['fg'])
            batch_raw['edge_weight'] = None
            mol_rep_raw, atom_rep, fg_out = self.model(batch_raw)
            edge_logit = self.view(batch_raw, atom_rep)  # 计算边的概率
            batch_aug_edge_weight = reparame_trick(edge_logit)  # 重参数化

            batch_mask['edge_weight'] = batch_aug_edge_weight
            mol_rep_aug, _, _ = self.model(batch_mask)

            reg = regular_trick(batch_raw, batch_aug_edge_weight)  # 正则化

            # print('mol_rep_raw.size=', mol_rep_raw.size())
            # print('mol_rep_aug.size=', mol_rep_aug.size())
            if self.config['DP'] or self.config['DDP']:
                if self.training_mode == "original":
                    view_loss = self.model.module.loss_cl(mol_rep_raw, mol_rep_aug)
                elif self.training_mode == "fg_enhance":
                    view_loss = self.model.module.loss_cl(mol_rep_raw, mol_rep_aug) + self.model.loss_fg(fg_out,
                                                                                                          batch_raw[
                                                                                                              'fg'])
                # model_loss = self.model.module.loss_cl(mol_rep_raw, mol_rep_aug)
            else:
                # model_loss = self.model.loss_cl(mol_rep_raw, mol_rep_aug)
                if self.training_mode == "original":
                    view_loss = self.model.module.loss_cl(mol_rep_raw, mol_rep_aug)
                elif self.training_mode == "fg_enhance":
                    view_loss = self.model.module.loss_cl(mol_rep_raw, mol_rep_aug) + self.model.loss_fg(fg_out,
                                                                                                          batch_raw[
                                                                                                              'fg'])

            dist.barrier()
            view_loss_all += view_loss.item()
            reg_all += reg.item()
            # gradient ascent formulation
            # print('view_loss=', view_loss)
            (-view_loss).backward()
            self.view_optim.step()
            view_loss = reduce_mean(view_loss)

            self.model.train()
            self.view.eval()
            # train model to minimize contrastive loss
            self.model_optim.zero_grad()

            mol_rep_raw, atom_rep, fg_out = self.model(batch_raw)
            edge_logit = self.view(batch_raw, atom_rep)  # 计算边的概率
            batch_aug_edge_weight = reparame_trick(edge_logit)  # 重参数化
            # batch_mask['batch_aug_edge_weight'] = batch_aug_edge_weight
            batch_mask['edge_weight'] = batch_aug_edge_weight
            mol_rep_aug, _, _ = self.model(batch_mask)


            # print('batch_raw.fg.size=', batch_raw['fg'].size())

            if self.config['DP'] or self.config['DDP']:
                if self.training_mode == "original":
                    model_loss = self.model.module.loss_cl(mol_rep_raw, mol_rep_aug)
                elif self.training_mode == "fg_enhance":
                    model_loss = self.model.module.loss_cl(mol_rep_raw, mol_rep_aug) + self.model.loss_fg(fg_out, batch_raw['fg'])
                # model_loss = self.model.module.loss_cl(mol_rep_raw, mol_rep_aug)
            else:
                # model_loss = self.model.loss_cl(mol_rep_raw, mol_rep_aug)
                if self.training_mode == "original":
                    model_loss = self.model.module.loss_cl(mol_rep_raw, mol_rep_aug)
                elif self.training_mode == "fg_enhance":
                    model_loss = self.model.module.loss_cl(mol_rep_raw, mol_rep_aug) + self.model.loss_fg(fg_out,batch_raw['fg'])

            dist.barrier()
            model_loss_all += model_loss.item()
            # standard gradient descent formulation
            # print('model_loss=', model_loss)
            model_loss.backward()
            model_loss = reduce_mean(model_loss)
            self.model_optim.step()

            self.writer.add_scalar('model_loss', model_loss, global_step=self.optim_steps)
            self.writer.add_scalar('view_loss', view_loss, global_step=self.optim_steps)

            self.optim_steps += 1

            # if ((batch_index + 1) % 100 == 0 or batch_index == (total_step - 1)) and args.local_rank == 0:
            #     self.model.eval()
            #     acc = test_acc(fg_out.softmax(dim=-1), batch_raw['fg'])
            #     self.model.train()
            #     print()
            #     print(f'Accuracy on batch {batch_index}: {acc}')

            batch_index += 1
            # if args.local_rank == 0:
            #     pre_time = time.time()
            #     cost1 = pre_time - st1
            #     index += 1
            #     print(f'第{index}个batch训练花费的时间: {cost1:.5f}')
        model_loss_mean = model_loss_all / len(self.loader)
        # print('model_loss_mean1=', model_loss_mean)
        view_loss_mean = view_loss_all / len(self.loader)
        reg_mean = reg_all / len(self.loader)
        # model_loss_mean = reduce_mean(torch.tensor(model_loss_mean))
        # print('model_loss_mean2=', model_loss_mean)
        return model_loss_mean, view_loss_mean, reg_mean

    def save_ckpt(self, epoch):
        model_dict = {
            'model': self.model.state_dict()
        }
        checkpoint = {
            "net": model_dict,
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
        self.writer = SummaryWriter(os.path.dirname(load_pth))
        self.model.load_state_dict(checkpoint['net']['model'])
        self.model_optim.load_state_dict(checkpoint['model_optim'])
        self.view_optim.load_state_dict(checkpoint['view_optim'])
        self.start_epoch = checkpoint['epoch'] + 1
        self.best_loss = checkpoint['best_loss']
        self.optim_steps = checkpoint['optim_steps']

    def train(self):
        # print(self.config)
        write_record(self.txtfile, self.config)
        for i in range(self.start_epoch, self.config['epochs'] + 1):
            self.data_sampler.set_epoch(i)
            if self.config['lr_scheduler']['type'] in ['cos', 'square', 'linear']:
                self.lr_scheduler.adjust_lr(self.model_optim, i)
                self.lr_scheduler.adjust_lr(self.view_optim, i)

            model_loss, view_loss, reg = self._train_step()
            # print('model_loss=', model_loss)

            if self.config['save_model'] == 'best_valid':
                if model_loss < self.best_loss:
                    self.best_loss = model_loss
                    if args.local_rank == 0:
                        if self.config['DP'] or self.config['DDP']:
                            model_dict = {
                                'model': self.model.module.state_dict()
                            }
                            torch.save(model_dict, os.path.join(self.writer.log_dir, 'model.pth'))
                        else:
                            model_dict = {
                                'model': self.model.state_dict()
                            }
                            torch.save(model_dict, os.path.join(self.writer.log_dir, 'model.pth'))
            if i % self.config['save_ckpt'] == 0 and args.local_rank == 0:
                self.save_ckpt(i)
            if args.local_rank == 0:
                print(f'Epoch:{i} model_loss:{model_loss} view_loss:{view_loss}')
            write_record(self.txtfile, f'Epoch:{i} model_loss:{model_loss} view_loss:{view_loss}')

        cleanup()


def reduce_mean(value, average=True):                     # 用于平均所有gpu上的运行结果，比如loss
    dist.all_reduce(value, op=dist.ReduceOp.SUM)  # 求每一个GPU上value值的和
    if average:
        value /= dist.get_world_size()                    # 除以进程数
    return value


def cleanup():
    dist.destroy_process_group()


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--local_rank", default=os.getenv('LOCAL_RANK', -1), type=int)  # 输入当前进行分配GPU编号
    args = parser.parse_args()
    if args.local_rank != -1:
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        dist.init_process_group(backend="nccl", init_method='env://')
    num_gpus = torch.cuda.device_count()
    if num_gpus > 1:
        print('use {} gpus!'.format(num_gpus))

    path = "config/config_pretrain_adcl_fg.yaml"
    config = yaml.load(open(path, "r"), Loader=yaml.FullLoader)
    print(config['task_name'])
    set_seed(config['seed'])
    pre_trainer = PreTrainer(config, path)
    pre_trainer.train()
