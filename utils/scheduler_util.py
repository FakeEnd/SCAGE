import math


class SquareSche():
    """A simple wrapper class for learning rate scheduling"""

    def __init__(self, total_epoch, init_lr=0, warm_up_end_lr=2e-5, warm_up_epoch=1):
        self.total_epoches = total_epoch
        self.init_lr = init_lr
        self.base_lr = warm_up_end_lr
        self.warm_up_epoches = warm_up_epoch
        self.warm_up_step = (warm_up_end_lr - init_lr) / warm_up_epoch
        self.decay_factor = warm_up_end_lr * warm_up_epoch**0.5

    def adjust_lr(self, optimizer, epoch):
        if epoch < self.warm_up_epoches:
            lr = epoch * self.warm_up_step + self.init_lr
        else:
            lr = self.decay_factor * epoch**-0.5
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr


class LinearSche():
    """A simple wrapper class for learning rate scheduling"""

    def __init__(self, total_epoch, init_lr=0, warm_up_end_lr=2e-5, warm_up_epoch=1):
        self.total_epoches = total_epoch
        self.init_lr = init_lr
        self.base_lr = warm_up_end_lr
        self.warm_up_epoches = warm_up_epoch
        self.warm_up_step = (warm_up_end_lr - init_lr) / warm_up_epoch
        self.decay_factor = (warm_up_end_lr - init_lr) / (total_epoch - warm_up_epoch + 1)

    def adjust_lr(self, optimizer, epoch):
        if epoch < self.warm_up_epoches:
            lr = epoch * self.warm_up_step + self.init_lr
        else:
            lr = self.base_lr - (epoch - self.warm_up_epoches)* self.decay_factor
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr


class CosSche:
    def __init__(self, total_epoch, init_lr=0, warm_up_end_lr=2e-5, warm_up_epoch=1):
        self.total_epoch = total_epoch
        self.init_lr = init_lr
        self.base_lr = warm_up_end_lr
        self.warm_up_epoch = warm_up_epoch
        self.warm_up_step = (warm_up_end_lr - init_lr) / warm_up_epoch

    def adjust_lr(self, optimizer, epoch):
        if epoch < self.warm_up_epoch:
            lr = epoch * self.warm_up_step + self.init_lr
        else:
            lr = self.base_lr * 0.5 * (1.+math.cos(math.pi*(epoch-self.warm_up_epoch)/(self.total_epoch-self.warm_up_epoch)))
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
