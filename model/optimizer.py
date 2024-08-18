"""
Copyright (C) 2018 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""

import torch.optim as optim


class Optimizer:
    """
    Wrapper around torch.optim + learning rate
    """
    def __init__(self, params, config):
        self.config = config
        if config.optimizer == 'sgd':
            self.lr_decay = config.lr_decay
            self.lr_stepvalues = sorted(config.lr_stepvalues)
            self.learner = optim.SGD(params, lr=config.learning_rate,
                                     weight_decay=config.weight_decay,
                                      eps=config.eps)
            
        elif config.optimizer == 'adam':
            self.lr_decay = config.lr_decay
            self.lr_stepvalues = sorted(config.lr_stepvalues)
            self.learner = optim.Adam(params, lr=config.learning_rate,
                                      weight_decay=config.weight_decay,
                                      eps=config.eps)
            
        elif config.optimizer == 'rmsprop':
            self.lr_decay = config.lr_decay
            self.learner = optim.RMSprop(params, lr=config.learning_rate,
                                         weight_decay=config.weight_decay)

    def adjust_lr(self, epoch):
        if self.config.optimizer not in ['sgd', 'adam', 'rmsprop']:
            return self.config.learning_rate
        
        for param_group in self.learner.param_groups:
            param_group['lr'] = param_group['lr'] * self.lr_decay


        return self.learner.param_groups[0]['lr']

    def mult_lr(self, f):
        for param_group in self.learner.param_groups:
            param_group['lr'] *= f
