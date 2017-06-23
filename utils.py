import numpy as np
import random
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

####### Util Functions ############
def discount_with_dones(rewards, dones, gamma):
    discounted = []
    r = 0
    for reward, done in zip(rewards[::-1], dones[::-1]):
        r = reward + gamma*r
        r = r*(1.-done)
        discounted.append(r)
    return discounted[::-1]


def split_batched_array(action, shape):
    actions = []
    p = 0
    for d in shape:
        actions.append(action[:, p:(p + d)])
        p += d
    return actions


def sample_n_unique(sampling_f, n):
    """Helper function. Given a function `sampling_f` that returns
    comparable objects, sample n such unique objects.
    """
    res = []
    while len(res) < n:
        candidate = sampling_f()
        if candidate not in res:
            res.append(candidate)
    return res


def clip_grad_norm(parameters, max_norm):
    parameters = list(filter(lambda p: p.grad is not None, parameters))
    max_norm = float(max_norm)
    for p in parameters:
        param_norm = p.grad.data.norm()
        clip_coef = max_norm / (param_norm + 1e-6)
        if clip_coef < 1:
            p.grad.data.mul_(clip_coef)


############ Weight Initialization ############
def initialize_weights(cls):
    for m in cls.modules():
        if isinstance(m, nn.Conv2d):
            n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            m.weight.data.normal_(0, math.sqrt(2. / n))
            m.bias.data.zero_()
        elif isinstance(m, nn.BatchNorm2d):
            m.weight.data.fill_(1)
            m.bias.data.zero_()
        elif isinstance(m, nn.Linear):
            #m.weight.data.normal_(0, math.sqrt(1. / m.in_features))
            #m.weight.data.linear_(-3e-4, 3e-4)
            m.weight.data.normal_(0, 0.01)
            m.bias.data.zero_()
        elif isinstance(m, nn.GRU) or isinstance(m, nn.LSTM):
            for key, param in m.named_parameters():
                if 'weight' in key:
                    #param.data.normal_(0, 0.01)
                    std = math.sqrt(2.0 / (param.size(0) + param.size(1)))
                    param.data.normal_(0, std)  # Xavier
                else:
                    param.data.zero_()


######## Logging Utils ############
class MyLogger:
    def __init__(self, logdir, clear_file = False, filename = None):
        import os
        if not os.path.exists(logdir):
            os.makedirs(logdir)
        self.fname = logdir + '/' + (filename or 'progress.txt')
        if clear_file:
            try:
                os.remove(self.fname)
            except OSError:
                pass

    def print(self, str, to_screen = True):
        if to_screen:
            print(str)
        with open(self.fname, 'a') as f:
            print(str, file=f)


class FakeLogger:
    def __init__(self, *args, **dict_args):
        pass

    def print(self, *args, **dict_args):
        pass


def log_var_stats(logger, v):
    logger.print('  -> Param<{}>, '.format(v.size())+\
                 'Val Stats = [norm = %.7f, mean = %.7f, var = %.7f, min = %.7f, max = %.7f]'%(
                    v.data.norm(), v.data.mean(), v.data.var(), v.data.min(), v.data.max()), False)
    if v.grad is None:
        logger.print('              >> Grad Stats = None', False)
    else:
        g = v.grad
        logger.print('              >> Grad Stats = [norm = %.7f, mean = %.7f, var = %.7f, min = %.7f, max = %.7f]' % (
            g.data.norm(), g.data.mean(), g.data.var(), g.data.min(), g.data.max()
        ), False)

def log_parameter_stats(logger, p):
    if (logger is None) or isinstance(logger, FakeLogger):
        return
    assert isinstance(p, nn.Module), '[Error in <utils.log_parameter_stats>] policy must be an instance of <nn.Module>'
    assert hasattr(logger, 'print'), '[Error in <utils.log_parameter_stats>] logger must have method <print>'
    if hasattr(p,'conv_layers'):
        for i,conv in enumerate(p.conv_layers):
            if conv is None: continue
            logger.print('>> Conv Layer#{} <{}>'.format(i,conv),False)
            for v in conv.parameters():
                log_var_stats(logger, v)
    if hasattr(p,'bc_layers'):
        for i,bc in enumerate(p.bc_layers):
            if bc is None: continue
            logger.print('>> Batch Norm Layer#{} <{}>'.format(i,bc),False)
            for v in bc.parameters():
                log_var_stats(logger, v)
    if hasattr(p,'linear_layers'):
        for i,lr in enumerate(p.linear_layers):
            if lr is None: continue
            logger.print('>> Linear Layer#{} <{}>'.format(i,lr),False)
            for v in lr.parameters():
                log_var_stats(logger, v)
