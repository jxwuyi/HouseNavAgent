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
def initialize_weights(cls, small_init=False):
    for m in cls.modules():
        if isinstance(m, nn.Conv2d):
            n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            m.weight.data.normal_(0, math.sqrt(2. / n))
            m.bias.data.zero_()
        elif isinstance(m, nn.BatchNorm2d):
            m.weight.data.fill_(1)
            m.bias.data.zero_()
        elif isinstance(m, nn.Linear):
            if not small_init:
                m.weight.data.normal_(0, math.sqrt(1. / m.in_features))
            else:
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

########### Scheduler #############
class Schedule(object):
    def value(self, t):
        """Value of the schedule at time t"""
        raise NotImplementedError()

class ConstantSchedule(object):
    def __init__(self, value):
        """Value remains constant over time.
        Parameters
        ----------
        value: float
            Constant value of the schedule
        """
        self._v = value

    def value(self, t):
        """See Schedule.value"""
        return self._v

def linear_interpolation(l, r, alpha):
    return l + alpha * (r - l)

class PiecewiseSchedule(object):
    def __init__(self, endpoints, interpolation=linear_interpolation, outside_value=None):
        """Piecewise schedule.
        endpoints: [(int, int)]
            list of pairs `(time, value)` meanining that schedule should output
            `value` when `t==time`. All the values for time must be sorted in
            an increasing order. When t is between two times, e.g. `(time_a, value_a)`
            and `(time_b, value_b)`, such that `time_a <= t < time_b` then value outputs
            `interpolation(value_a, value_b, alpha)` where alpha is a fraction of
            time passed between `time_a` and `time_b` for time `t`.
        interpolation: lambda float, float, float: float
            a function that takes value to the left and to the right of t according
            to the `endpoints`. Alpha is the fraction of distance from left endpoint to
            right endpoint that t has covered. See linear_interpolation for example.
        outside_value: float
            if the value is requested outside of all the intervals sepecified in
            `endpoints` this value is returned. If None then AssertionError is
            raised when outside value is requested.
        """
        idxes = [e[0] for e in endpoints]
        assert idxes == sorted(idxes)
        self._interpolation = interpolation
        self._outside_value = outside_value
        self._endpoints      = endpoints

    def value(self, t):
        """See Schedule.value"""
        for (l_t, l), (r_t, r) in zip(self._endpoints[:-1], self._endpoints[1:]):
            if l_t <= t and t < r_t:
                alpha = float(t - l_t) / (r_t - l_t)
                return self._interpolation(l, r, alpha)

        # t does not belong to any of the pieces, so doom.
        assert self._outside_value is not None
        return self._outside_value

class LinearSchedule(object):
    def __init__(self, schedule_timesteps, final_p, initial_p=1.0):
        """Linear interpolation between initial_p and final_p over
        schedule_timesteps. After this many timesteps pass final_p is
        returned.
        Parameters
        ----------
        schedule_timesteps: int
            Number of timesteps for which to linearly anneal initial_p
            to final_p
        initial_p: float
            initial output value
        final_p: float
            final output value
        """
        self.schedule_timesteps = schedule_timesteps
        self.final_p            = final_p
        self.initial_p          = initial_p

    def value(self, t):
        """See Schedule.value"""
        fraction  = min(float(t) / self.schedule_timesteps, 1.0)
        return self.initial_p + fraction * (self.final_p - self.initial_p)

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
