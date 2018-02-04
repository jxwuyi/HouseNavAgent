from headers import *
import common
import random
import utils
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable

class CNNGumbelPolicy(torch.nn.Module):
    def __init__(self, D_shape_in, D_out, hiddens, kernel_sizes=5, strides=2,
                linear_hiddens=[],
                activation=F.relu, use_batch_norm = True):
        """
        D_shape_in: tupe of two ints, the shape of input images
        D_out: a int or a list of ints in length of degree of freedoms
        hiddens, kernel_sizes, strides: either an int or a list of ints with the same length
        """
        super(CNNGumbelPolicy, self).__init__()
        if isinstance(hiddens, int): hiddens = [hiddens]
        if isinstance(kernel_sizes, int): kernel_sizes = [kernel_sizes]
        if isinstance(strides, int): strides = [strides]
        self.n_layer = max(len(hiddens),len(kernel_sizes),len(strides))
        self.hiddens = hiddens
        self.kernel_sizes = kernel_sizes
        self.strides = strides
        if len(self.hiddens) == 1: self.hiddens = self.hiddens * self.n_layer
        if len(self.kernel_sizes) == 1: self.kernel_sizes = self.kernel_sizes * self.n_layer
        if len(self.strides) == 1: self.strides = self.strides * self.n_layer

        assert ((len(self.hiddens) == len(self.kernel_sizes)) and (len(self.strides) == len(self.hiddens))), '[CNNGumbelPolicy] hiddens, kernel_sizes, strides must share the same length'

        if isinstance(D_out, int):
            self.D_out = [D_out]
        else:
            self.D_out = D_out
        self.out_dim = sum(D_out)
        self.func=activation

        self.conv_layers = []
        self.bc_layers = []
        prev_hidden = D_shape_in[0]
        for i, dat in enumerate(zip(self.hiddens, self.kernel_sizes, self.strides)):
            h, k, s = dat
            self.conv_layers.append(nn.Conv2d(prev_hidden, h, kernel_size=k, stride=s))
            setattr(self, 'conv_layer%d'%i, self.conv_layers[-1])
            utils.initialize_weights(self.conv_layers[-1])
            if use_batch_norm:
                self.bc_layers.append(nn.BatchNorm2d(h))
                setattr(self, 'bc_layer%d'%i, self.bc_layers[-1])
                utils.initialize_weights(self.bc_layers[-1])
            else:
                self.bc_layers.append(None)
            prev_hidden = h
        self.feat_size = self._get_feature_dim(D_shape_in)
        cur_dim = self.feat_size
        self.linear_layers = []
        self.l_bc_layers = []
        for i,d in enumerate(linear_hiddens):
            self.linear_layers.append(nn.Linear(cur_dim, d))
            setattr(self, 'linear_layer%d'%i, self.linear_layers[-1])
            utils.initialize_weights(self.linear_layers[-1])
            if use_batch_norm:
                self.l_bc_layers.append(nn.BatchNorm1d(d))
                setattr(self, 'l_bc_layer%d'%i, self.l_bc_layers[-1])
                utils.initialize_weights(self.l_bc_layers[-1])
            else:
                self.l_bc_layers.append(None)
            cur_dim = d
        self.final_size = cur_dim
        self.final_layers = []
        for i, d in enumerate(self.D_out):
            self.final_layers.append(nn.Linear(self.final_size, d))
            setattr(self, 'output_layer%d'%i, self.final_layers[-1])
            utils.initialize_weights(self.final_layers[-1], small_init=True)

    ######################
    def _forward_feature(self, x):
        for conv, bc in zip(self.conv_layers, self.bc_layers):
            x = conv(x)
            if bc is not None:
                x = bc(x)
            x = self.func(x)
            if common.debugger is not None:
                common.debugger.print("------>[P] Forward of Conv<{}>, Norm = {}, Var = {}, Max = {}, Min = {}".format(
                                    conv, x.data.norm(), x.data.var(), x.data.max(), x.data.min()), False)
        return x

    def _get_feature_dim(self, D_shape_in):
        bs = 1
        inp = Variable(torch.rand(bs, *D_shape_in))
        out_feat = self._forward_feature(inp)
        print('>> Final CNN Shape = {}'.format(out_feat.size()))
        n_size = out_feat.data.view(bs, -1).size(1)
        print('Feature Size = %d' % n_size)
        return n_size
    #######################
    def _get_concrete_stats(self, linear, feat, gumbel_noise = 1.0):
        logits = linear(feat)
        if gumbel_noise is not None:
            u = torch.rand(logits.size()).type(FloatTensor)
            eps = 1e-15  # IMPORTANT!!!!
            x = Variable(torch.log(-torch.log(u + eps) + eps))
            logits_with_noise = logits * gumbel_noise - x
            prob = F.softmax(logits_with_noise, dim=-1)
            logp = F.log_softmax(logits_with_noise, dim=-1)
        else:
            prob = F.softmax(logits, dim=-1)
            logp = F.log_softmax(logits, dim=-1)
        return logits, prob, logp

    def forward(self, x, gumbel_noise = 1.0):
        """
        compute the forward pass of the model.
        return logits and the softmax prob w./w.o. gumbel noise
        """
        self.feat = feat = self._forward_feature(x)
        feat = feat.view(-1, self.feat_size)
        common.debugger.print("------>[P] Forward of Policy, Feature Norm = {}, Var = {}, Max = {}, Min = {}".format(
                                feat.data.norm(), feat.data.var(), feat.data.max(), feat.data.min()), False)
        for l,bc in zip(self.linear_layers, self.l_bc_layers):
            feat = l(feat)
            if bc is not None:
                feat = bc(feat)
            feat = self.func(feat)
            common.debugger.print("------>[P] Forward of Policy, Mid-Feature Norm = {}, Var = {}, Max = {}, Min = {}".format(
                                    feat.data.norm(), feat.data.var(), feat.data.max(), feat.data.min()), False)
        self.logits = []
        self.logp = []
        self.prob = []
        for l in self.final_layers:
            _logits, _prob, _logp = self._get_concrete_stats(l, feat, gumbel_noise)
            self.logits.append(_logits)

            common.debugger.print("------>[P] Forward of Policy, Logits{}, Norm = {}, Var = {}, Max = {}, Min = {}".format(
                                    _logits.size(1), _logits.data.norm(), _logits.data.var(), _logits.data.max(), _logits.data.min()), False)
            common.debugger.print("------>[P] Forward of Policy, LogP{}, Norm = {}, Var = {}, Max = {}, Min = {}".format(
                                    _logp.size(1), _logp.data.norm(), _logp.data.var(), _logp.data.max(), _logp.data.min()), False)

            self.logp.append(_logp)
            self.prob.append(_prob)
        return self.prob  #self.logp

    ########################
    def logprob(self, actions, logp = None):
        if logp is None: logp = self.logp
        ret = 0
        for a, p in zip(actions, logp):
            ret += torch.sum(a * p, -1)
        return ret

    def entropy(self, logits=None):
        if logits is None: logits = self.logits
        ret = 0
        for l, d in zip(logits, self.D_out):
            a0 = l - torch.max(l, dim=1)[0].repeat(1, d)
            ea0 = torch.exp(a0)
            z0 = ea0.sum(1).repeat(1, d)
            p0 = ea0 / z0
            ret = ret + torch.sum(p0 * (torch.log(z0 + 1e-8) - a0), dim=1)
        return ret
