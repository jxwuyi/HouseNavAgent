import random
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
import utils

class CNNGumbelPolicy(torch.nn.Module):
    def __init__(self, D_shape_in, D_out, hiddens, kernel_sizes=5, strides=2,
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

        self.all_params = []
        self.conv_layers = []
        self.bc_layers = []
        prev_hidden = D_shape_in[0]
        for h, k, s in zip(self.hiddens, self.kernel_sizes, self.strides):
            self.conv_layers.append(nn.Conv2d(prev_hidden, h, kernel_size=k, stride=s))
            self.all_params += list(self.conv_layers[-1].parameters())
            if use_batch_norm:
                self.bc_layers.append(nn.BatchNorm2d(h))
                self.all_params += list(self.bc_layers[-1].parameters())
            else:
                self.bc_layers.append(None)
            prev_hidden = h
        self.feat_size = self._get_feature_dim(D_shape_in)
        print('Feature Size = %d' % self.feat_size)
        self.linear_layers = []
        for d in self.D_out:
            self.linear_layers.append(nn.Linear(self.feat_size, d))
            self.all_params += list(self.linear_layers[-1].parameters())

    ######################
    def parameters(self):
        return self.all_params

    ######################
    def _forward_feature(self, x):
        for conv, bc in zip(self.conv_layers, self.bc_layers):
            x = conv(x)
            if bc is not None:
                x = bc(x)
            x = self.func(x)
        return x

    def _get_feature_dim(self, D_shape_in):
        bs = 1
        inp = Variable(torch.rand(bs, *D_shape_in))
        out_feat = self._forward_feature(inp)
        n_size = out_feat.data.view(bs, -1).size(1)
        return n_size
    #######################
    def _get_concrete_stats(self, linear, feat, gumbel_noise = True):
        logits = linear(feat)
        if gumbel_noise:
            u = torch.rand(logits.size())
            x = Variable(torch.log(-torch.log(u)))
            logits_with_noise = logits - x
            prob = F.softmax(logits_with_noise)
            logp = F.log_softmax(logits_with_noise)
        else:
            prob = F.softmax(logits)
            logp = F.log_softmax(logits)
        return logits, prob, logp

    def forward(self, x, gumbel_noise = True):
        """
        compute the forward pass of the model.
        return logits and the softmax prob w./w.o. gumbel noise
        """
        self.feat = feat = self._forward_feature(x)
        feat = feat.view(-1, self.feat_size)
        self.logits = []
        self.logp = []
        self.prob = []
        for l in self.linear_layers:
            _logits, _prob, _logp = self._get_concrete_stats(l, feat, gumbel_noise)
            self.logits.append(_logits)
            self.logp.append(_logp)
            self.prob.append(_prob)
        return self.logits, self.logp, self.prob

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
        for l,d in zip(logits, self.D_out):
            a0 = l - torch.max(l, axis=1, keepdims=True).repeat(1, d)
            ea0 = torch.exp(a0)
            z0 = ea0.sum(1).repeat(1, d)
            p0 = ea0 / z0
            ret = ret + torch.sum(p0 * (torch.log(z0) - a0), dim=1)
        return ret
