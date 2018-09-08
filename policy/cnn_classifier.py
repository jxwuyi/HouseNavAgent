from headers import *
import common
import random
import utils
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable

class CNNClassifier(torch.nn.Module):
    def __init__(self, D_shape_in, n_class, hiddens, kernel_sizes=5, strides=2,
                 linear_hiddens=[32],
                 activation=F.elu, use_batch_norm=False,
                 multi_label=False):
        """
        D_shape_in: tupe of two ints, the shape of input images
        n_class: a int for number of semantic classes
        hiddens, kernel_sizes, strides: either an int or a list of ints with the same length
        """
        super(CNNClassifier, self).__init__()
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

        assert ((len(self.hiddens) == len(self.kernel_sizes)) and (len(self.strides) == len(self.hiddens))), '[CNNClassifier] hiddens, kernel_sizes, strides must share the same length'

        self.out_dim = n_class
        self.func = activation

        self.multi_label=multi_label
        if multi_label:
            print('[CNNClassifier] Multi_label is True, use <Sigmoid> output!')

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
        self.dropout_layers = []
        for i,d in enumerate(linear_hiddens):
            self.linear_layers.append(nn.Linear(cur_dim, d))
            setattr(self, 'linear_layer%d'%i, self.linear_layers[-1])
            utils.initialize_weights(self.linear_layers[-1])
            cur_dim = d

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
        print('>> Final CNN Shape = {}'.format(out_feat.size()))
        n_size = out_feat.data.view(bs, -1).size(1)
        print('Feature Size = %d' % n_size)
        return n_size
    #######################

    def forward(self, x, return_logits=False, return_logprob=False):
        """
        compute the forward pass of the model.
        return logits and the softmax prob w./w.o. gumbel noise
        """
        self.feat = feat = self._forward_feature(x)
        feat = feat.view(-1, self.feat_size)
        for i, l in enumerate(self.linear_layers):
            if i > 0: feat = self.func(feat)
            feat = l(feat)
        self.logits = feat
        if return_logits: return feat
        if return_logprob: 
            return F.log_softmax(feat) if not self.multi_label else F.logsigmoid(feat)
        return F.softmax(feat) if not self.multi_label else F.sigmoid(feat)

    ########################
    def logprob(self, label, logp):
        ret = 0
        if not self.multi_label:  # softmax
            ret = torch.sum(label * logp, -1)
        else:  # sigmoid
            p = torch.exp(logp)
            ret = p * label + (1 - p) * (1 - label)
            ret = torch.log(ret)
        return ret

    def entropy(self, logits=None):
        if logits is None: logits = self.logits
        ret = 0
        if not self.multi_label:  # softmax
            d = self.out_dim
            l = logits
            a0 = l - torch.max(l, dim=1)[0].repeat(1, d)
            ea0 = torch.exp(a0)
            z0 = ea0.sum(1).repeat(1, d)
            p0 = ea0 / z0
            ret = torch.sum(p0 * (torch.log(z0 + 1e-8) - a0), dim=1)
        else:  # sigmoid
            p = 1. / (1. + torch.exp(-logits))
            ret = p * torch.log(p + 1e-10) + (1 - p) * torch.log(1 - p + 1e-10)
        return ret
