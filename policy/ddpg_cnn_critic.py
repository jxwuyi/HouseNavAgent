from headers import *
import random
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable

class DDPGCNNCritic(torch.nn.Module):
    def __init__(self, D_shape_in, A_dim,
                conv_hiddens, kernel_sizes=5, strides=2,
                linear_hiddens = [], D_out = 1,
                activation=F.relu, use_batch_norm = True):
        """
        D_shape_in: tupe of two ints, the shape of input images
        D_out: a int or a list of ints in length of degree of freedoms
        hiddens, kernel_sizes, strides: either an int or a list of ints with the same length
        """
        super(DDPGCNNCritic, self).__init__()
        if isinstance(conv_hiddens, int): conv_hiddens = [conv_hiddens]
        if isinstance(linear_hiddens, int): linear_hiddens = [linear_hiddens]
        if isinstance(kernel_sizes, int): kernel_sizes = [kernel_sizes]
        if isinstance(strides, int): strides = [strides]
        self.n_layer = max(len(conv_hiddens),len(kernel_sizes),len(strides))
        self.hiddens = conv_hiddens
        self.kernel_sizes = kernel_sizes
        self.strides = strides
        if len(self.hiddens) == 1: self.hiddens = self.hiddens * self.n_layer
        if len(self.kernel_sizes) == 1: self.kernel_sizes = self.kernel_sizes * self.n_layer
        if len(self.strides) == 1: self.strides = self.strides * self.n_layer

        assert ((len(self.hiddens) == len(self.kernel_sizes)) and (len(self.strides) == len(self.hiddens))), '[CNNGumbelPolicy] hiddens, kernel_sizes, strides must share the same length'

        assert (isinstance(D_out, int))

        self.out_dim = D_out
        self.func = activation

        self.conv_layers = []
        self.bc_layers = []
        prev_hidden = D_shape_in[0]
        for i, dat in enumerate(zip(self.hiddens, self.kernel_sizes, self.strides)):
            h, k, s = dat
            self.conv_layers.append(nn.Conv2d(prev_hidden, h, kernel_size=k, stride=s))
            setattr(self, 'conv_layer%d'%i, self.conv_layers[-1])
            if use_batch_norm:
                self.bc_layers.append(nn.BatchNorm2d(h))
                setattr(self, 'bc_layer%d'%i, self.bc_layers[-1])
            else:
                self.bc_layers.append(None)
            prev_hidden = h
        self.feat_size = self._get_feature_dim(D_shape_in)
        print('Feature Size = %d' % self.feat_size)
        self.linear_layers = []
        prev_dim = self.feat_size + A_dim
        linear_hiddens.append(self.out_dim)
        for i, d in enumerate(linear_hiddens):
            self.linear_layers.append(nn.Linear(prev_dim, d))
            setattr(self, 'linear_layer%d'%i, self.linear_layers[-1])
            prev_dim = d

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

    def forward(self, x, act):
        """
        compute the forward pass of the model.
        return logits and the softmax prob w./w.o. gumbel noise
        """
        self.feat = val = self._forward_feature(x)
        val = val.view(-1, self.feat_size)
        if isinstance(act, list):
            val = torch.cat([val] + act, dim=1)
        else:
            val = torch.cat([val, act], dim=1)
        for i, l in enumerate(self.linear_layers):
            if i > 0:
                val = self.func(val)
            val = l(val)
        self.val = val.squeeze()
        return val
