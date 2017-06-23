from headers import *
import common
import random
import utils
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable

class RNNCritic(torch.nn.Module):
    def __init__(self, D_shape_in, A_dim,
                conv_hiddens=[], kernel_sizes=5, strides=2,
                rnn_cell = 'lstm', rnn_layers=1, rnn_units=128,
                linear_hiddens = [], D_out = 1,
                activation=F.relu, use_batch_norm = True):
        """
        D_shape_in: (channel, n_row, n_col)
        D_out: a int or a list of ints in length of degree of freedoms
        hiddens, kernel_sizes, strides: either an int or a list of ints with the same length
        """
        super(RNNCritic, self).__init__()
        if conv_hiddens is None: conv_hiddens = []
        if kernel_sizes is None: kernel_sizes = []
        if strides is None: strides = []
        if isinstance(conv_hiddens, int): conv_hiddens = [conv_hiddens]
        if isinstance(linear_hiddens, int): linear_hiddens = [linear_hiddens]
        if isinstance(kernel_sizes, int): kernel_sizes = [kernel_sizes]
        if isinstance(strides, int): strides = [strides]
        self.cnn_layers = len(conv_hiddens)
        self.cnn_hiddens = conv_hiddens
        self.cnn_kernel_sizes = kernel_sizes
        self.cnn_strides = strides
        if len(self.cnn_kernel_sizes) == 1: self.cnn_kernel_sizes = self.cnn_kernel_sizes * self.cnn_layers
        if len(self.cnn_strides) == 1: self.cnn_strides = self.cnn_strides * self.cnn_layers

        assert ((len(self.cnn_layers) == len(self.cnn_hiddens)) and (len(self.cnn_hiddens) == len(self.cnn_strides)))
        assert isinstance(D_out, int)

        self.in_shape = D_shape_in
        self.out_dim = D_out
        self.func = activation
        self.rnn_layers = rnn_layers
        self.rnn_units = rnn_units

        # build convolutional neural net
        self.conv_layers = []
        self.bc_layers = []
        prev_hidden = D_shape_in[0]
        for i, dat in enumerate(zip(self.cnn_hiddens, self.cnn_kernel_sizes, self.cnn_strides)):
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
        print('Feature Size = %d' % self.feat_size)
        self.rnn_input_size = self.feat_size + A_dim

        # build rnn
        self.cell_type = rnn_cell
        cell_obj = nn.LSTM if rnn_cell == 'lstm' else nn.GRU
        self.cell = cell_obj(input_size=self.rnn_input_size,
                             hidden_size=self.rnn_units,
                             num_layers=self.rnn_layers,
                             batch_first=True)
        utils.initialize_weights(self.cell)

        # build final linear layers
        self.rnn_output_size = self.rnn_units
        self.linear_layers = []
        prev_dim = self.rnn_output_size
        linear_hiddens.append(self.out_dim)
        for i, d in enumerate(linear_hiddens):
            self.linear_layers.append(nn.Linear(prev_dim, d))
            setattr(self, 'linear_layer%d'%i, self.linear_layers[-1])
            utils.initialize_weights(self.linear_layers[-1])
            prev_dim = d

    ######################
    def _forward_feature(self, x):
        for conv, bc in zip(self.conv_layers, self.bc_layers):
            x = conv(x)
            if bc is not None:
                x = bc(x)
            x = self.func(x)
            if common.debugger is not None:
                common.debugger.print("------>[C] Forward of Conv<{}>, Norm = {}, Var = {}, Max = {}, Min = {}".format(
                                    conv, x.data.norm(), x.data.var(), x.data.max(), x.data.min()), False)
        return x

    def _get_feature_dim(self, D_shape_in):
        bs = 1
        inp = Variable(torch.rand(bs, *D_shape_in))
        out_feat = self._forward_feature(inp)
        n_size = out_feat.data.view(bs, -1).size(-1)
        return n_size

    def _get_zero_state(self, batch):
        z = Variable(torch.zeros(batch, self.rnn_layers, self.rnn_units)).type(FloatTensor)
        if self.cell_type == 'lstm':
            return (z, z)
        else:  # gru
            return z
    #######################

    def forward(self, x, act, new_act=None):
        # TODO: to allow history act and new act!!!
        """
        x: (batch, seq_len, n_channel, n_row, n_col)
        act: (seq_len, batch, dim)
        """
        seq_len = x.size(1)
        batch = x.size(0)
        packed_x = x.view(-1, self.in_shape[0], self.in_shape[1], self.in_shape[2])
        self.feat = val = self._forward_feature(packed_x)
        val = val.view(batch, seq_len, self.feat_size)

        common.debugger.print("------>[C] Forward of Critic, Feature Norm = {}, Var = {}, Max = {}, Min = {}".format(
                                val.data.norm(), val.data.var(), val.data.max(), val.data.min()), False)

        raw_val = val
        if isinstance(act, list):
            val = torch.cat([raw_val] + act, dim=-1)
            if new_act is not None:
                new_val = torch.cat([raw_val] + new_act, dim=-1)
            common.debugger.print("                              Norm of Action = {}".format([a.data.norm() for a in act]) , False)
        else:
            val = torch.cat([raw_val, act], dim=-1)
            if new_act is not None:
                new_val = torch.cat([raw_val, new_act], dim=-1)
            common.debugger.print("                              Norm of Action = {}".format(act.data.norm()), False)

        # input rnn
        if new_act is None:  # simpley call rnn protocal
            outputs, _ = self.cell(val, self._get_zero_state(batch))
            val = outputs.view(-1, self.rnn_output_size)
        else:
            # run a for loop
            h = self._get_zero_state(batch)
            outputs = []
            for i in range(seq_len):
                cur_input = val[:, i:i+1, ...]
                cur_input_new = new_val[:, i:i+1, ...]
                cur_output, _ = self.cell(cur_input_new, h)
                outputs.append(cur_output)
                _, h = self.cell(cur_input, h)  # propagate raw history
            packed_outputs = torch.cat(outputs, dim=1)
            val = packed_outputs.view(-1, self.rnn_output_size)

        for i, l in enumerate(self.linear_layers):
            if i > 0:
                val = self.func(val)
            val = l(val)
        val = val.view(batch, seq_len)
        self.val = val.squeeze()
        return val
