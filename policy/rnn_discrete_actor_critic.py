from headers import *
import common
import random
import utils
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable

class DiscreteRNNPolicyCritic(torch.nn.Module):
    def __init__(self, D_shape_in, D_out,
                 conv_hiddens = [], kernel_sizes=5, strides=2,
                 linear_hiddens = [],
                 policy_hiddens = [],
                 critic_hiddens = [],
                 rnn_cell = 'lstm', rnn_layers=1, rnn_units=128,
                 activation=F.relu, use_batch_norm = True):
        """
        D_shape_in: (n_channel, n_row, n_col)
        D_out: a int or a list of ints in length of degree of freedoms
        hiddens, kernel_sizes, strides: either an int or a list of ints with the same length
        """
        super(DiscreteRNNPolicyCritic, self).__init__()
        if conv_hiddens is None: conv_hiddens = []
        if kernel_sizes is None: kernel_sizes = []
        if strides is None: strides = []
        if isinstance(conv_hiddens, int): conv_hiddens = [conv_hiddens]
        if isinstance(kernel_sizes, int): kernel_sizes = [kernel_sizes]
        if isinstance(strides, int): strides = [strides]
        self.cnn_layers = len(conv_hiddens)
        self.cnn_hiddens = conv_hiddens
        self.cnn_kernel_sizes = kernel_sizes
        self.cnn_strides = strides
        if len(self.cnn_kernel_sizes) == 1: self.cnn_kernel_sizes = self.cnn_kernel_sizes * self.cnn_layers
        if len(self.cnn_strides) == 1: self.cnn_strides = self.cnn_strides * self.cnn_layers

        assert ((len(self.cnn_kernel_sizes) == len(self.cnn_hiddens)) and (len(self.cnn_hiddens) == len(self.cnn_strides)))

        assert isinstance(D_out, int), '[DiscreteRNNPolicy] D_out must be an interger!'
        self.out_dim = D_out
        self.in_shape = D_shape_in
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
        self.feat_size = feat_size = self._get_feature_dim(D_shape_in)
        print('Feature Size = %d' % self.feat_size)

        # extra linear layers
        for i, d in enumerate(linear_hiddens):
            self.conv_layers.append(nn.Linear(feat_size, d))
            setattr(self, 'linear_layer%d'%i, self.conv_layers[-1])
            utils.initialize_weights(self.conv_layers[-1])
            if use_batch_norm:
                self.bc_layers.append(nn.BatchNorm1d(d))
                setattr(self, 'l_bc_layer%d'%i, self.bc_layers[-1])
                utils.initialize_weights(self.bc_layers[-1])
            else:
                self.bc_layers.append(None)

        self.rnn_input_size = feat_size

        # build rnn
        self.cell_type = rnn_cell
        cell_obj = nn.LSTM if rnn_cell == 'lstm' else nn.GRU
        self.cell = cell_obj(input_size=self.rnn_input_size,
                             hidden_size=self.rnn_units,
                             num_layers=self.rnn_layers,
                             batch_first=True)
        utils.initialize_weights(self.cell)
        self.rnn_output_size = self.rnn_units

        # build policy layers
        policy_hiddens.append(self.D_out)
        self.policy_layers = []
        cur_dim = self.rnn_output_size
        for i,d in enumerate(policy_hiddens):
            self.policy_layers.append(nn.Linear(cur_dim, d))
            setattr(self, 'policy_layer%d'%i, self.policy_layers[-1])
            utils.initialize_weights(self.policy_layers[-1], True)  # small weight init
            cur_dim = d

        # build critic layers
        critic_hiddens.append(1)
        self.critic_layers = []
        cur_dim = self.rnn_output_size
        for i,d in enumerate(critic_hiddens):
            self.critic_layers.append(nn.Linear(cur_dim, d))
            setattr(self, 'critic_layers%d'%i, self.critic_layers[-1])
            utils.initialize_weights(self.critic_layers[-1], True)  # small weight init
            cur_dim = d


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
        n_size = out_feat.data.view(bs, -1).size(1)
        return n_size

    def _get_zero_state(self, batch):
        z = Variable(torch.zeros(self.rnn_layers, batch, self.rnn_units)).type(FloatTensor)
        if self.cell_type == 'lstm':
            return (z, z)
        else:  # gru
            return z
    #######################


    #<<<<<<<< TODO Start From Here <<<<<<<<
    def forward(self, x, h=None, return_value=True, only_value=False):
        """
        compute the forward pass of the model.
        return logits and the softmax prob w./w.o. gumbel noise
        x: (batch, seq_len, channel, n_row, n_col)
        act: (batch, seq_len, act_dim) or a list of tensors
        """
        seq_len = x.size(1)
        batch = x.size(0)
        packed_x = x.view(-1, self.in_shape[0], self.in_shape[1], self.in_shape[2])
        self.feat = feat = self._forward_feature(packed_x)
        feat = feat.view(batch, seq_len, self.feat_size)

        common.debugger.print("------>[P] Forward of Policy, Feature Norm = {}, Var = {}, Max = {}, Min = {}".format(
                                feat.data.norm(), feat.data.var(), feat.data.max(), feat.data.min()), False)

        if isinstance(act, list):
            feat = torch.cat([feat] + act, dim=-1)
        else:
            if act is None: act = Variable(torch.zeros(batch, seq_len, self.out_dim)).type(FloatTensor)
            feat = torch.cat([feat, act], dim=-1)

        if h is None: h = self._get_zero_state(batch)
        # output: [batch, seq_len, rnn_output_size]
        outputs, new_h = self.cell(feat, h)

        feat = outputs.resize(batch * seq_len, self.rnn_output_size)
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
            _logits, _prob, _logp = self._get_concrete_stats(batch, seq_len, l, feat, gumbel_noise)
            self.logits.append(_logits)

            common.debugger.print("------>[P] Forward of Policy, Logits{}, Norm = {}, Var = {}, Max = {}, Min = {}".format(
                                    _logits.size(-1), _logits.data.norm(), _logits.data.var(), _logits.data.max(), _logits.data.min()), False)
            common.debugger.print("------>[P] Forward of Policy, LogP{}, Norm = {}, Var = {}, Max = {}, Min = {}".format(
                                    _logp.size(-1), _logp.data.norm(), _logp.data.var(), _logp.data.max(), _logp.data.min()), False)

            self.logp.append(_logp)
            self.prob.append(_prob)
        return self.prob, new_h  #self.logp

    ########################
    def logprob(self, actions, logp = None):
        if logp is None: logp = self.logp
        ret = 0
        for a, p in zip(actions, logp):
            ret += torch.sum(a * p, -1)
        return ret

    def entropy(self, logits=None, weight=None):
        """
        weight: (batch, seq_len)
        """
        if logits is None: logits = self.logits
        if weight is not None: weight = weight.view(-1)
        ret = 0
        for _l, d in zip(logits, self.D_out):
            l = _l.view(-1, d)
            a0 = l - torch.max(l, dim=1)[0].repeat(1, d)
            ea0 = torch.exp(a0)
            z0 = ea0.sum(1).repeat(1, d)
            p0 = ea0 / z0
            cur_ent = torch.sum(p0 * (torch.log(z0 + 1e-8) - a0), dim=1).squeeze()
            if weight is not None:
                cur_ent = cur_ent * weight
            ret = ret + cur_ent
        return ret
