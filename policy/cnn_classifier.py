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
                 activation=F.relu, use_batch_norm=False,
                 multi_label=False,
                 dropout_rate=None,
                 stack_frame=None,
                 self_attention_dim=None):
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

        if (stack_frame is None) or (stack_frame <= 1):
            stack_frame = None
            self_attention_dim = None
        self.stack_frame = stack_frame
        self.attention_dim = self_attention_dim

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
            if h == 0:  # max pooling layer
                self.conv_layers.append(None)
                self.bc_layers.append((nn.MaxPool2d(k, stride=s, padding=(k//2))))
                setattr(self, 'max_pool%d'%i, self.bc_layers[-1][0])
                continue   # do not update prev_hidden
            self.conv_layers.append(nn.Conv2d(prev_hidden, h, kernel_size=k, stride=s))
            setattr(self, 'conv_layer%d'%i, self.conv_layers[-1])
            utils.initialize_weights(self.conv_layers[-1])
            if use_batch_norm or dropout_rate:
                cur_bc = []
                if use_batch_norm:
                    cur_bc.append(nn.BatchNorm2d(h))
                    utils.initialize_weights(cur_bc[-1])
                    setattr(self, 'bc_layer%d'%i, cur_bc[-1])
                if dropout_rate is not None:
                    cur_bc.append(nn.Dropout2d(dropout_rate))
                    setattr(self, 'dropout2d%d'%i, cur_bc[-1])
                self.bc_layers.append(cur_bc)
                #self.bc_layers.append(nn.BatchNorm2d(h))
                #setattr(self, 'bc_layer%d'%i, self.bc_layers[-1])
                #utils.initialize_weights(self.bc_layers[-1])
            else:
                self.bc_layers.append(None)
            prev_hidden = h

        #self.feat_size = self._get_feature_dim(D_shape_in)
        n_size, n_col, n_row = self._get_feature_dim(D_shape_in)
        self.avg_pool = nn.AvgPool2d((n_row, n_col))
        self.feat_size = prev_hidden
        if self.stack_frame:
            if self.attention_dim:
                self.att_trans = nn.Linear(prev_hidden * self.stack_frame, self.attention_dim)
                utils.initialize_weights(self.att_trans)
                self.att_proj = nn.Linear(prev_hidden, self.attention_dim)
                utils.initialize_weights(self.att_proj)
            

        cur_dim = self.feat_size
        linear_hiddens.append(n_class)
        self.linear_layers = []
        self.dropout_layers = []  #TODO: add dropout layers
        for i,d in enumerate(linear_hiddens):
            if (i > 0) and (dropout_rate is not None):
                self.dropout_layers.append(nn.Dropout(dropout_rate))
                setattr(self, 'dropout_layer%d' % (i - 1), self.dropout_layers[-1])
            self.linear_layers.append(nn.Linear(cur_dim, d))
            setattr(self, 'linear_layer%d'%i, self.linear_layers[-1])
            utils.initialize_weights(self.linear_layers[-1])
            cur_dim = d

    ######################
    def _forward_feature(self, x):
        for conv, bc in zip(self.conv_layers, self.bc_layers):
            if conv is not None:
                x = conv(x)
            if bc is not None:
                for l in bc:
                    x = l(x)
            if conv is not None:
                x = self.func(x)
        return x

    def _get_feature_dim(self, D_shape_in):
        bs = 1
        inp = Variable(torch.rand(bs, *D_shape_in))
        out_feat = self._forward_feature(inp)
        print('>> Final CNN Shape = {}'.format(out_feat.size()))
        n_size = out_feat.data.view(bs, -1).size(1)
        print('Feature Size = %d' % n_size)
        return n_size, out_feat.size(-2), out_feat.size(-1)
    #######################

    def forward(self, x, return_logits=False, return_logprob=False):
        """
        compute the forward pass of the model.
        return logits and the softmax prob w./w.o. gumbel noise
        x: shape is [batch, channel, n, m] or [batch, stack_frames, channel, n, m]
        """
        batch_size = x.size(0)
        if self.stack_frame:
            assert len(x.size()) == 5
            chn, n, m = x.size(2), x.size(3), x.size(4)
            x = x.view(-1, chn, n, m)
        self.feat = feat = self.avg_pool(self._forward_feature(x))
        feat = feat.view(-1, self.feat_size)
        if self.stack_frame:
            unpacked_feat = feat.view(batch_size, self.stack_frame, self.feat_size)
            if self.attention_dim:
                hidden = F.tanh(self.att_trans(unpacked_feat.view(batch_size, -1))).view(batch_size, 1, self.attention_dim)   # [batch, 1, att_dim]
                proj = self.att_proj(feat).view(batch_size, self.stack_frame, self.attention_dim)
                rep_hidden = hidden.repeat(1, self.stack_frame, 1)
                weight = F.softmax(torch.sum(proj * rep_hidden, dim=-1, keepdim=False)).view(batch_size, self.stack_frame, 1)   # attention weight, [batch, stack_frame, 1]
                att_feat = weight.repeat(1, 1, self.feat_size) * unpacked_feat   # [batch, stack_frame, feat_size]
                feat = torch.sum(att_feat, dim=1, keepdim=False)
            else:
                feat = torch.sum(unpacked_feat, dim=1, keepdim=False) / float(self.stack_frame)
        for i, l in enumerate(self.linear_layers):
            if i > 0:
                feat = self.func(feat)
                if i - 1 < len(self.dropout_layers):
                    feat = self.dropout_layers[i-1](feat)
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
            a0 = l - torch.max(l, dim=1, keepdim=True)[0].repeat(1, d)
            ea0 = torch.exp(a0)
            z0 = ea0.sum(1, keepdim=True).repeat(1, d)
            p0 = ea0 / z0
            ret = torch.sum(p0 * (torch.log(z0 + 1e-8) - a0), dim=1)
        else:  # sigmoid
            p = 1. / (1. + torch.exp(-logits))
            ret = p * torch.log(p + 1e-10) + (1 - p) * torch.log(1 - p + 1e-10)
        return ret
