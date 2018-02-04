from headers import *
import common
import random
import utils
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable

class DiscreteCNNPolicyCritic(torch.nn.Module):
    def __init__(self, D_shape_in, D_out,
                cnn_hiddens, kernel_sizes=5, strides=2,
                linear_hiddens=[],
                critic_hiddens=[],
                act_hiddens=[],
                activation=F.relu, use_batch_norm = True):
        """
        D_shape_in: tupe of two ints, the shape of input images
        D_out: a int or a list of ints in length of degree of freedoms
        cnn_hiddens, kernel_sizes, strides: either an int or a list of ints with the same length
        """
        super(DiscreteCNNPolicyCritic, self).__init__()
        if isinstance(cnn_hiddens, int): cnn_hiddens = [cnn_hiddens]
        if isinstance(kernel_sizes, int): kernel_sizes = [kernel_sizes]
        if isinstance(strides, int): strides = [strides]
        self.n_layer = max(len(cnn_hiddens),len(kernel_sizes),len(strides))
        self.cnn_hiddens = cnn_hiddens
        self.kernel_sizes = kernel_sizes
        self.strides = strides
        if len(self.cnn_hiddens) == 1: self.cnn_hiddens = self.cnn_hiddens * self.n_layer
        if len(self.kernel_sizes) == 1: self.kernel_sizes = self.kernel_sizes * self.n_layer
        if len(self.strides) == 1: self.strides = self.strides * self.n_layer

        assert ((len(self.cnn_hiddens) == len(self.kernel_sizes)) and (len(self.strides) == len(self.cnn_hiddens))), \
                '[DiscreteCNNPolicyCritic] cnn_hiddens, kernel_sizes, strides must share the same length'

        assert isinstance(D_out, int), '[DiscreteCNNPolicyCritic] output dimension must be an integer!'
        self.out_dim = D_out
        self.func=activation

        self.conv_layers = []
        self.bc_layers = []
        prev_hidden = D_shape_in[0]
        for i, dat in enumerate(zip(self.cnn_hiddens, self.kernel_sizes, self.strides)):
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
        for i,d in enumerate(linear_hiddens):
            self.linear_layers.append(nn.Linear(cur_dim, d))
            setattr(self, 'linear_layer%d'%i, self.linear_layers[-1])
            utils.initialize_weights(self.linear_layers[-1])
            cur_dim = d
        # Output Action
        act_hiddens.append(D_out)
        self.final_size = cur_dim
        self.policy_layers = []
        for i, d in enumerate(act_hiddens):
            self.policy_layers.append(nn.Linear(cur_dim, d))
            setattr(self, 'policy_layer%d'%i, self.policy_layers[-1])
            utils.initialize_weights(self.policy_layers[-1], small_init=True)
            cur_dim = d
        # Output Critic
        critic_hiddens.append(1)
        self.critic_size = cur_dim = self.final_size
        self.critic_layers = []
        for i, d in enumerate(critic_hiddens):
            self.critic_layers.append(nn.Linear(cur_dim, d))
            setattr(self, 'critic_layer%d'%i, self.critic_layers[-1])
            utils.initialize_weights(self.critic_layers[-1], small_init=True)
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
        print('>> Final CNN Shape = {}'.format(out_feat.size()))
        n_size = out_feat.data.view(bs, -1).size(1)
        print('Feature Size = %d' % n_size)
        return n_size

    #######################
    def forward(self, x, return_value=True, only_value=False):
        """
        compute the forward pass of the model.
        return logits and the softmax prob w./w.o. gumbel noise
        """
        self.feat = feat = self._forward_feature(x)
        feat = feat.view(-1, self.feat_size)
        common.debugger.print("------>[P] Forward of Policy, Feature Norm = {}, Var = {}, Max = {}, Min = {}".format(
                                feat.data.norm(), feat.data.var(), feat.data.max(), feat.data.min()), False)
        for l in self.linear_layers:
            feat = l(feat)
            feat = self.func(feat)
            common.debugger.print("------>[P] Forward of Policy, Mid-Feature Norm = {}, Var = {}, Max = {}, Min = {}".format(
                                    feat.data.norm(), feat.data.var(), feat.data.max(), feat.data.min()), False)
        raw_feat = feat
        # Compute Critic
        for i,l in enumerate(self.critic_layers):
            if i > 0:
                feat = self.func(feat)
            feat = l(feat)
        self.val = feat
        if only_value:
            return self.val
        feat = raw_feat
        # Compute Action
        for i,l in enumerate(self.policy_layers):
            if i > 0:
                feat = self.func(feat)
            feat = l(feat)
        self.logits = feat
        self.prob = F.softmax(feat, dim=-1)
        self.logp = F.log_softmax(feat, dim=-1)
        self.act = self.prob.multinomial()
        if return_value:
            return self.act, self.val
        else:
            return self.act

    ########################
    def logprob(self, action, logp = None):
        # action is a LongTensor!
        if len(action.size()) == 1:
            action = action.view(-1,1)
        action = action.cpu()
        if logp is None: logp = self.logp
        onehot = torch.zeros(logp.size())
        onehot.scatter_(1, action, 1.0)
        onehot = Variable(onehot).type(FloatTensor)
        ret = onehot * logp
        ret = torch.sum(ret, dim=-1)
        return ret

    def entropy(self, logits=None):
        if logits is None: logits = self.logits
        l = logits
        d = self.out_dim
        a0 = l - torch.max(l, dim=1)[0].repeat(1, d)
        ea0 = torch.exp(a0)
        z0 = ea0.sum(1).repeat(1, d)
        p0 = ea0 / z0
        ret = torch.sum(p0 * (torch.log(z0 + 1e-8) - a0), dim=1)
        return ret
