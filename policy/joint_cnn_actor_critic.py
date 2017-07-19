from headers import *
import common
import random
import utils
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable

class JointCNNPolicyCritic(torch.nn.Module):
    def __init__(self, D_shape_in, D_out,
                cnn_hiddens, kernel_sizes=5, strides=2,
                linear_hiddens=[], critic_hiddens=[], policy_hiddens=[],
                activation=F.relu, use_batch_norm = True,
                transform_hiddens=[],
                use_action_gating = False,
                use_residual = False):
        """
        D_shape_in: tupe of two ints, the shape of input images
        D_out: a int or a list of ints in length of degree of freedoms
        cnn_hiddens, kernel_sizes, strides: either an int or a list of ints with the same length
        """
        super(JointCNNPolicyCritic, self).__init__()
        if isinstance(cnn_hiddens, int): cnn_hiddens = [cnn_hiddens]
        if isinstance(kernel_sizes, int): kernel_sizes = [kernel_sizes]
        if isinstance(strides, int): strides = [strides]
        self.n_layer = max(len(cnn_hiddens),len(kernel_sizes),len(strides))
        self.cnn_hiddens = cnn_hiddens
        self.kernel_sizes = kernel_sizes
        self.strides = strides
        self.use_residual = use_residual
        assert not use_residual, '[JointCNNPolicyCritic] Current do not support resnet'
        if len(self.cnn_hiddens) == 1: self.cnn_hiddens = self.cnn_hiddens * self.n_layer
        if len(self.kernel_sizes) == 1: self.kernel_sizes = self.kernel_sizes * self.n_layer
        if len(self.strides) == 1: self.strides = self.strides * self.n_layer

        assert ((len(self.cnn_hiddens) == len(self.kernel_sizes)) and (len(self.strides) == len(self.cnn_hiddens))), \
                '[JointCNNPolicy] cnn_hiddens, kernel_sizes, strides must share the same length'

        if isinstance(D_out, int):
            self.D_out = [D_out]
        else:
            self.D_out = D_out
        self.out_dim = sum(D_out)
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
        # Output Action
        self.final_size = cur_dim
        self.policy_layers = []
        for i, d in enumerate(policy_hiddens):
            self.policy_layers.append(nn.Linear(cur_dim, d))
            setattr(self, 'policy_layer%d'%i, self.policy_layers[-1])
            utils.initialize_weights(self.policy_layers[-1], small_init=True)
            cur_dim = d
        self.output_layers = []
        for i, d in enumerate(self.D_out):
            self.output_layers.append(nn.Linear(cur_dim, d))
            setattr(self, 'output_layer%d'%i, self.output_layers[-1])
            utils.initialize_weights(self.output_layers[-1], small_init=True)
        # Output Critic
        self.use_action_gating = use_action_gating
        #  >> transform dimension of actions
        cur_dim = self.out_dim
        self.trans_layers = []
        if self.use_action_gating and \
            ((len(transform_hiddens) == 0) or \
                (transform_hiddens[-1] != self.final_size)):
            transform_hiddens.append(self.final_size)
        for i, d in enumerate(transform_hiddens):
            self.trans_layers.append(nn.Linear(cur_dim, d))
            setattr(self, 'transform_layer%d'%i, self.trans_layers[-1])
            utils.initialize_weights(self.trans_layers[-1])
            cur_dim = d
        self.critic_size = self.final_size
        if not self.use_action_gating:
            self.critic_size += cur_dim
        cur_dim = self.critic_size
        self.critic_layers = []
        if (len(critic_hiddens) == 0) or (critic_hiddens[-1] != 1):
            critic_hiddens.append(1)
        for i, d in enumerate(critic_hiddens):
            self.critic_layers.append(nn.Linear(cur_dim, d))
            setattr(self, 'critic_layer%d'%i, self.critic_layers[-1])
            utils.initialize_weights(self.critic_layers[-1], small_init=True)
            cur_dim = d

    ######################
    def clear_critic_specific_grad(self):
        q_layers = self.critic_layers + self.trans_layers
        for l in q_layers:
            for p in l.parameters():
                if p.grad is not None:
                    if p.grad.volatile:
                        p.grad.data.zero_()
                    else:
                        data = p.grad.data
                        p.grad = Variable(data.new().resize_as_(data).zero_())

    ######################
    def _forward_feature(self, x):
        for s, conv, bc in zip(self.strides, self.conv_layers, self.bc_layers):
            raw_x = x
            x = conv(x)
            if bc is not None:
                x = bc(x)
            x = self.func(x)
            if (s == 1) and self.use_residual:
                x += raw_x  # skip connection
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
            prob = F.softmax(logits_with_noise)
            logp = F.log_softmax(logits_with_noise)
        else:
            prob = F.softmax(logits)
            logp = F.log_softmax(logits)
        return logits, prob, logp

    def forward(self, x, action=None, gumbel_noise = 1.0, output_critic = True):
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
        raw_feat = feat
        # Compute Action
        if action is None:
            for l in self.policy_layers:
                feat = self.func(l(feat))
            self.logits = []
            self.logp = []
            self.prob = []
            for l in self.output_layers:
                _logits, _prob, _logp = self._get_concrete_stats(l, feat, gumbel_noise)
                self.logits.append(_logits)

                common.debugger.print("------>[P] Forward of Policy, Logits{}, Norm = {}, Var = {}, Max = {}, Min = {}".format(
                                        _logits.size(1), _logits.data.norm(), _logits.data.var(), _logits.data.max(), _logits.data.min()), False)
                common.debugger.print("------>[P] Forward of Policy, LogP{}, Norm = {}, Var = {}, Max = {}, Min = {}".format(
                                        _logp.size(1), _logp.data.norm(), _logp.data.var(), _logp.data.max(), _logp.data.min()), False)

                self.logp.append(_logp)
                self.prob.append(_prob)
        if not output_critic:
            return self.prob
        if action is None:
            action = self.prob

        if isinstance(action, list):
            action = torch.cat(action, dim=-1)  # concatenate actions

        for i, l in enumerate(self.trans_layers):
            action = l(action)
            if self.use_action_gating and (i+1 == len(self.trans_layers)):
                action = F.sigmoid(action)
            else:
                action = self.func(action)

        feat = raw_feat
        if self.use_action_gating:
            feat = feat * action
        else:
            feat = torch.cat([feat, action], dim=-1)

        # compute critic
        for i,l in enumerate(self.critic_layers):
            if i > 0:
                feat = self.func(feat)
            feat = l(feat)
        val = feat.squeeze()
        return val

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
