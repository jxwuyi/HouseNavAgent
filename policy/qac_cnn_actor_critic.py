from headers import *
import common
import random
import utils
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable

class DiscreteCNNPolicyQFunc(torch.nn.Module):
    def __init__(self, D_shape_in, D_out,
                 cnn_hiddens, kernel_sizes=5, strides=2,
                 linear_hiddens=[],
                 critic_hiddens=[],
                 act_hiddens=[],
                 activation=F.relu, use_batch_norm=True,
                 only_q_network=False,
                 multi_target=False,  # whether to train target embedding
                 target_embedding_dim=25,  # embedding dimension of target instruction
                 use_target_gating=False
                 ):
        """
        D_shape_in: tupe of two ints, the shape of input images
        D_out: a int or a list of ints in length of degree of freedoms
        cnn_hiddens, kernel_sizes, strides: either an int or a list of ints with the same length
        """
        super(DiscreteCNNPolicyQFunc, self).__init__()
        if isinstance(cnn_hiddens, int): cnn_hiddens = [cnn_hiddens]
        if isinstance(kernel_sizes, int): kernel_sizes = [kernel_sizes]
        if isinstance(strides, int): strides = [strides]
        self.n_layer = max(len(cnn_hiddens),len(kernel_sizes),len(strides))
        self.cnn_hiddens = cnn_hiddens
        self.kernel_sizes = kernel_sizes
        self.strides = strides
        self.multi_target = multi_target
        self.use_target_gating = multi_target and use_target_gating
        if len(self.cnn_hiddens) == 1: self.cnn_hiddens = self.cnn_hiddens * self.n_layer
        if len(self.kernel_sizes) == 1: self.kernel_sizes = self.kernel_sizes * self.n_layer
        if len(self.strides) == 1: self.strides = self.strides * self.n_layer

        assert ((len(self.cnn_hiddens) == len(self.kernel_sizes)) and (len(self.strides) == len(self.cnn_hiddens))), \
                '[DiscreteCNNPolicyQFunc] cnn_hiddens, kernel_sizes, strides must share the same length'

        assert isinstance(D_out, int), '[DiscreteCNNPolicyQFunc] output dimension must be an integer!'
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
        self.final_size = cur_dim
        # multi-target instructions
        if multi_target:
            self.target_embed = nn.Linear(common.n_target_instructions, target_embedding_dim, bias=False)
            utils.initialize_weights(self.target_embed)
            self.target_trans = []
            if use_target_gating:
                self.target_trans.append(nn.Linear(target_embedding_dim, cur_dim))
                setattr(self, 'target_transform_layer0', self.target_trans[-1])
                utils.initialize_weights(self.target_trans[-1])
            else:
                self.final_size += target_embedding_dim
        # Output Action
        cur_dim = self.final_size
        self.policy_layers = []
        if not only_q_network:
            act_hiddens.append(D_out)
            for i, d in enumerate(act_hiddens):
                self.policy_layers.append(nn.Linear(cur_dim, d))
                setattr(self, 'policy_layer%d'%i, self.policy_layers[-1])
                utils.initialize_weights(self.policy_layers[-1], small_init=True)
                cur_dim = d
        # Output Q-Value
        #   dueling network
        critic_hiddens.append(1)
        self.critic_size = cur_dim = self.final_size
        self.val_layers = []
        self.adv_layers = []
        for i, d in enumerate(critic_hiddens):
            self.val_layers.append(nn.Linear(cur_dim, d))
            setattr(self, 'val_layer%d'%i, self.val_layers[-1])
            utils.initialize_weights(self.val_layers[-1], small_init=True)
            adv_d = d if i + 1 < len(critic_hiddens) else self.out_dim
            self.adv_layers.append(nn.Linear(cur_dim, adv_d))
            setattr(self, 'adv_layer%d'%i, self.adv_layers[-1])
            utils.initialize_weights(self.val_layers[-1], small_init=True)
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
    def forward(self, x, return_q_value=True, only_q_value=False, sample_action=True, return_act_prob=False, target=None):
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
        # Insert Multi-Target Instruction
        if self.multi_target:
            assert target is not None
            target = self.target_embed(target)
            if self.use_target_gating:
                for i, l in enumerate(self.target_trans):
                    target = l(target)
                    if i + 1 < len(self.target_trans):
                        target = F.relu(target)
                target = F.sigmoid(target)
                raw_feat = feat = feat * target
            else:
                raw_feat = feat = torch.cat([feat, target], dim=-1)
        # Compute Critic
        # >> Dueling Network
        #    -> compute value
        for i,l in enumerate(self.val_layers):
            if i > 0:
                feat = self.func(feat)
            feat = l(feat)
        self.val = feat
        #    -> compute advantage
        feat = raw_feat
        for i,l in enumerate(self.adv_layers):
            if i > 0:
                feat = self.func(feat)
            feat = l(feat)
        self.adv = feat
        # compute final q_value
        adv_mean = feat.mean(dim=1, keepdim=True)  # [batch, 1]
        dim = self.out_dim
        self.q_val = self.val + self.adv - adv_mean
        if only_q_value:
            return self.q_val
        # Compute Action
        feat = raw_feat
        for i,l in enumerate(self.policy_layers):
            if i > 0:
                feat = self.func(feat)
            feat = l(feat)
        self.logits = feat
        self.prob = F.softmax(feat, dim=-1)
        self.logp = F.log_softmax(feat, dim=-1)
        if sample_action:
            self.act = self.prob.multinomial()
        else:
            self.act = self.logits.max(dim=1, keepdim=True)[1]
        ret_act = self.prob if return_act_prob else self.act
        if return_q_value:
            return ret_act, self.q_val
        else:
            return ret_act

    ########################
    def logprob(self, action, logp = None):
        # action is a LongTensor!
        if len(action.size()) == 1:
            action = action.view(-1,1)
        if logp is None: logp = self.logp
        #onehot = torch.zeros(logp.size())
        #onehot.scatter_(1, action, 1.0)
        #onehot = Variable(onehot).type(FloatTensor)
        ret = torch.gather(logp, 1, action)
        #ret = torch.sum(ret, dim=-1)
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
