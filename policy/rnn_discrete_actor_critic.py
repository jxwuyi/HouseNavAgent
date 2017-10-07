from headers import *
import common
import random
import utils
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable

class DiscreteRNNPolicy(torch.nn.Module):
    def __init__(self, D_shape_in, D_out,
                 conv_hiddens = [], kernel_sizes=5, strides=2,
                 linear_hiddens = [],
                 policy_hiddens = [],
                 critic_hiddens = [],
                 rnn_cell = 'lstm', rnn_layers=1, rnn_units=128,
                 activation=F.relu, use_batch_norm=True,
                 multi_target=False,  # whether to train target embedding
                 target_embedding_dim=25,  # embedding dimension of target instruction
                 use_target_gating=False
                 ):
        """
        D_shape_in: (n_channel, n_row, n_col)
        D_out: a int or a list of ints in length of degree of freedoms
        hiddens, kernel_sizes, strides: either an int or a list of ints with the same length
        """
        super(DiscreteRNNPolicy, self).__init__()
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
        self.multi_target = multi_target
        self.use_target_gating = multi_target and use_target_gating
        self.target_embed_dim = target_embedding_dim
        if len(self.cnn_kernel_sizes) == 1: self.cnn_kernel_sizes = self.cnn_kernel_sizes * self.cnn_layers
        if len(self.cnn_strides) == 1: self.cnn_strides = self.cnn_strides * self.cnn_layers

        assert ((len(self.cnn_kernel_sizes) == len(self.cnn_hiddens)) and (len(self.cnn_hiddens) == len(self.cnn_strides)))

        assert isinstance(D_out, int), '[DiscreteRNNPolicy] D_out must be an integer!'
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
        self.linear_layers = []
        self.ln_bc_layers = []
        for i, d in enumerate(linear_hiddens):
            self.linear_layers.append(nn.Linear(feat_size, d))
            setattr(self, 'linear_layer%d'%i, self.linear_layers[-1])
            utils.initialize_weights(self.linear_layers[-1])
            if use_batch_norm:
                self.ln_bc_layers.append(nn.BatchNorm1d(d))
                setattr(self, 'l_bc_layer%d'%i, self.ln_bc_layers[-1])
                utils.initialize_weights(self.ln_bc_layers[-1])
            else:
                self.ln_bc_layers.append(None)
            feat_size = d

        self.feat_size = feat_size
        self.rnn_input_size = feat_size

        # multi target instructions
        if self.multi_target:
            self.target_embed = nn.Linear(common.n_target_instructions, target_embedding_dim, bias=False)
            utils.initialize_weights(self.target_embed)
            self.target_trans = []
            if use_target_gating:
                self.target_trans.append(nn.Linear(target_embedding_dim, self.feat_size))
                setattr(self, 'target_transform_layer0', self.target_trans[-1])
                utils.initialize_weights(self.target_trans[-1])
            self.rnn_input_size += target_embedding_dim  # feat instruction to rnn!

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
        policy_hiddens.append(self.out_dim)
        self.policy_layers = []
        cur_dim = self.rnn_output_size + self.feat_size
        for i,d in enumerate(policy_hiddens):
            self.policy_layers.append(nn.Linear(cur_dim, d))
            setattr(self, 'policy_layer%d'%i, self.policy_layers[-1])
            utils.initialize_weights(self.policy_layers[-1], True)  # small weight init
            cur_dim = d

        # build critic layers
        critic_hiddens.append(1)
        self.critic_layers = []
        cur_dim = self.rnn_output_size + self.feat_size
        for i,d in enumerate(critic_hiddens):
            self.critic_layers.append(nn.Linear(cur_dim, d))
            setattr(self, 'critic_layers%d'%i, self.critic_layers[-1])
            utils.initialize_weights(self.critic_layers[-1], True)  # small weight init
            cur_dim = d


    ######################
    def _forward_feature(self, x, compute_linear=False):
        for conv, bc in zip(self.conv_layers, self.bc_layers):
            x = conv(x)
            if bc is not None:
                x = bc(x)
            x = self.func(x)
        if compute_linear:
            x = x.view(-1, self.feat_size)
            for l, bc in zip(self.linear_layers, self.ln_bc_layers):
                x = l(x)
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

    def get_zero_state(self, batch=1, return_variable=False, volatile=False):
        z = torch.zeros(self.rnn_layers, batch, self.rnn_units).type(FloatTensor)
        if return_variable: z = Variable(z, volatile=volatile)
        if self.cell_type == 'lstm':
            return (z, z)
        else:  # gru
            return z

    def _pack_hidden_states(self, hiddens):
        """
        :param hiddens: a list of hiddens
        :return: a packed tensor of hidden states, [layers, batch, units]
        """
        if self.cell_type == 'lstm':
            c = torch.cat([h[0] for h in hiddens], dim=1)
            g = torch.cat([h[1] for h in hiddens], dim=1)
            return (c, g)
        return torch.cat(hiddens, dim=1)

    def _unpack_hidden_states(self, hidden):
        """
        :param hidden: a tensor of hidden states [layers, batch, units]
        :return: unpack the states to a list of individual hiddens
        """
        if self.cell_type == 'lstm':
            batch = hidden[0].size(1)
            c = torch.chunk(hidden[0], batch, dim=1)
            g = torch.chunk(hidden[1], batch, dim=1)
            return [(c_i, g_i) for (c_i, g_i) in zip(c, g)]
        else:
            batch = hidden.size(1)
            return torch.chunk(hidden, batch, dim=1)

    def mark_hidden_states(self, hidden, done):
        """
        :param hidden: a tensor of hidden states [layer, batch, units]
        :param done: a float tensor of 0/1, whether an epis ends, [batch]
        :return: a marked hidden
        """
        done = 1.0 - done
        done = done.view(1, -1, 1)  # torch 0.2 required
        if self.cell_type == 'lstm':
            hidden = (hidden[0] * done, hidden[1] * done)
        else:
            hidden = hidden * done
        return hidden

    #######################

    def forward(self, x, h, only_value = False, return_value=True, sample_action=False,
                unpack_hidden=False, return_tensor=False, target=None):
        """
        compute the forward pass of the model.
        @:param x: [seq_len, batch, n_channel, n_row, n_col]
        @:param h: [layer, batch, units] or a list of <batch_size> individual hiddens
        @:param return_value: when False, only return action
        @:param sample_action: when True, action will be the sampled LongTensor, [batch, seq_len, 1]
        @:param target: when self.multi_target, target will be one-hot matrix of [batch, seq_len, n_target_instructions]
        @:return (action, value, hiddens) or (action, hiddens)
        """
        seq_len = x.size(1)
        batch = x.size(0)
        packed_x = x.view(-1, self.in_shape[0], self.in_shape[1], self.in_shape[2])
        self.feat = feat = self._forward_feature(packed_x, compute_linear=True)   # both conv layers and linear layer
        if self.multi_target:
            assert target is not None
            target = self.target_embed(target.view(-1, common.n_target_instructions))
            if self.use_target_gating:
                alpha = target
                for i, l in self.target_trans:
                    alpha = l(alpha)
                    if i + 1 < len(self.target_trans):
                        alpha = F.relu(alpha)
                self.feat = feat = feat * F.sigmoid(alpha)
        rnn_input = feat.view(batch, seq_len, self.feat_size)
        if self.multi_target:
            target = target.view(batch, seq_len, self.target_embed_dim)
            rnn_input = torch.cat([rnn_input, target], dim=-1)

        if isinstance(h, list): h = self._pack_hidden_states(h)

        rnn_output, final_h = self.cell(rnn_input, h)  # [seq_len, batch, units], [layer, batch, units]
        self.last_h = final_h
        if return_tensor:
            if isinstance(final_h, tuple):
                final_h = (final_h[0].data, final_h[1].data)
            else:
                final_h = final_h.data
        if unpack_hidden: final_h = self._unpack_hidden_states(final_h)

        rnn_feat = torch.cat([rnn_output.view(-1, self.rnn_output_size), self.feat], dim=1)

        # compute action
        if not only_value:
            feat = rnn_feat
            for i, l in enumerate(self.policy_layers):
                feat = l(feat)
                if i < len(self.policy_layers) - 1: feat = self.func(feat)
            self.logits = feat.view(batch, seq_len, self.out_dim)
            self.prob = F.softmax(feat).view(batch, seq_len, self.out_dim)
            self.logp = F.log_softmax(feat).view(batch, seq_len, self.out_dim)

            if sample_action:
                ret_act = torch.multinomial(self.prob.view(-1, self.out_dim), 1).view(batch, seq_len, 1)
            else:
                ret_act = self.logp

            if return_tensor: ret_act = ret_act.data
            if not return_value: return ret_act, final_h

        # compute value
        feat = rnn_feat
        for i, l in enumerate(self.critic_layers):
            feat = l(feat)
            if i < len(self.critic_layers) - 1: feat = self.func(feat)
        self.value = ret_val = feat.view(batch, seq_len)  # torch 0.2 required
        if return_tensor: ret_val = ret_val.data
        if only_value: return ret_val
        return ret_act, ret_val, final_h

    ########################
    def logprob(self, actions, logp=None):
        """
        :param actions: LongTensor, [batch, seq_len, 1]
        :param logp: None or [batch, seq_len, D_out]
        :return: log prob, [batch, seq_len]
        """
        if logp is None: logp = self.logp
        if len(actions.size()) == 2: actions = actions.unsqueeze(2)
        ret = torch.gather(logp, 2, actions)
        return ret.squeeze(dim=2)

    def entropy(self, logits=None):
        """
        logits: [batch, seq_len, D_out]
        return: [batch, seq_len]
        """
        if logits is None: logits = self.logits
        a0 = logits - logits.max(dim=2, keepdim=True)[0]
        ea0 = torch.exp(a0)
        z0 = ea0.sum(dim=2, keepdim=True)
        p0 = ea0 / z0
        ret = p0 * (torch.log(z0 + 1e-8) - a0)
        return ret.sum(dim=2)

    def kl_divergence(self, old_logP, new_logP):
        """
        :param old_logP: log probability [batch, seq_len, D_out]
        :param new_logP: [batch, seq_Len, D_out]
        :return: KL(new_P||old_P) [batch, seq_len]
        """
        kl = torch.exp(new_logP) * (new_logP - old_logP)
        return kl.sum(dim=-1)
