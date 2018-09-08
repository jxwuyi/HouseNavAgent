from headers import *
import numpy as np
import common
import zmq_trainer.zmq_util
import random
import utils
import time
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable


class SemanticTrainer(AgentTrainer):
    def __init__(self, policy_creator, obs_shape, n_class, args):
        super(SemanticTrainer, self).__init__()
        self.name = 'SUP'
        self.policy = policy_creator()
        self.multi_label = self.policy.multi_label
        assert isinstance(self.policy, torch.nn.Module), \
            'SUPTrainer.policy must be an instantiated instance of torch.nn.Module'

        self.obs_shape = obs_shape
        self.out_dim = n_class
        # training args
        self.args = args
        self.lrate = args['lrate'] if 'lrate' in args else 0.001
        self.batch_size = args['batch_size'] if 'batch_size' in args else 64
        self.grad_batch = args['grad_batch'] if 'grad_batch' in args else 1
        self.accu_grad_steps = 0
        self.accu_ret_dict = dict()
        if ('logits_penalty' in args) and (args['logits_penalty'] is not None):
            self.logit_loss_coef = args['logits_penalty']
            print("[Trainer] Using Logits Loss Coef = %.4f" % self.logit_loss_coef)
        else:
            self.logit_loss_coef = None
        if 'optimizer' not in args:
            self.optim = None
        elif args['optimizer'] == 'adam':
            self.optim = optim.Adam(self.policy.parameters(), lr=self.lrate, weight_decay=args['weight_decay'])  #,betas=(0.5,0.999))
        else:
            self.optim = optim.RMSprop(self.policy.parameters(), lr=self.lrate, weight_decay=args['weight_decay'])
        self.grad_norm_clip = args['grad_clip'] if 'grad_clip' in args else None

    def _create_feature_tensor(self, feature, return_variable=True, volatile=False):
        # feature: [batch, t_max, feature_dim]
        ret = torch.from_numpy(feature).type(ByteTensor).type(FloatTensor)
        if return_variable:
            ret = Variable(ret, volatile=volatile)
        return ret

    def _create_target_tensor(self, targets, seq_len, return_variable=True, volatile=False):
        # targets: [batch]
        # return: [batch, seq_len, n_instructions]
        batch = len(targets)
        target_n = torch.zeros(batch, 1, self.policy.n_target_instructions).type(FloatTensor)
        ids = torch.from_numpy(np.array(targets)).type(LongTensor).view(batch, 1, 1)
        target_n.scatter_(2, ids, 1.0)
        target_n = target_n.repeat(1, seq_len, 1)
        if return_variable:
            target_n = Variable(target_n, volatile=volatile)
        return target_n

    def _create_gpu_tensor(self, frames, return_variable=True, volatile=False):
        # convert to tensor
        gpu_tensor = torch.from_numpy(frames).type(ByteTensor).permute(0,3,1,2).type(FloatTensor)
        if self.args['segment_input'] != 'index':
            if self.args['depth_input'] or ('attentive' in self.args['model_name']):
                gpu_tensor /= 256.0  # special hack here for depth info
            else:
                gpu_tensor = (gpu_tensor - 128.0) / 128.0
        if return_variable:
            gpu_tensor = Variable(gpu_tensor, volatile=volatile)
        return gpu_tensor

    def action(self, obs, return_numpy=False, greedy_act=False, return_argmax=False):
        # Assume all input data are numpy arrays!
        # obs: [batch, n, m, channel], uint8
        # return: [batch, n_class], probability over classes
        batch_size = obs.shape[0]
        obs = self._create_gpu_tensor(obs, return_variable=True, volatile=True)  # [batch, t_max, n, m, channel]
        prob = self.policy(obs).data   # tensor
        if greedy_act:
            if self.multi_label:  # sigmoid
                prob = (prob > 0.5).type(ByteTensor)
            else:  # softmax
                if return_argmax:
                    prob = torch.max(prob, dim=-1, keepdim=False)[1]
                else:
                    max_val, _ = torch.max(prob, dim=-1, keepdim=True).repeat(1, self.out_dim)
                    prob = (prob == max_val).type(ByteTensor)
        if return_numpy:
            return prob.cpu().numpy()
        else:
            return prob  # probability, [batch, n_class]

    def train(self):
        self.policy.train()

    def eval(self):
        self.policy.eval()

    def update(self, obs, label):
        """
        all input params are numpy arrays
        :param obs: [batch, seq_len, n, m, channel]
        :param label: [batch, n_class] (sigmoid) or [batch] (softmax)
        """
        tt = time.time()
        # convert data to Variables
        batch_size = obs.shape[0]
        obs = self._create_gpu_tensor(obs, return_variable=True)  # [batch, channel, n, m]

        # create label tensor
        if self.multi_label:
            t_label = torch.from_numpy(np.array(label)).type(FloatTensor)
        else:
            t_label = torch.from_numpy(np.array(label)).type(LongTensor)
        label = Variable(t_label)

        time_counter[0] += time.time() - tt

        tt = time.time()

        if self.accu_grad_steps == 0:  # clear grad
            self.optim.zero_grad()

        # forward pass
        # logits: [batch, n_class]
        logits = self.policy(obs, return_logits=True)

        # compute loss
        if self.multi_label:
            loss = torch.mean(F.binary_cross_entropy_with_logits(logits, label))
        else:
            loss = torch.mean(F.cross_entropy(logits, label))

        # entropy penalty
        L_ent = torch.mean(self.policy.entropy(logits=logits))
        if self.args['entropy_penalty'] is not None:
            loss -= self.args['entropy_penalty'] * L_ent

        # L^2 penalty
        L_norm = torch.mean(torch.sum(logits * logits, dim=-1))
        if self.args['logits_penalty'] is not None:
            loss += self.args['logits_penalty'] * L_norm

        # compute accuracy
        if self.multi_label:
            max_idx = (logits.data > 0.5).type(FloatTensor)
            total_sample = batch_size * self.out_dim
        else:
            _, max_idx = torch.max(logits.data, dim=-1, keepdim=False)
            total_sample = batch_size
        L_accu = torch.sum((max_idx == t_label).type(FloatTensor)) / batch_size

        ret_dict = dict(loss=loss.data.cpu().numpy()[0],
                        entropy=L_ent.data.cpu().numpy()[0],
                        logits_norm=L_norm.data.cpu().numpy()[0],
                        accuracy=L_accu)

        # backprop
        if self.grad_batch > 1:
            loss = loss / float(self.grad_batch)
        loss.backward()

        # accumulative stats
        if self.accu_grad_steps == 0:
            self.accu_ret_dict = ret_dict
        else:
            for k in ret_dict:
                self.accu_ret_dict[k] += ret_dict[k]

        self.accu_grad_steps += 1
        if self.accu_grad_steps < self.grad_batch:  # do not update parameter now
            time_counter[1] += time.time() - tt
            return None

        # update stats
        for k in self.accu_ret_dict:
            self.accu_ret_dict[k] /= self.grad_batch
        ret_dict = self.accu_ret_dict
        self.accu_grad_steps = 0

        # grad clip
        if self.grad_norm_clip is not None:
            utils.clip_grad_norm(self.policy.parameters(), self.grad_norm_clip)
        self.optim.step()

        time_counter[1] += time.time() - tt
        return ret_dict

    def is_rnn(self):
        return False
