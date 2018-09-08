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


class SUPTrainer(AgentTrainer):
    def __init__(self, policy_creator, obs_shape, act_shape, args):
        super(SUPTrainer, self).__init__()
        self.name = 'SUP'
        self.policy = policy_creator()
        assert isinstance(self.policy, torch.nn.Module), \
            'SUPTrainer.policy must be an instantiated instance of torch.nn.Module'

        self.net = self.policy
        self._is_multigpu = False
        if isinstance(args['train_gpu'], list) and len(args['train_gpu']) > 1:
            # Multi-GPU
            self.net = torch.nn.DataParallel(self.policy, device_ids=args['train_gpu'])
            self._is_multigpu = True

        self.obs_shape = obs_shape
        self.act_shape = act_shape
        self.act_dim = sum(act_shape)
        # training args
        self.args = args
        self.multi_target = args['multi_target']
        self.mask_feature = args['mask_feature'] if 'mask_feature' in args else False
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
        gpu_tensor = torch.from_numpy(frames).type(ByteTensor).permute(0,1,4,2,3).type(FloatTensor)
        if self.args['segment_input'] != 'index':
            if self.args['depth_input'] or ('attentive' in self.args['model_name']):
                gpu_tensor /= 256.0  # special hack here for depth info
            else:
                gpu_tensor = (gpu_tensor - 128.0) / 128.0
        if return_variable:
            gpu_tensor = Variable(gpu_tensor, volatile=volatile)
        return gpu_tensor

    def action(self, obs, target=None, hidden=None, return_numpy=False, temperature=None, mask_input=None, greedy_act=False):
        # Assume all input data are numpy arrays!
        # obs: [batch, t_max, n, m, channel], uint8
        # target: [batch], int32
        # mask_input: [batch, t_max, mask_feat]
        batch_size = obs.shape[0]
        t_max = obs.shape[1]
        if hidden is None:
            hidden = self.policy.get_zero_state(batch=batch_size, return_variable=True, volatile=True,
                                                hidden_batch_first=self._is_multigpu)
        obs = self._create_gpu_tensor(obs, return_variable=True, volatile=True)  # [batch, t_max, channel, n, m]
        if target is not None:
            target = self._create_target_tensor(target, t_max, return_variable=True, volatile=True)
        if mask_input is not None:
            mask_input = self._create_feature_tensor(mask_input, return_variable=True, volatile=True)
        act, _ = self.net(obs, hidden, return_value=False, sample_action=False,
                          unpack_hidden=True, return_tensor=True, target=target,
                          temperature=temperature, extra_input_feature=mask_input)
        if greedy_act:
            _, act_idx = torch.max(act, dim=-1, keepdim=False)
            act = act_idx   # [batch, seq_len\]
        if return_numpy:
            act = act.cpu().numpy()
        return act  # log(prob), [batch, seq_len, n_act]

    def train(self):
        self.policy.train()

    def eval(self):
        self.policy.eval()

    def update(self, obs, act, length_mask, target=None, mask_input=None,
               hidden=None):
        """
        all input params are numpy arrays
        :param obs: [batch, seq_len, n, m, channel]
        :param act: [batch, seq_len]
        :param length_mask: [batch, seq_len]
        :param target: [batch] or None (when single-target)
        :param mask_input: (optional) [batch, seq_len, feat_dim]
        """
        tt = time.time()
        # convert data to Variables
        batch_size = obs.shape[0]
        seq_len = obs.shape[1]
        total_samples = float(np.sum(length_mask))
        obs = self._create_gpu_tensor(obs, return_variable=True)  # [batch, t_max, dims...]
        if hidden is None:
            hidden = self.policy.get_zero_state(batch=batch_size, return_variable=True, hidden_batch_first=self._is_multigpu)
        if target is not None:
            target = self._create_target_tensor(target, seq_len, return_variable=True)
        if mask_input is not None:
            mask_input = self._create_feature_tensor(mask_input, return_variable=True)
        length_mask = self._create_feature_tensor(length_mask, return_variable=True)  #[batch, t_max]

        # create action tensor
        #act = Variable(torch.from_numpy(act).type(LongTensor))  # [batch, t_max]
        act_n = torch.zeros(batch_size, seq_len, self.policy.out_dim).type(FloatTensor)
        ids = torch.from_numpy(np.array(act)).type(LongTensor).view(batch_size, seq_len, 1)
        act_n.scatter_(2, ids, 1.0)
        act_n = Variable(act_n)

        time_counter[0] += time.time() - tt

        tt = time.time()

        if self.accu_grad_steps == 0:  # clear grad
            self.optim.zero_grad()

        # forward pass
        # logits: [batch, seq_len, n_act]
        logits, _ = self.net(obs, hidden, return_value=False, sample_action=False,
                             return_tensor=False, target=target,
                             extra_input_feature=mask_input, return_logits=True, hidden_batch_first=self._is_multigpu)

        # compute loss
        #critic_loss = F.smooth_l1_loss(V, R)
        block_size = batch_size * seq_len
        act_size = logits.size(-1)
        flat_logits = logits.view(block_size, act_size)
        logp = torch.sum(F.log_softmax(flat_logits).view(batch_size, seq_len, act_size) * act_n, dim=-1) * length_mask
        loss = -torch.sum(logp) / total_samples

        # entropy penalty
        L_ent = torch.sum(self.policy.entropy(logits=logits) * length_mask) / total_samples
        if self.args['entropy_penalty'] is not None:
            loss -= self.args['entropy_penalty'] * L_ent

        # L^2 penalty
        L_norm = torch.sum(torch.sum(logits * logits, dim=-1) * length_mask) / total_samples
        if self.args['logits_penalty'] is not None:
            loss += self.args['logits_penalty'] * L_norm

        # compute accuracy
        _, max_idx = torch.max(logits.data, dim=-1, keepdim=True)
        L_accu = torch.sum((max_idx == ids).type(FloatTensor) * length_mask.data.view(batch_size, seq_len, 1)) / total_samples

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
        return True
