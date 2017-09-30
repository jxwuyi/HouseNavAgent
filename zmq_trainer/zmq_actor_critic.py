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
from zmq_trainer.zmqsimulator import SimulatorProcess, SimulatorMaster, ensure_proc_terminate


flag_max_lrate = 1e-3
flag_min_lrate = 1e-5
flag_max_kl_diff = 2e-2
flag_min_kl_diff = 1e-4
flag_lrate_coef = 1.5

class ZMQA3CTrainer(AgentTrainer):
    def __init__(self, name, model_creator, obs_shape, act_shape, args):
        super(ZMQA3CTrainer, self).__init__()
        self.name = name
        self.policy = model_creator()
        assert isinstance(self.policy, torch.nn.Module), \
            'ZMQ_A3C_Network must be an instantiated instance of torch.nn.Module'

        self.obs_shape = obs_shape
        self.act_shape = act_shape
        self.act_dim = sum(act_shape)
        # training args
        self.args = args
        self.gamma = args['gamma']
        self.lrate = args['lrate']
        self.batch_size = args['batch_size']
        if 't_max' not in args:
            args['t_max'] = 5
        self.t_max = args['t_max']
        if 'q_loss_coef' in args:
            self.q_loss_coef = args['q_loss_coef']
        else:
            self.q_loss_coef = 1.0
        if args['optimizer'] == 'adam':
            self.optim = optim.Adam(self.policy.parameters(), lr=self.lrate, weight_decay=args['weight_decay'])  #,betas=(0.5,0.999))
        else:
            self.optim = optim.RMSprop(self.policy.parameters(), lr=self.lrate, weight_decay=args['weight_decay'])
        self.grad_norm_clip = args['grad_clip']
        self._hidden = None

    def _create_gpu_tensor(self, frames, return_variable=True, volatile=False):
        # convert to tensor
        if isinstance(frames, np.ndarray): frames = [[torch.from_numpy(frames).type(ByteTensor)]]
        if not isinstance(frames, list): frames=[[frames]]
        """
        for i in range(len(frames)):
            if not isinstance(frames[i], list): frames[i] = [frames[i]]
            for j in range(len(frames[i])):
                if isinstance(frames[i][j], np.ndarray):
                    frames[i][j] = torch.from_numpy(frames[i][j]).type(ByteTensor)
        """
        tensor = [torch.stack(dat, dim=0) for dat in frames]
        gpu_tensor = torch.stack(tensor, dim=0).permute(0, 1, 4, 2, 3).type(FloatTensor)  # [batch, ....]
        if self.args['segment_input'] != 'index':
            if self.args['depth_input'] or ('attentive' in self.args['model_name']):
                gpu_tensor /= 256.0  # special hack here for depth info
            else:
                gpu_tensor = (gpu_tensor - 128.0) / 128.0
        if return_variable:
            gpu_tensor = Variable(gpu_tensor, volatile=volatile)
        return gpu_tensor

    def _create_gpu_hidden(self, tensor, return_variable=True, volatile=False):
        if not isinstance(tensor, list): tensor = [tensor]
        # convert to gpu tensor
        """
        for i in range(len(tensor)):
            if isinstance(tensor[i], tuple):
                if isinstance(tensor[i][0], np.ndarray):
                    tensor[i] = (torch.from_numpy(tensor[i][0]).type(FloatTensor),
                                 torch.from_numpy(tensor[i][1]).type(FloatTensor))
            else:
                if isinstance(tensor[i], np.ndarray):
                    tensor[i] = torch.from_numpy(tensor[i]).type(FloatTensor)
        """
        if isinstance(tensor[0], tuple):
            g = torch.cat([h[0] for h in tensor], dim=1)
            c = torch.cat([h[1] for h in tensor], dim=1)
            if return_variable:
                g = Variable(g, volatile=volatile)
                c = Variable(c, volatile=volatile)
            return (g, c)
        else:
            h = torch.cat(tensor, dim=1)
            if return_variable:
                h = Variable(h, volatile=volatile)
            return h

    def get_init_hidden(self):
        return self.policy.get_zero_state()

    def reset_agent(self):
        self._hidden = self.get_init_hidden()

    def action(self, obs, hidden=None, return_numpy=False):
        if hidden is None:
            hidden = self._hidden
            self._hidden = None
        assert (hidden is not None), '[ZMQA3CTrainer] Currently only support recurrent policy, please input last hidden state!'
        obs = self._create_gpu_tensor(obs, return_variable=True, volatile=True)  # [batch, 1, n, m, channel]
        hidden = self._create_gpu_hidden(hidden, return_variable=True, volatile=True)  # a list of hidden tensors
        act, nxt_hidden = self.policy(obs, hidden, return_value=False, sample_action=True,
                                   unpack_hidden=True, return_tensor=True)
        if self._hidden is None:
            self._hidden = nxt_hidden
        if return_numpy: # currently only for action
            act = act.cpu().numpy()
        return act, nxt_hidden   # NOTE: everything remains on gpu!

    def train(self):
        self.policy.train()

    def eval(self):
        self.policy.eval()

    def process_experience(self, idx, act, rew, done, terminal, info):
        pass

    def update(self, obs, init_hidden, act, rew, done, return_kl_divergence=True):
        """
        :param obs:  list of list of [dims]...
        :param init_hidden: list of [layer, 1, units]
        :param act: [batch, seq_len]
        :param rew: [batch, seq_len]
        :param done: [batch_seq_len]
        """
        tt = time.time()

        # reward clipping
        rew = np.clip(rew, -1, 1)

        # convert data to Variables
        obs = self._create_gpu_tensor(obs, return_variable=True)  # [batch, t_max+1, dims...]
        init_hidden = self._create_gpu_hidden(init_hidden, return_variable=True)  # [layers, batch, units]
        act = Variable(torch.from_numpy(act).type(LongTensor))  # [batch, t_max]
        mask = 1.0 - torch.from_numpy(done).type(FloatTensor) # [batch, t_max]
        mask_var = Variable(mask)

        time_counter[0] += time.time() - tt

        batch_size = self.batch_size
        t_max = self.t_max
        gamma = self.gamma

        tt = time.time()

        self.optim.zero_grad()

        # forward pass
        logits = []
        logprobs = []
        values = []
        obs = obs
        t_obs_slices = torch.chunk(obs, t_max + 1, dim=1)
        obs_slices = [t.contiguous() for t in t_obs_slices]
        cur_h = init_hidden
        for t in range(t_max):
            #cur_obs = obs[:, t:t+1, ...].contiguous()
            cur_obs = obs_slices[t]
            cur_logp, cur_val, nxt_h = self.policy(cur_obs, cur_h)
            cur_h = self.policy.mark_hidden_states(nxt_h, mask_var[:, t:t+1])
            values.append(cur_val)
            logprobs.append(cur_logp)
            logits.append(self.policy.logits)
        #cur_obs = obs[:, t_max:t_max + 1, ...].contiguous()
        cur_obs = obs_slices[-1]
        nxt_val = self.policy(cur_obs, cur_h, only_value=True, return_tensor=True)
        V = torch.cat(values, dim=1)  # [batch, t_max]
        P = torch.cat(logprobs, dim=1)  # [batch, t_max, n_act]
        L = torch.cat(logits, dim=1)
        p_ent = torch.mean(self.policy.entropy(L))  # compute entropy

        # estimate accumulative rewards
        rew = torch.from_numpy(rew).type(FloatTensor)  # [batch, t_max]
        R = []
        cur_R = nxt_val.squeeze()  # [batch]
        for t in range(t_max-1, -1, -1):
            cur_mask = mask[:, t]
            cur_R = rew[:, t] + gamma * cur_R * cur_mask
            R.append(cur_R)
        R.reverse()
        R = Variable(torch.stack(R, dim=1))  # [batch, t_max]

        # estimate advantage
        A = Variable(R.data - V.data)  # stop gradient here
        # [optional]  A = Variable(rew) - V

        # compute loss
        #critic_loss = F.smooth_l1_loss(V, R)
        critic_loss = torch.mean((R - V) ** 2)
        pg_loss = -torch.mean(self.policy.logprob(act, P) * A)
        if self.args['entropy_penalty'] is not None:
            pg_loss -= self.args['entropy_penalty'] * p_ent  # encourage exploration

        loss = self.q_loss_coef * critic_loss + pg_loss

        # backprop
        loss.backward()

        # grad clip
        if self.grad_norm_clip is not None:
            utils.clip_grad_norm(self.policy.parameters(), self.grad_norm_clip)
        self.optim.step()

        ret_dict = dict(pg_loss=pg_loss.data.cpu().numpy()[0],
                        policy_entropy=p_ent.data.cpu().numpy()[0],
                        critic_loss=critic_loss.data.cpu().numpy()[0])

        if return_kl_divergence:
            cur_h = init_hidden
            new_logprobs = []
            for t in range(t_max):
                # cur_obs = obs[:, t:t+1, ...].contiguous()
                cur_obs = obs_slices[t]
                cur_logp, nxt_h = self.policy(cur_obs, cur_h, return_value=False)
                cur_h = self.policy.mark_hidden_states(nxt_h, mask_var[:, t:t + 1])
                new_logprobs.append(cur_logp)
            new_P = torch.cat(new_logprobs, dim=1)
            kl = self.policy.kl_divergence(new_P, P).mean().data.cpu()[0]
            ret_dict['KL(P_new||P_old)'] = kl

            if kl > flag_max_kl_diff:
                self.lrate /= flag_lrate_coef
                self.optim.__dict__['param_groups'][0]['lr']=self.lrate
                ret_dict['!!![NOTE]:'] = ('------>>>> KL is too large (%.6f), decrease lrate to %.5f' % (kl, self.lrate))
            elif (kl < flag_min_kl_diff) and (self.lrate < flag_max_lrate):
                self.lrate *= flag_lrate_coef
                self.optim.__dict__['param_groups'][0]['lr'] = self.lrate
                ret_dict['!!![NOTE]:'] = ('------>>>> KL is too small (%.6f), increase lrate to %.5f' % (kl, self.lrate))


        time_counter[1] += time.time() - tt
        return ret_dict

    def is_rnn(self):
        return True
