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
import zmq_trainer.zmq_actor_critic as ZMQA3C
from zmq_trainer.zmq_actor_critic import ZMQA3CTrainer


flag_max_lrate = ZMQA3C.flag_max_lrate
flag_min_lrate = ZMQA3C.flag_min_lrate
flag_max_kl_diff = ZMQA3C.flag_max_kl_diff
flag_min_kl_diff = ZMQA3C.flag_min_kl_diff
flag_lrate_coef = ZMQA3C.flag_lrate_coef

# when supervised loss, weight for <indoor> is 0.1
# when reinforce loss, reward for correctly predict <indoor> is 0.1
aux_uncertain_weight = 0.1
aux_uncertain_id = common.all_aux_predictions['indoor']

# cache aux target mask
aux_max_allowed_mask_value = (1 << common.n_aux_predictions)
aux_mask_dict = []
for msk in range(aux_max_allowed_mask_value):
    cur = []
    for i in range(common.n_aux_predictions):
        if (msk and (1 << i)) > 0:
            cur.append(i)
    aux_mask_dict.append(cur)

class ZMQAuxTaskTrainer(ZMQA3CTrainer):
    def __init__(self, name, model_creator, obs_shape, act_shape, args):
        super(ZMQAuxTaskTrainer, self).__init__(name, model_creator, obs_shape, act_shape, args)
        self.use_supervised_loss = not args['reinforce_loss']
        self.aux_loss_coef = args['aux_loss_coef']

    def _create_aux_target_tensor(self, targets):
        aux_tar = torch.from_numpy(np.array(targets)).type(FloatTensor)
        if self.use_supervised_loss:
            aux_tar[:, :, aux_uncertain_id] *= aux_uncertain_weight
        else:
            aux_tar[:, :, aux_uncertain_id] *= 0.5 * (aux_uncertain_weight + 1)
            aux_tar = aux_tar * 2 - 1
        aux_tar = Variable(aux_tar)
        return aux_tar

    def process_aux_target(self, mask):
        ret = np.zeros(common.n_aux_predictions, dtype=np.float32)
        ret[aux_mask_dict[mask]] = 1
        return ret

    def get_aux_task_reward(self, pred, mask):
        if (mask and (1 << pred)) > 0:
            if pred == aux_uncertain_id:
                return aux_uncertain_weight
            else:
                return 1.0
        else:
            return -1.0

    def action(self, obs, hidden=None, return_numpy=False, target=None, return_aux_pred=False):
        if hidden is None:
            hidden = self._hidden
            self._hidden = None
        assert (hidden is not None), '[ZMQA3CTrainer] Currently only support recurrent policy, please input last hidden state!'
        obs = self._create_gpu_tensor(obs, return_variable=True, volatile=True)  # [batch, 1, n, m, channel]
        hidden = self._create_gpu_hidden(hidden, return_variable=True, volatile=True)  # a list of hidden tensors
        if target is not None:
            target = self._create_target_tensor(target, return_variable=True, volatile=True)
        ret_vals = self.policy(obs, hidden, return_value=False, sample_action=True,
                               unpack_hidden=True, return_tensor=True, target=target,
                               compute_aux_pred=return_aux_pred, sample_aux_pred=True)

        act, nxt_hidden = ret_vals[0], ret_vals[1]
        if self._hidden is None:
            self._hidden = nxt_hidden
        if return_numpy: # currently only for action
            act = act.cpu().numpy()
        if return_aux_pred:
            aux_pred = ret_vals[2]
            if return_numpy: aux_pred = aux_pred.cpu().numpy()
            return act, nxt_hidden, aux_pred
        else:
            return act, nxt_hidden   # NOTE: everything remains on gpu!

    def update(self, obs, init_hidden, act, rew, done, target=None, aux_target=None, return_kl_divergence=True):
        """
        :param obs:  list of list of [dims]...
        :param init_hidden: list of [layer, 1, units]
        :param act: [batch, seq_len]
        :param rew: [batch, seq_len]
        :param done: [batch, seq_len]
        :param target: [batch, seq_len, n_instruction] or None (when single-target)
        :param aux_target: 0/1 label matrix [batch, seq_len, n_aux_pred] or None (not updating the aux-loss)
        """
        assert(aux_target is not None), 'AuxTrainer must be given <aux_target>'
        tt = time.time()

        # reward clipping
        rew = np.clip(rew, -1, 1)

        # convert data to Variables
        obs = self._create_gpu_tensor(obs, return_variable=True)  # [batch, t_max+1, dims...]
        init_hidden = self._create_gpu_hidden(init_hidden, return_variable=True)  # [layers, batch, units]
        if target is not None:
            target = self._create_target_tensor(target, return_variable=True)
        aux_target = self._create_aux_target_tensor(aux_target)
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
        aux_preds = []
        obs = obs
        t_obs_slices = torch.chunk(obs, t_max + 1, dim=1)
        obs_slices = [t.contiguous() for t in t_obs_slices]
        if target is not None:
            t_target_slices = torch.chunk(target, t_max + 1, dim=1)
            target_slices = [t.contiguous() for t in t_target_slices]
        cur_h = init_hidden
        for t in range(t_max):
            #cur_obs = obs[:, t:t+1, ...].contiguous()
            cur_obs = obs_slices[t]
            if target is not None:
                ret_vals = self.policy(cur_obs, cur_h, target=target_slices[t],
                                       compute_aux_pred=True, return_aux_logprob=self.use_supervised_loss)
            else:
                ret_vals = self.policy(cur_obs, cur_h,
                                       compute_aux_pred=True, return_aux_logprob=self.use_supervised_loss)
            cur_logp, cur_val, nxt_h, aux_p = ret_vals
            cur_h = self.policy.mark_hidden_states(nxt_h, mask_var[:, t:t+1])
            values.append(cur_val)
            logprobs.append(cur_logp)
            logits.append(self.policy.logits)
            aux_preds.append(aux_p)
        #cur_obs = obs[:, t_max:t_max + 1, ...].contiguous()
        cur_obs = obs_slices[-1]
        if target is not None:
            nxt_val = self.policy(cur_obs, cur_h, only_value=True, return_tensor=True, target=target_slices[-1])
        else:
            nxt_val = self.policy(cur_obs, cur_h, only_value=True, return_tensor=True)
        V = torch.cat(values, dim=1)  # [batch, t_max]
        P = torch.cat(logprobs, dim=1)  # [batch, t_max, n_act]
        L = torch.cat(logits, dim=1)
        p_ent = torch.mean(self.policy.entropy(L))  # compute entropy
        Aux_P = torch.cat(aux_preds, dim=1)  # [batch, t_max, n_aux_pred]

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

        # aux task loss
        aux_loss = -(Aux_P * aux_target).sum(dim=-1).mean()

        loss = self.q_loss_coef * critic_loss + pg_loss + self.aux_loss_coef * aux_loss

        # backprop
        loss.backward()

        # grad clip
        if self.grad_norm_clip is not None:
            utils.clip_grad_norm(self.policy.parameters(), self.grad_norm_clip)
        self.optim.step()

        ret_dict = dict(pg_loss=pg_loss.data.cpu().numpy()[0],
                        aux_task_loss=aux_loss.data.cpu().numpy()[0],
                        policy_entropy=p_ent.data.cpu().numpy()[0],
                        critic_loss=critic_loss.data.cpu().numpy()[0])

        if return_kl_divergence:
            cur_h = init_hidden
            new_logprobs = []
            for t in range(t_max):
                # cur_obs = obs[:, t:t+1, ...].contiguous()
                cur_obs = obs_slices[t]
                if self.multi_target:
                    cur_target = target_slices[t]
                else:
                    cur_target = None
                cur_logp, nxt_h = self.policy(cur_obs, cur_h, return_value=False, target=cur_target)
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
