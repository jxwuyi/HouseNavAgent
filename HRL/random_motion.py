from headers import *
import common

import numpy as np

from House3D.roomnav import allowed_actions_for_supervision, discrete_actions, discrete_angle_delta_value

allowed_action_index = allowed_actions_for_supervision
n_allowed_actions = len(allowed_action_index)


class RandomMotion(BaseMotion):
    def __init__(self, task, trainer=None, pass_target=True, term_measure='mask', oracle_func=None):
        assert trainer is None
        assert term_measure != 'stay'
        super(RandomMotion, self).__init__(task, trainer, pass_target, term_measure, oracle_func)
        self.skilled_rate = None

    def set_skilled_rate(self, rate):
        self.skilled_rate = rate

    """
    return a list of [aux_mask, action, reward, done, info]
    """
    def run(self, target, max_steps):
        skilled_steps = (self.skilled_rate or 1) * max_steps
        flag_term = False
        task = self.task
        target_id = common.target_instruction_dict[target]
        final_target = task.get_current_target()
        final_target_id = common.target_instruction_dict[final_target]
        env = self.env
        ret = []
        restore_state = None
        accu_success_steps = 0
        for _s in range(skilled_steps):
            act = allowed_action_index[np.random.randint(n_allowed_actions)]
            det_fwd, det_hor, det_rot = discrete_actions[act]
            move_fwd = det_fwd * task.move_sensitivity
            move_hor = det_hor * task.move_sensitivity
            rotation = det_rot * task.rot_sensitivity
            env.rotate(rotation)
            if self.env.move_forward(move_fwd, move_hor):
                if task.discrete_angle is not None:
                    task._yaw_ind = (task._yaw_ind + discrete_angle_delta_value[act] + task.discrete_angle) % task.discrete_angle
            if (_s == max_steps - 1) and (max_steps < skilled_steps):
                restore_state = task.info
            mask = task.get_feature_mask()  # if self._oracle_func is None else self._oracle_func.get(task), NOTE: we do not need oracle actually.. this is random policy
            flag = self._is_success(final_target_id, mask, term_measure='see')
            if flag:
                accu_success_steps += 1
            else:
                accu_success_steps = 0
            if accu_success_steps == 3:
                done = True
            else:
                done = False
            rew = (10 if done else 0)
            ret.append((mask, act, rew, done, task.info))
            if (done and (_s < max_steps)) or ((target != final_target) and self._is_success(target_id, mask, term_measure=self.term_measure)):
                flag_term = True
                break
        if flag_term:
            return ret[-max_steps:]
        else:
            if max_steps < skilled_steps:
                task.set_state(restore_state)
            return ret[:max_steps]
