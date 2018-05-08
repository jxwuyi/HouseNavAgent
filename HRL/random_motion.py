from headers import *
import common

import numpy as np

from House3D.roomnav import allowed_actions_for_supervision, discrete_actions, discrete_angle_delta_value

allowed_action_index = allowed_actions_for_supervision
n_allowed_actions = len(allowed_action_index)


class RandomMotion(BaseMotion):
    def __init__(self, task, trainer=None, pass_target=True, term_measure='mask'):
        assert trainer is None
        assert term_measure != 'stay'
        super(RandomMotion, self).__init__(task, trainer, pass_target, term_measure)

    """
    return a list of [aux_mask, action, reward, done, info]
    """
    def run(self, target, max_steps):
        task = self.task
        target_id = common.target_instruction_dict[target]
        final_target = task.get_current_target()
        final_target_id = common.target_instruction_dict[final_target]
        env = self.env
        ret = []
        for _ in range(max_steps):
            act = allowed_action_index[np.random.randint(n_allowed_actions)]
            det_fwd, det_hor, det_rot = discrete_actions[act]
            move_fwd = det_fwd * task.move_sensitivity
            move_hor = det_hor * task.move_sensitivity
            rotation = det_rot * task.rot_sensitivity
            env.rotate(rotation)
            if self.env.move_forward(move_fwd, move_hor):
                if task.discrete_angle is not None:
                    task._yaw_ind = (task._yaw_ind + discrete_angle_delta_value[act] + task.discrete_angle) % task.discrete_angle
            mask = task.get_feature_mask()
            done = self._is_success(final_target_id, mask, term_measure='see')
            ret.append((mask, act, (10 if done else 0), done, task.info))
            if done or self._is_success(target_id, mask, term_measure=self.term_measure):
                break
        return ret
