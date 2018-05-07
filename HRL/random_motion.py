from headers import *
import common

import numpy as np

from House3D.roomnav import allowed_actions_for_supervision, discrete_actions, discrete_angle_delta_value

allowed_action_index = allowed_actions_for_supervision
n_allowed_actions = len(allowed_action_index)


class RandomMotion(BaseMotion):
    def __init__(self, task, trainer=None, pass_target=True):
        assert trainer is None
        super(RandomMotion, self).__init__(task, trainer, pass_target)
        # fetch target mask graph
        self.env = task.env

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
            done = (mask[final_target_id] > 0)   # TODO: task._is_success()
            ret.append((mask, act, (10 if done else 0), done, task.info))
            if mask[target_id] > 0:
                break
        return ret
