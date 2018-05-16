from headers import *
import common
import utils

import sys, os, platform

import numpy as np
import random

from House3D.house import ALLOWED_TARGET_ROOM_TYPES, ALLOWED_OBJECT_TARGET_TYPES

from HRL.rnn_motion import RNNMotion

all_allowed_targets = ALLOWED_TARGET_ROOM_TYPES + ALLOWED_OBJECT_TARGET_TYPES


"""
arg_dict: map from <target> to <model_args>
"""
def create_mixture_motion_trainer_dict(arg_dict):
    import zmq_train
    trainer_dict = dict()
    pass_tar_dict = dict()
    obs_mode_dict = dict()  # segment_input, depth_signal=True, target_mask_signal=False, joint_visual_signal=False
    loaded_model = dict()
    for target in all_allowed_targets:
        assert target in arg_dict, '[MixtureMotion] Invalid <arg_dict>! Key=<{}> does not exist!'.format(target)
        args = arg_dict[target]
        model_file = args['warmstart']
        assert (model_file is not None) and os.path.exists(model_file), \
            '[MixtureMotion] model file <{}> for target <{}> does not exist!!'.format(args[model_file], target)
        if model_file in loaded_model:
            trainer_dict[target] = trainer_dict[loaded_model[model_file]]
            pass_tar_dict[target] = pass_tar_dict[loaded_model[model_file]]
            obs_mode_dict[target] = obs_mode_dict[loaded_model[model_file]]
            continue
        common.process_observation_shape('rnn', args['resolution_level'],
                                         segmentation_input=args['segment_input'],
                                         depth_input=args['depth_input'],
                                         history_frame_len=1,
                                         target_mask_input=args['target_mask_input'])
        # ensure object target
        __backup_CFG = common.CFG.copy()
        common.ensure_object_targets(args['object_target'])
        trainer = zmq_train.create_zmq_trainer('a3c', 'rnn', args)
        common.CFG = __backup_CFG  # backup
        # load model
        trainer.load(model_file)
        trainer.eval()
        loaded_model[model_file] = target
        trainer_dict[target] = trainer
        pass_tar_dict[target] = args['multi_target']
        obs_mode_dict[target] = dict(segment_input=(args['segment_input'] != 'none'),
                                     depth_signal=args['depth_input'],
                                     target_mask_signal=args['target_mask_input'],
                                     joint_visual_signal=(args['segment_input'] == 'joint'))
    return trainer_dict, pass_tar_dict, obs_mode_dict


"""
trainer: a dictionary from <target> -> trainer
"""
class MixMotion(RNNMotion):
    def __init__(self, task, trainer=None, pass_target=None, term_measure='mask', obs_mode=None):
        assert isinstance(trainer, dict), '[MixMotion] trainer must be a dict!'
        assert isinstance(pass_target, bool) or isinstance(pass_target, dict), '[MixMotion] pass_target must be a dict or a boolean!'
        assert sorted(trainer.keys()) == sorted(all_allowed_targets), '[MixMotion] keys of trainer must contain all the targets!'
        if isinstance(pass_target, dict):
            assert sorted(pass_target.keys()) == sorted(all_allowed_targets), '[MixMotion] keys of pass_target must contain all the targets!'
        else:
            flag = pass_target
            pass_target = dict()
            for k in trainer.keys():
                pass_target[k] = flag
        self.obs_mode_dict = None
        if obs_mode is not None:
            if isinstance(obs_mode, dict):
                assert sorted(obs_mode.keys()) == sorted(all_allowed_targets), '[MixMotion] keys of obs_mode_dict must contain all the targets!'
                self.obs_mode_dict = obs_mode
            else:
                if self.task.get_obs_mode() != obs_mode:
                    self.task.reset_obs_mode(**obs_mode)

        self.pass_target_dict = pass_target
        self.trainer_dict = trainer
        super(MixMotion, self).__init__(task, None, True, term_measure)

    def reset(self):
        for trainer in self.trainer_dict.values():
            trainer.reset_agent()
        self._interrupt=None

    """
    return a list of [aux_mask, action, reward, done, info]
    """
    def run(self, target, max_steps, temperature=None):
        self.trainer = self.trainer_dict[target]
        if hasattr(self.trainer.policy, 'extra_feature_dim'):
            self._use_mask_feat_dim = self.trainer.policy.extra_feature_dim
        self.pass_target = self.pass_target_dict[target]
        if self.obs_mode_dict is not None:
            obs_mode = self.obs_mode_dict[target]
            #print('target = {}, obs_mode = {}'.format(target, obs_mode))
            #print('  -> task obs_mode = {}'.format(self.task.get_obs_mode()))
            if obs_mode != self.task.get_obs_mode():
                self.task.reset_obs_mode(**obs_mode)
        return super(MixMotion, self).run(target, max_steps, temperature)
