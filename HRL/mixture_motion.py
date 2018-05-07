from headers import *
import common
import utils

import sys, os, platform

import numpy as np
import random

from House3D.house import ALLOWED_PREDICTION_ROOM_TYPES, ALLOWED_OBJECT_TARGET_TYPES

from HRL.rnn_motion import RNNMotion

all_allowed_targets = ALLOWED_PREDICTION_ROOM_TYPES + ALLOWED_OBJECT_TARGET_TYPES

"""
trainer: a dictionary from <target> -> trainer
"""
class MixMotion(RNNMotion):
    def __init__(self, task, trainer=None, pass_target=None, term_measure='mask'):
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

        self.pass_target_dict = pass_target
        self.trainer_dict = trainer
        super(MixMotion, self).__init__(task, None, True, term_measure)

    def reset(self):
        for trainer in self.trainer_dict.keys():
            trainer.reset_agent()

    """
    return a list of [aux_mask, action, reward, done, info]
    """
    def run(self, target, max_steps):
        self.trainer = self.trainer_dict[target]
        self.pass_target = self.pass_target_dict[target]
        return super(MixMotion, self).run(target, max_steps)
