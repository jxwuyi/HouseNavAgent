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
    def __init__(self, task, trainer=None):
        assert isinstance(trainer, dict), '[MixMotion] trainer must be a dict!'
        assert sorted(trainer.keys()) == sorted(all_allowed_targets), '[MixMotion] keys of trainer must contain all the targets!'

        self.trainer_dict = trainer
        super(MixMotion, self).__init__(task, None)

    def reset(self):
        for trainer in self.trainer_dict.keys():
            trainer.reset_agent()

    """
    return a list of [aux_mask, action, reward, done, info]
    """
    def run(self, target, max_steps):
        self.trainer = self.trainer_dict[target]
        return super(MixMotion, self).run(target, max_steps)
