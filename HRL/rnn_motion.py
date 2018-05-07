from headers import *
import common
import utils

import sys, os, platform

import numpy as np
import random

import House3D
from House3D.roomnav import n_discrete_actions
from House3D import Environment as HouseEnv
from House3D import MultiHouseEnv
from House3D import House
from House3D.house import ALLOWED_PREDICTION_ROOM_TYPES, ALLOWED_OBJECT_TARGET_INDEX, ALLOWED_TARGET_ROOM_TYPES, ALLOWED_OBJECT_TARGET_TYPES
from House3D.roomnav import RoomNavTask
from House3D.objnav import ObjNavTask


class RNNMotion(BaseMotion):
    def __init__(self, task, trainer=None, pass_target=True):
        super(RNNMotion, self).__init__(task, trainer, pass_target)

    def reset(self):
        self.trainer.reset_agent()

    """
    return a list of [aux_mask, action, reward, done, info]
    """
    def run(self, target, max_steps):
        task = self.task
        trainer = self.trainer
        target_id = common.target_instruction_dict[target]
        #trainer.set_target(target)
        consistent_target = (target == self.task.get_current_target())

        episode_stats = []
        obs = task._cached_obs
        for _st in range(max_steps):
            # get action
            action, _ = trainer.action(obs, return_numpy=True, target=[[target_id]] if self.pass_target else None)
            action = int(action.squeeze())
            # environment step
            _, rew, done, info = task.step(action)
            feature_mask = task.get_feature_mask()
            episode_stats.append((feature_mask, action, rew, done, info))
            # check terminate
            if done or (not consistent_target and (feature_mask[target_id] > 0)):
                break
        return episode_stats
