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

n_rooms = len(ALLOWED_TARGET_ROOM_TYPES)

class RNNMotion(BaseMotion):
    def __init__(self, task, trainer=None, pass_target=True, term_measure='mask', oracle_func=None):
        super(RNNMotion, self).__init__(task, trainer, pass_target, term_measure, oracle_func)
        self._interrupt = None
        self._use_mask_feat_dim = (trainer is not None) and \
                hasattr(trainer.policy, 'extra_feature_dim') and trainer.policy.extra_feature_dim
        self._cached_target_id = None

    def reset(self):
        self.trainer.reset_agent()
        self._interrupt = None

    def is_interrupt(self):
        if self._interrupt is None:
            final_target = self.task.get_current_target()
            final_target_id = common.target_instruction_dict[final_target]
            if (final_target_id > n_rooms) and (final_target_id != self._cached_target_id):
                self._interrupt = self._is_insight(obs_seg=self.task._fetch_cached_segmentation(),
                                                   n_pixel=100)  # see 100 pixels
            else:
                self._interrupt = False
        return self._interrupt

    def check_terminate(self, target_id, mask, act):
        if self.term_measure == 'interrupt':
            self._interrupt = None
            self._cached_target_id = target_id
            if self.is_interrupt():
                return True
        if self.term_measure == 'see':
            return self._is_success(target_id, mask, self.term_measure,
                                    obs_seg=self.task._fetch_cached_segmentation(),
                                    target_name=common.all_target_instructions[target_id])
        if self.term_measure == 'stay':
            return self._is_success(target_id, mask, self.term_measure,
                                    is_stay=(act==n_discrete_actions-1))
        return mask[target_id] > 0   # term_measure == 'mask'

    def _get_feature_mask(self):
        feat = self.task.get_feature_mask() if self._oracle_func is None else self._oracle_func(self.task)
        return feat[: self._use_mask_feat_dim]

    """
    return a list of [aux_mask, action, reward, done, info]
    """
    def run(self, target, max_steps, temperature=None):
        task = self.task
        trainer = self.trainer
        target_id = common.target_instruction_dict[target]
        #self._cached_target_id = target_id
        #trainer.set_target(target)
        consistent_target = (target == self.task.get_current_target())

        episode_stats = []
        obs = task._cached_obs
        for _st in range(max_steps):
            # mask feature if necessary
            mask = None if self._use_mask_feat_dim is None else self._get_feature_mask()
            # get action
            action, _ = trainer.action(obs, return_numpy=True,
                                       target=[[target_id]] if self.pass_target else None,
                                       temperature=temperature,
                                       mask_input= None if mask is None else [[mask]])
            action = int(action.squeeze())
            # environment step
            _, rew, done, info = task.step(action)
            feature_mask = task.get_feature_mask() if self._oracle_func is None else self._oracle_func(task)
            episode_stats.append((feature_mask, action, rew, done, info))
            # check terminate
            if done or (not consistent_target and self.check_terminate(target_id, feature_mask, action)):
                break
        return episode_stats
