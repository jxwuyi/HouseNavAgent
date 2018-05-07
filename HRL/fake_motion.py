from headers import *

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
n_targets = len(ALLOWED_OBJECT_TARGET_INDEX) + len(ALLOWED_PREDICTION_ROOM_TYPES)
all_allowed_target_names = ALLOWED_TARGET_ROOM_TYPES + ALLOWED_OBJECT_TARGET_TYPES

#########################
# Hyper Parameters
#############
rate_succ_within_range = 0.97
range_sight = 5  # steps to be seen
rate_succ_within_sight = 0.85
range_reach = 10  # steps able to be reached
rate_succ_within_reach = 0.5

range_for_exploration = 15

rate_guess_right_direction = 0.65
steps_guess_right_direction = 10

def rand():
    return np.random.rand()

def _is_object_target(target):
    return target in ALLOWED_OBJECT_TARGET_INDEX

def _is_room_target(target):
    return target in ALLOWED_PREDICTION_ROOM_TYPES

def _get_target_index(target):
    if _is_room_target(target):
        return ALLOWED_PREDICTION_ROOM_TYPES[target]
    else:
        return ALLOWED_OBJECT_TARGET_INDEX[target]

class FakeMotion(BaseMotion):
    def __init__(self, task, trainer=None, pass_target=True):
        super(FakeMotion, self).__init__(task, trainer, pass_target)
        # fetch target mask graph
        self.env = task.env
        all_houses = self.env.all_houses if hasattr(self.env, 'all_houses') else [self.env.house]
        self.target_dist = []
        self.target_list = []
        self.target_rooms = []
        for house in all_houses:
            self.target_list.append(house.all_desired_targetTypes)
            self.target_rooms.append(len(house.all_desired_roomTypes))
            self.target_dist.append(house.get_graph())

    def _jump(self, cx, cy, flag_done=False):
        self.env.reset(x=cx, y=cy)
        mask = self.env.house.get_global_mask_feature(cx, cy)
        if flag_done:
            return mask, n_discrete_actions - 1, 10, True, self.task.info
        else:
            return mask, -1, 0, False, self.task.info

    def _run_single_step(self, target):
        house = self.env.house
        info = self.env.info
        # obs = self.task.cached_obs
        cx, cy = info['loc']
        gx, gy = info['grid']
        mask = house.get_global_mask_feature(cx, cy)
        dist = house.targetDist(target, gx, gy)
        if dist < 0:  # not connected
            return [self._jump(cx, cy, False)]
        target_idx = _get_target_index(target)
        if dist == 0:
            cx, cy = house.getRandomLocation(target)
            return [self._jump(cx, cy, (rand() < rate_succ_within_range))]

        # check whether within sight
        if dist <= house.getOptSteps(range_sight, self.task.move_sensitivity):
            if rand() < rate_succ_within_sight:
                cx, cy = house.getRandomLocation(target)
                return [self._jump(cx, cy, True)]  # directly reach the target
            else:
                cx, cy = house.getRandomLocationFromRange(target, (0, dist // 2))
                return [self._jump(cx, cy, False)]  # fail to reach

        # check object type
        object_within_room = False
        if _is_object_target(target):
            for i, r in enumerate(ALLOWED_TARGET_ROOM_TYPES):
                if mask[i] > 0:
                    room_mask = self.env.house.getRegionMaskForTarget(r)[0]
                    if room_mask[target_idx] > 0:
                        object_within_room = True
                        break

        # check whether possible to reach the target
        if object_within_room or (dist <= house.getOptSteps(range_reach, self.task.move_sensitivity)):
            if rand() < rate_succ_within_reach:
                cx, cy = house.getRandomLocation(target)
            else:
                cx, cy = house.getRandomLocationFromRange(target, (0, int(0.75 * dist + 1e-10)))
            return [self._jump(cx, cy, False)]

        # too faraway --- consider masks
        all_targets = self.target_list[house._id]
        good_targets = []
        steps_to_targets = []
        for t in all_targets:
            d = house.targetDist(t, gx, gy)
            if d >= 0:
                good_targets.append(t)
            steps_to_targets.append(10000000 if d < 0 else house.getOptSteps(d, self.task.move_sensitivity))

        if min(steps_to_targets) > range_for_exploration:
            if rand() < rate_guess_right_direction:
                lo = max(0, dist - house.getAllowedGridDist(steps_guess_right_direction))
                cx, cy = house.getRandomLocationFromRange(target, (lo, dist))
            else:
                new_t = np.random.choice(good_targets)
                new_dist = house.targetDist(new_t, gx, gy)
                lo = max(0, new_dist - house.getAllowedGridDist(steps_guess_right_direction))
                cx, cy = house.getRandomLocationFromRange(new_t, (lo, new_dist))
            return [self._jump(cx, cy, False)]

        close_targets = [all_targets[i] for i, s in enumerate(steps_to_targets) if s <= range_for_exploration]
        if (len(close_targets) < 2) and (target not in close_targets):
            close_targets.append(target)

        new_t = np.random.choice(close_targets)
        new_dist = house.targetDist(new_t, gx, gy)
        lo = max(0, new_dist - house.getAllowedGridDist(steps_guess_right_direction))
        cx, cy = house.getRandomLocationFromRange(new_t, (lo, new_dist))
        return [self._jump(cx, cy, False)]

    """
    return a list of [aux_mask, action, reward, done, info]
    """
    def run(self, target, max_steps):
        assert max_steps > 0
        ret = []
        while max_steps > 0:
            curr = self._run_single_step(target)
            ret.append(curr[0])
            if curr[0][3]:
                break
            max_steps -= 50
        return ret
