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

"""
Learning a Bayesian Graph over Objects and Rooms
Rooms: 9 rooms total
    8 rooms + indoor
Objects: 15 objects total
Parameters:
  --> for each pair of rooms R1, R2: connect(R1,R2) ~ Bernoulli(theta(R1, R2))
  --> for each room R and object O: contain(R, O) ~ Bernoulli(theta(R, O))
  --> noisy observation parameters: if connect(X, Y) == 1: obs ~ Bernoulli(psi_1)
                                    else: obs ~ Bernoulli(psi_2)
Total #Params = 9 * 8 / 2 + 9 * 15 + 2 = 308
"""

class GraphPlanner(object):
    def __init__(self, task, motion):
        self.task = task
        self.motion = motion

    def parameters(self):
        return None

    def set_param(self):
        return None

    def learn(self):
        pass

    def observe(self):
        pass

    def infer(self):
        pass

    def clear(self):
        pass