from headers import *
import common
import utils
import json

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


###############################
# Copy from semantic_train.py
###############################
def create_policy(args, observation_shape, n_class):
    model = common.CNNClassifier(observation_shape, n_class,
                          #hiddens=[64, 64, 128, 128],
                          #kernel_sizes=5, strides=2,
                          #linear_hiddens=[128, 64],
                          hiddens=[4, 8, 16, 16, 32, 32, 64, 64, 128, 256],
                          kernel_sizes=[3, 3, 3, 3, 3, 3, 3, 3, 3, 3],
                          strides=[1, 1, 1,  2,  1, 2, 1, 2, 1, 2],
                          linear_hiddens=[32],
                          use_batch_norm=args['batch_norm'],
                          multi_label=False,
                          dropout_rate=0.05)
    if common.use_cuda:
        if 'train_gpu' in args:
            train_gpus = args['train_gpu']
            if isinstance(train_gpus, list):  # multi-gpu
                #model = torch.nn.DataParallel(model, device_ids=train_gpus)
                model.cuda(device_id=train_gpus[0])
            else:  # single gpu
                model.cuda(device_id=train_gpus)  # TODO: Actually we only support training on gpu_id=0
        else:
            model.cuda()
    return model


def create_trainer(args, n_class):
    observation_shape = common.observation_shape
    model_gen = lambda: create_policy(args, observation_shape, n_class)
    trainer = common.SemanticTrainer(model_gen, observation_shape, n_class, args)
    return trainer
###############################


class SemanticOracle(object):
    def __init__(self, model_dir=None, model_device=None, include_object=False):
        self.allowed_targets = ALLOWED_TARGET_ROOM_TYPES
        if include_object: self.allowed_targets = self.allowed_targets + ALLOWED_OBJECT_TARGET_TYPES
        self.n_target = len(self.allowed_targets)
        self.classifiers = None
        if model_dir is None: return
        if not os.path.exists(model_dir):
            print('[SemanticOracle] model_dir <{}> not found!'.format(model_dir))
        self.classifiers = []
        if isinstance(model_device, int): model_device=[model_device]
        for i, target in enumerate(self.allowed_targets):
            print('---> current target = {}'.format(target))
            cur_dir = os.path.join(model_dir, target)
            assert os.path.exists(cur_dir), '[SemanticOracle] model dir <{}> for target <{}> not found!'.format(cur_dir, target)
            config_file = os.path.join(cur_dir, 'train_args.json')
            assert os.path.exists(config_file), '[SemanticOracle] config file <{}> for target <{}> not found!'.format(config_file, target)
            with open(config_file, 'r') as f:
                args = json.load(f)
            if 'train_gpu' in args: del args['train_gpu']
            args['train_gpu'] = model_device[i % len(model_device)]
            cur_trainer = create_trainer(args, n_class=2)
            cur_trainer.load(cur_dir, version='best')
            self.classifiers.append(cur_trainer)    # TODO: multi-class softmax classifier
        print('[SemanticOracle] Successfully Launched trainers for target <{}>'.format(self.allowed_targets))
    
    @property
    def targets(self):
        return self.allowed_targets

    def get_mask_feature(self, np_frame, threshold=None):
        """
        threshold: when not None, return a np.array with binary signals; otherwise return a list of float number
        """
        if len(np_frame.shape) == 3:
            np_frame = np_frame[np.newaxis, ...]
        ret = np.zeros(self.n_target, dtype=(np.uint8 if threshold is not None else np.float))
        for i, trainer in enumerate(self.classifiers):
            prob = trainer.action(np_frame, return_numpy=True)[0]
            if threshold is not None:
                ret[i] = (prob[0] > threshold)
            else:
                ret[i] = prob[0]
        return ret
    
    def to_string(self, probs):
        return ["%.2f" % probs[i] for i in range(len(probs))]
