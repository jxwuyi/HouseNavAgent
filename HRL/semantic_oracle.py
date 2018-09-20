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
                          dropout_rate=args['dropout_rate'],
                          stack_frame=args['stack_frame'],
                          self_attention_dim=args['self_attention_dim'])
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

class OracleFunction(object):
    def __init__(self, oracle, threshold=0.5, filter_steps=None):
        self.oracle = oracle
        self.n_target = oracle.n_target
        self.threshold = threshold
        if (filter_steps is not None) and (filter_steps < 2):
            filter_steps = None
        self.filter_steps = filter_steps
        self._filter_cnt = None if filter_steps is None else np.zeros(self.n_target, dtype=np.int32)
        self._zero_mask = np.zeros(self.n_target, dtype=np.uint8)
        self.stack_frame = oracle.stack_frame
        self.flag_panoramic = oracle.panoramic
        self._step_cnt = 0
        self._frame_stack = None if (self.stack_frame is not None) and not self.flag_panoramic else [None] * self.stack_frame
    
    def reset(self):
        self._step_cnt = 0
        self._filter_cnt[:] = 0
        self._frame_stack = None if (self.stack_frame is not None) and not self.flag_panoramic else [None] * self.stack_frame

    def get(self, task, return_current=False):
        if self.flag_panoramic:   # using panoramic view
            cur_mask = self.oracle.get_mask_feature(task._render_panoramic(n_frames=self.stack_frame), threshold=self.threshold)
        else:
            cur_obs = task._cached_obs
            if self.stack_frame is None:
                cur_mask = self.oracle.get_mask_feature(cur_obs, threshold=self.threshold)
            else:
                self._frame_stack = self._frame_stack[1:] + [cur_obs]
                cur_mask = self.oracle.get_mask_feature(self._frame_stack, threshold=self.threshold)
        self._step_cnt += 1
        ret_val = None
        if self.filter_steps is not None:
            self._filter_cnt *= cur_mask
            self._filter_cnt += cur_mask
            if self._step_cnt < self.filter_steps:
                ret_val = self._zero_mask
            else:
                ret_val = (self._filter_cnt >= self.filter_steps).astype(np.uint8)
        else:
            ret_val = cur_mask
        if return_current:
            return ret_val, cur_mask
        else:
            return ret_val


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
        self.stack_frame = 0
        self.panoramic=None
        for i, target in enumerate(self.allowed_targets):
            print('---> current target = {}'.format(target))
            cur_dir = os.path.join(model_dir, target)
            assert os.path.exists(cur_dir), '[SemanticOracle] model dir <{}> for target <{}> not found!'.format(cur_dir, target)
            config_file = os.path.join(cur_dir, 'train_args.json')
            assert os.path.exists(config_file), '[SemanticOracle] config file <{}> for target <{}> not found!'.format(config_file, target)
            with open(config_file, 'r') as f:
                args = json.load(f)
            if 'panoramic' in args:
                if (self.panoramic is None):
                    self.panoramic = args['panoramic']
                else:
                    assert self.panoramic == args['panoramic'], '[ERROR] when --panoramic, all the sub-classifers must all has --panoramic option on'
            if 'train_gpu' in args: del args['train_gpu']
            if ('stack_frame' in args) and args['stack_frame']:
                self.stack_frame = max(self.stack_frame, args['stack_frame'])
            args['train_gpu'] = model_device[i % len(model_device)]
            cur_trainer = create_trainer(args, n_class=2)
            cur_trainer.load(cur_dir, version='best')
            self.classifiers.append(cur_trainer)    # TODO: multi-class softmax classifier
        if self.stack_frame <= 1:
            self.stack_frame = None
        print('[SemanticOracle] Successfully Launched trainers for target <{}>'.format(self.allowed_targets))
    
    @property
    def targets(self):
        return self.allowed_targets

    def get_mask_feature(self, np_frame, threshold=None):
        """
        threshold: when not None, return a np.array with binary signals; otherwise return a list of float number
        """
        if isinstance(np_frame, list):
            assert self.stack_frame and (self.stack_frame == len(np_frame))
            shape = None
            for i in range(len(np_frame)):
                if np_frame[i] is not None:
                    if len(np_frame[i].shape) == 4:
                        np_frame[i] = np_frame[i][0]
                    shape = np_frame[i].shape
            for i in range(len(np_frame)):
                if np_frame[i] is None:
                    np_frame[i] = np.zeros(shape, dtype=np.uint8)
            np_frame_list = np_frame
            np_frame = np.stack(np_frame_list)[np.newaxis, ...]
        else:
            assert not self.stack_frame
            if len(np_frame.shape) == 3:
                np_frame = np_frame[np.newaxis, ...]
        ret = np.zeros(self.n_target, dtype=(np.uint8 if threshold is not None else np.float))
        for i, trainer in enumerate(self.classifiers):
            if trainer.stack_frame == self.stack_frame:
                cur_frame = np_frame
            elif trainer.stack_frame:
                cur_frame = np_frame[:, -trainer.stack_frame:, ...]
            else:
                cur_frame = np_frame[:, -1, ...]
            prob = trainer.action(cur_frame, return_numpy=True)[0]
            if threshold is not None:
                ret[i] = (prob[0] > threshold)
            else:
                ret[i] = prob[0]
        return ret
    
    def to_string(self, probs):
        return ["%.2f" % probs[i] for i in range(len(probs))]
