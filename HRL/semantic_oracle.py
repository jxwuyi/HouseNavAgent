from headers import *
import common
import utils
import json

import sys, os, platform, time

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
                model.cuda(device=train_gpus[0])
            else:  # single gpu
                model.cuda(device=train_gpus)  # TODO: Actually we only support training on gpu_id=0
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
    def __init__(self, oracle, threshold=0.5, filter_steps=None, batched_size=None):
        self.oracle = oracle
        self.n_target = oracle.n_target
        self.threshold = threshold
        if (filter_steps is not None) and (filter_steps < 2):
            filter_steps = None
        self.filter_steps = filter_steps
        self._filter_cnt = None if filter_steps is None else np.zeros(self.n_target, dtype=np.int32)
        self.batched_size = batched_size
        self._zero_mask = np.zeros(self.n_target, dtype=np.uint8)
        self.stack_frame = oracle.has_stack_frame
        self.flag_panoramic = oracle.has_panoramic
        self.pano_stack_frame = oracle.pano_stack
        self._step_cnt = 0
        self._frame_stack = None if self.stack_frame is None else [None] * self.stack_frame
    
    def reset(self):
        self._step_cnt = 0
        self._filter_cnt[:] = 0
        self._frame_stack = None if self.stack_frame is None else [None] * self.stack_frame

    def get(self, task, return_current_prob=False):
        # get recent frames
        cur_obs = task._cached_obs
        if self.stack_frame is None:
            recent_frames = cur_obs
        else:
            self._frame_stack = self._frame_stack[1:] + [cur_obs]
            recent_frames = self._frame_stack
        # get panoramic frames
        pano_frames = None if not self.flag_panoramic else task._render_panoramic(n_frames=self.pano_stack_frame)

        # compute mask feature
        if return_current_prob:
            cur_prob = self.oracle.get_mask_feature(recent_frames, pano_frames, threshold=None)
            cur_mask = self.oracle.to_binary(cur_prob, threshold=self.threshold)
        else:
            cur_mask = self.oracle.get_mask_feature(recent_frames, pano_frames, threshold=self.threshold)
            cur_prob = None

        # run filtering
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

        if return_current_prob:
            return ret_val, cur_prob
        else:
            return ret_val
    
    def batched_clear(self):
        self.batched_pano = [] if self.flag_panoramic else None
        self.batched_frame = []

    def batched_add(self, task):
        cur_obs = task._cached_obs
        if self.flag_panoramic:
            self.batched_pano.append(task._render_panoramic(n_frames=self.pano_stack_frame))
        if self.stack_frame is None:
            self.batched_frame.append(cur_obs)
        else:
            if self._frame_stack[-1] is None:
                shape = cur_obs.shape
                for i in len(self._frame_stack):
                    if self._frame_stack[i] is None:
                        self._frame_stack[i] = np.zeros(shape, dtype=np.uint8)
            self.batched_frame.append([a for a in self._frame_stack])
        
    def batched_get(self):
        # compute per-frame masks, [n_steps/batch, n_target]
        batched_mask = self.oracle.batched_get_mask_feature(self.batched_frame, batched_pano=self.batched_pano, threshold=self.threshold)
        n_steps = batched_mask.shape[0]

        # run filtering
        ret_list = []
        for t in range(n_steps):
            self._step_cnt += 1
            ret_val = None
            cur_mask = batched_mask[t]
            if self.filter_steps is not None:
                self._filter_cnt *= cur_mask
                self._filter_cnt += cur_mask
                if self._step_cnt < self.filter_steps:
                    ret_val = self._zero_mask
                else:
                    ret_val = (self._filter_cnt >= self.filter_steps).astype(np.uint8)
            else:
                ret_val = cur_mask
            ret_list.append(ret_val)
        return ret_list


class SemanticOracle(object):
    def __init__(self, model_dir, model_device=None, include_object=False):
        self.allowed_targets = ALLOWED_TARGET_ROOM_TYPES
        if include_object: self.allowed_targets = self.allowed_targets + ALLOWED_OBJECT_TARGET_TYPES
        self.n_target = len(self.allowed_targets)
        self.classifiers = None
        assert os.path.exists(model_dir), '[SemanticOracle] model_dir <{}> not found!'.format(model_dir)
        
        if os.path.isdir(model_dir):
            all_dirs = [os.path.join(model_dir, target) for target in self.allowed_targets]
        else:
            print('[SemanticOracle] Loading Classifer Info from <{}>...'.format(model_dir))
            with open(model_dir, 'r') as f:
                model_config = json.load(f)
            assert all([(target in model_config) for target in self.allowed_targets]), '[SemanticOracle] Missing Key in Model Config <{}>!'.format(model_dir)
            all_dirs = [model_config[target] for target in self.allowed_targets]

        self.classifiers = []
        if isinstance(model_device, int): model_device=[model_device]
        self.has_stack_frame = None
        self.has_panoramic = False
        self.pano_stack = None
        for i, target in enumerate(self.allowed_targets):
            print('---> current target = {}'.format(target))
            cur_dir = all_dirs[i]
            assert os.path.exists(cur_dir), '[SemanticOracle] model dir <{}> for target <{}> not found!'.format(cur_dir, target)
            config_file = os.path.join(cur_dir, 'train_args.json')
            assert os.path.exists(config_file), '[SemanticOracle] config file <{}> for target <{}> not found!'.format(config_file, target)
            with open(config_file, 'r') as f:
                args = json.load(f)
            if ('panoramic' in args) and args['panoramic']:
                self.has_panoramic = True
                if self.pano_stack is None:
                    self.pano_stack = args['stack_frame']
                else:
                    assert self.pano_stack == args['stack_frame'], '[SemanticOracle] all panoramic classifiers must have the same stack_frame size!'
            if 'train_gpu' in args: del args['train_gpu']
            if ('stack_frame' in args) and (('panoramic' not in args) or not args['panoramic']):
                assert args['stack_frame'] is not None
                if self.has_stack_frame is None:
                    self.has_stack_frame = args['stack_frame']
                else:
                    assert self.has_stack_frame == args['stack_frame']
            args['train_gpu'] = model_device[i % len(model_device)]
            cur_trainer = create_trainer(args, n_class=2)
            cur_trainer.load(cur_dir, version='best')
            cur_trainer.eval()
            cur_trainer.panoramic = ('panoramic' in args) and args['panoramic']
            self.classifiers.append(cur_trainer)    # TODO: multi-class softmax classifier
        if (self.has_stack_frame is not None) and (self.has_stack_frame <= 1):
            self.has_stack_frame = None
        if self.has_panoramic:
            assert self.pano_stack is not None
        print('[SemanticOracle] Successfully Launched trainers for target <{}>'.format(self.allowed_targets))
    
    @property
    def targets(self):
        return self.allowed_targets

    def get_mask_feature(self, recent_frames=None, pano_frames=None, threshold=None):
        """
        recent_frames: a list of recent frames, or a single frame
        pano_frames: a list of frames, the panoramic view
        threshold: when not None, return a np.array with binary signals; otherwise return a list of float number
        """
        if recent_frames is not None:
            var_frame = None
            if isinstance(recent_frames, list):
                shape = None
                for i in range(len(recent_frames)):
                    if recent_frames[i] is not None:
                        if len(recent_frames[i].shape) == 4:
                            recent_frames[i] = recent_frames[i][0]
                        shape = recent_frames[i].shape
                for i in range(len(recent_frames)):
                    if recent_frames[i] is None:
                        recent_frames[i] = np.zeros(shape, dtype=np.uint8)
                np_frame = np.stack(recent_frames)[np.newaxis, ...]
            else:
                np_frame = recent_frames
                if len(np_frame.shape) == 3:
                    np_frame = np_frame[np.newaxis, ...]
        else:
            np_frame = None
        if pano_frames is not None:
            assert isinstance(pano_frames, list) and (len(pano_frames) == self.pano_stack)
            np_pano = np.stack(pano_frames)[np.newaxis, ...]
            var_pano = None
        else:
            assert not self.has_panoramic
            np_pano = None

        ret = np.zeros(self.n_target, dtype=(np.uint8 if threshold is not None else np.float))
        for i, trainer in enumerate(self.classifiers):
            if trainer.panoramic:
                if var_pano is None:
                    var_pano = trainer._create_gpu_tensor(np_pano, return_variable=True, volatile=True)
                prob = trainer.action(var_pano, return_numpy=True, input_tensor=True)[0]
            else:
                if trainer.stack_frame is None:
                    if isinstance(recent_frames, list):
                        cur_frame = np_frame[:, -1, ...]
                    else:
                        cur_frame = np_frame
                else:
                    cur_frame = np_frame
                prob = trainer.action(cur_frame, return_numpy=True)[0]
            if threshold is not None:
                ret[i] = (prob[0] > threshold)
            else:
                ret[i] = prob[0]
        return ret
    
    def batched_get_mask_feature(self, batched_frames=None, batched_pano=None, threshold=None):
        """
        batched_frames: a list of stacked frames
        batched_pano: a list of stacked panoramic frames
        threshold: when not None, return a np.array with binary signals; otherwise return a list of float number
        """
        batch_size = len(batched_frames) if batched_frames is not None else len(batched_pano)
        np_frame = np.array(batched_frames) if batched_frames is not None else None
        np_pano = np.array(batched_pano) if batched_pano is not None else None
        var_pano = None
        var_frame = None

        ret = np.zeros((batch_size, self.n_target), dtype=(np.uint8 if threshold is not None else np.float))
        for i, trainer in enumerate(self.classifiers):
            if trainer.panoramic:
                if var_pano is None:
                    var_pano = trainer._create_gpu_tensor(np_pano, return_variable=True, volatile=True)
                prob = trainer.action(var_pano, return_numpy=True, input_tensor=True)[:, 0]
            else:
                if var_frame is None:
                    var_frame = trainer._create_gpu_tensor(np_frame, return_variable=True, volatile=True)
                prob = trainer.action(var_frame, return_numpy=True, input_tensor=True)[:, 0]
            if threshold is not None:
                ret[:, i] = (prob[0] > threshold)
            else:
                ret[:, i] = prob[0]
        return ret

    def to_binary(self, probs, threshold):
        return (probs > threshold).astype(np.uint8)

    def to_string(self, probs):
        return ["%.2f" % probs[i] for i in range(len(probs))]
