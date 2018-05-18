from headers import *
import numpy as np
import common
import utils

import os, sys, time, json

ts = time.time()

all_rooms = common.ALLOWED_TARGET_ROOM_TYPES
all_objects = common.ALLOWED_OBJECT_TARGET_TYPES

flag_joint_object_model = True
#flag_joint_object_model = False

# model stored in <./_model_/nips_HRL/seg_large_birth15/only-XXXX/>
#  or .../any-object or .../any-room
#room_model_dir = './_model_/nips_HRL/visual_large_birth15/'
#object_model_dir = './_model_/nips_HRL/seg_mask_large_birth15/'

#room_model_dir = './_model_/nips_tune_old/seg_large_c5_3_1w/'   # visual_large_c5_3_1w
#object_model_dir = './_model_/nips_tune_old/seg_mask_large_c3-3-1w/'
#save_dir = './_graph_/mix_motion/nips_tune_old/seg_mask/'

#room_model_dir = ['./_model_/nips_tune/visual_large_c3_3_1w/']
#object_model_dir = ['./_model_/nips_tune/visual_large_c3-3-15k/']
#save_dir = ['./_graph_/mix_motion/nips_tune/visual/']

#room_model_dir.append('./_model_/nips_tune/seg_large_c3_3_1w/')
#object_model_dir.append('./_model_/nips_tune/seg_large_c3-3-15k/')
#save_dir.append('./_graph_/mix_motion/nips_tune/seg/')

#room_model_dir.append('./_model_/nips_tune/visual_large_c3_3_1w/')
#object_model_dir.append('./_model_/nips_tune/visual_mask_large_c3-3-15k/')
#save_dir.append('./_graph_/mix_motion/nips_tune/visual_mask/')

#room_model_dir = ['./_model_/nips_tune_old/seg_large_c5_3_1w/']
#object_model_dir = ['./_model_/nips_tune_obj/seg_large_c3_3_1w/']
#save_dir = ['./_graph_/mix_motion/nips_obj/seg/']

room_model_dir = ['./_model_/nips_mask_baseline/seg_large_c3_3_1w/']
object_model_dir = ['./_model_/nips_tune_old/seg_mask_large_c3-3-1w/']
save_dir = ['./_graph_/mix_motion/nips_mask_baseline/seg/']


def create_dict(room_model_dir, object_model_dir, save_dir):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    motion_dict = dict()

    # Generate Dicts for Rooms
    print('Generate Dicts for Room Targets ...')
    for room in all_rooms:
        prefix_dir = room_model_dir + room + '/'
        args_file = prefix_dir + 'train_args.json'
        with open(args_file, 'r') as f:
            args = json.load(f)
        args['warmstart'] = prefix_dir + 'ZMQA3CTrainer.pkl'
        motion_dict[room] = args

    # Generate Dicts for Objects
    print('Generate Dicts for Object Targets ...')
    for obj in all_objects:
        prefix_dir = object_model_dir
        if not flag_joint_object_model:  # separate object-policy
            prefix_dir = prefix_dir + obj + '/'
        else:
            prefix_dir = prefix_dir + 'any-object/'
        args_file = prefix_dir + 'train_args.json'
        with open(args_file, 'r') as f:
            args = json.load(f)
        args['warmstart'] = prefix_dir + 'ZMQA3CTrainer.pkl'
        motion_dict[obj] = args

    # Save
    dict_file = save_dir + 'motion_dict.json'
    print('Saving to <{}> ...'.format(dict_file))
    with open(dict_file, 'w') as f:
        json.dump(motion_dict, f)
    dur = time.time() - ts
    print(' >> Done!  Time Elapsed = %.5fs' % dur)


for room, obj, save in zip(room_model_dir, object_model_dir, save_dir):
    create_dict(room, obj, save)


