from headers import *
import common
import utils

import sys, os, platform, pickle, json, argparse, time

import numpy as np
import random

from HRL.fake_motion import FakeMotion
from HRL.rnn_motion import RNNMotion
from HRL.random_motion import RandomMotion
from HRL.mixture_motion import MixMotion, create_mixture_motion_trainer_dict


def create_motion(args, task, oracle_func=None):
    if args['motion'] == 'rnn':
        if (args['warmstart_dict'] is not None) and os.path.isfile(args['warmstart_dict']):
            with open(args['warmstart_dict'], 'r') as f:
                trainer_args = json.load(f)
        else:
            trainer_args = args
        common.process_observation_shape('rnn', trainer_args['resolution_level'],
                                         segmentation_input=trainer_args['segment_input'],
                                         depth_input=trainer_args['depth_input'],
                                         history_frame_len=1,
                                         target_mask_input=trainer_args['target_mask_input'])
        import zmq_train
        trainer = zmq_train.create_zmq_trainer('a3c', 'rnn', trainer_args)
        model_file = args['warmstart']
        if model_file is not None:
            trainer.load(model_file)
        trainer.eval()
        motion = RNNMotion(task, trainer,
                           pass_target=args['multi_target'],
                           term_measure=args['terminate_measure'],
                           oracle_func=oracle_func)
    elif args['motion'] == 'random':
        motion = RandomMotion(task, None, term_measure=args['terminate_measure'], oracle_func=oracle_func)
    elif args['motion'] == 'fake':
        motion = FakeMotion(task, None, term_measure=args['terminate_measure'], oracle_func=oracle_func)
    else: # mixture motion
        mixture_dict_file = args['mixture_motion_dict']
        try:
            with open(mixture_dict_file, 'r') as f:
                arg_dict = json.load(f)
        except Exception as e:
            print('Invalid Mixture Motion Dict!! file = <{}>'.format(mixture_dict_file))
            raise e
        trainer_dict, pass_tar_dict, obs_mode_dict = create_mixture_motion_trainer_dict(arg_dict)
        motion = MixMotion(task, trainer_dict, pass_tar_dict,
                           term_measure=args['terminate_measure'],
                           obs_mode=obs_mode_dict,
                           oracle_func=oracle_func)
        common.ensure_object_targets(args['object_target'])

    return motion


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)


def proc_info(info):
    return dict(yaw=info['yaw'], loc=info['loc'], grid=info['grid'],
                dist=info['dist'])


def evaluate(ep_stats, args):

    elap = time.time()

    # Do not need to log detailed computation stats
    common.debugger = utils.FakeLogger()

    fixed_target = args['fixed_target']
    if (fixed_target is not None) and (fixed_target != 'any-room') and (fixed_target != 'any-object'):
        assert fixed_target in common.n_target_instructions, 'invalid fixed target <{}>'.format(fixed_target)

    __backup_CFG = common.CFG.copy()
    if fixed_target == 'any-room':
        common.ensure_object_targets(False)

    if args['hardness'] is not None:
        print('>>>> Hardness = {}'.format(args['hardness']))
    if args['max_birthplace_steps'] is not None:
        print('>>>> Max BirthPlace Steps = {}'.format(args['max_birthplace_steps']))
    set_seed(args['seed'])
    task = common.create_env(args['house'], task_name=args['task_name'], false_rate=args['false_rate'],
                            hardness=args['hardness'], max_birthplace_steps=args['max_birthplace_steps'],
                            success_measure=args['success_measure'],
                            depth_input=args['depth_input'],
                            target_mask_input=args['target_mask_input'],
                            segment_input=args['segmentation_input'],
                            genRoomTypeMap=False,
                            cacheAllTarget=args['multi_target'],
                            render_device=args['render_gpu'],
                            use_discrete_action=True,
                            include_object_target=args['object_target'] and (fixed_target != 'any-room'),
                            include_outdoor_target=args['outdoor_target'],
                            discrete_angle=True,
                            min_birthplace_grids=args['min_birthplace_grids'])

    if (fixed_target is not None) and (fixed_target != 'any-room') and (fixed_target != 'any-object'):
        task.reset_target(fixed_target)

    if fixed_target == 'any-room':
        common.CFG = __backup_CFG
        common.ensure_object_targets(True)

    # create motion
    print('Start Evaluating ...')
    ep_plans = []

    t = 0
    seed = args['seed']
    
    for it in range(args['max_iters']):
        set_seed(seed + it + 1)  # reset seed
        task.reset(target=fixed_target)

        cur_detail = dict(plan=task.get_optimal_plan(), targets=task.get_avail_targets(), graph=task.house.get_graph())

        ep_plans.append(cur_detail)

        dur = time.time() - elap
        print('Episode#%d, Elapsed = %.4fs' % (it+1, dur))

    print('Done!!!!!')

    return ep_plans


def parse_args():
    parser = argparse.ArgumentParser("Evaluation Locomotion for 3D House Navigation")
    parser.add_argument("--data", type=str, help="path to eval_data.pkl")
    parser.add_argument("--log-dir", type=str, help="dir to store info")
    parser.add_argument("--render-gpu", type=int, help="gpu for render")
    return parser.parse_args()


if __name__ == '__main__':
    cmd_args = parse_args()
    
    print('Data Loading ....')
    with open(cmd_args.data, 'rb') as f:
        episode_stats, args = pickle.load(f)

    args.render_gpu = cmd_args.render_gpu
    dict_args = args.__dict__

    details = evaluate(episode_stats, dict_args)

    filename = os.path.join(cmd_args.log_dir, 'details.pkl')
    print('Store Fetched Details to <{}>....'.format(filename))
    with open(filename, 'wb') as f:
        pickle.dump(details)
