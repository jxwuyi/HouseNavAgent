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

        #cur_detail = dict(plan=task.get_optimal_plan(), targets=task.get_avail_targets(), graph=task.house.get_graph())
        #cur_detail=dict(graph=task.house.get_graph(), targets=task.get_avail_targets().copy())
        targets = task.get_avail_targets().copy()
        print('   ---> target done!')
        print(targets)
        graph = task.house.get_graph().copy()
        print('   ---> graph done!')
        print(graph)
        plan = task.get_optimal_plan()
        print('   ---> plan done!')
        cur_detail=dict(plan=plan,graph=graph,targets=targets)

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
    parser.add_argument("--filename", type=str, default='_eval_details.pkl', help="gpu for render")
    parser.add_argument("--plan-dist-iters", type=str,
                        help="Required iterations for each plan-distance birthplaces. In the format of Dist1:Number1,Dist2:Number2,...")
    return parser.parse_args()


if __name__ == '__main__':
    cmd_args = parse_args()
    
    print('Data Loading ....')
    with open(cmd_args.data, 'rb') as f:
        episode_stats, args = pickle.load(f)

    common.set_house_IDs(args.env_set, ensure_kitchen=(not args.multi_target))
    print('>> Environment Set = <%s>, Total %d Houses!' % (args.env_set, len(common.all_houseIDs)))

    if args.object_target:
        common.ensure_object_targets()
    
    if not os.path.exists(args.log_dir):
        print('Directory <{}> does not exist! Creating directory ...'.format(args.log_dir))
        os.makedirs(args.log_dir)

    args.render_gpu = cmd_args.render_gpu
    dict_args = args.__dict__

    if ('plan_dist_iters' in dict_args) and (dict_args['plan_dist_iters'] is not None):
        assert cmd_args.plan_dist_iters is not None
    if cmd_args.plan_dist_iters is not None:
        print('>> Parsing Plan Dist Iters ...')
        try:
            all_dist = cmd_args.plan_dist_iters.split(',')
            assert len(all_dist) > 0
            req = dict()
            total_req = 0
            for dat in all_dist:
                vals = dat.split(':')
                a, b = int(vals[0]), int(vals[1])
                assert (a > 0) and (b > 0) and (a not in req)
                req[a] = b
                total_req += b
            dict_args['plan_dist_iters'] = req
            assert dict_args['max_iters'] == total_req
            print(' ---> Parsing Done! Set Max-Iters to <{}>'.format(total_req))
            print('    >>> Details = {}'.format(req))
        except Exception as e:
            print('[ERROR] PlanDistIters Parsing Error for input <{}>!'.format(args.plan_dist_iters))
            raise e

    details = evaluate(episode_stats, dict_args)

    filename = os.path.join(cmd_args.log_dir, args.filename)
    print('Store Fetched Details to <{}>....'.format(filename))
    with open(filename, 'wb') as f:
        pickle.dump(details, f)
