from headers import *
import common
import utils

from policy.rnn_discrete_actor_critic import DiscreteRNNPolicy

import os, sys, time, pickle, json, argparse
import numpy as np
import random

FLAG_SANITY_CHECK = True

def create_data_gen_config(args):
    config = dict()

    # task name
    config['task_name'] = args['task_name']

    # env param
    config['n_house'] = args['n_house']
    config['hardness'] = args['hardness']
    config['max_birthplace_steps'] = args['max_birthplace_steps']
    config['min_birthplace_grids'] = args['min_birthplace_grids']
    all_gpus = common.get_gpus_for_rendering()
    assert (len(all_gpus) > 0), 'No GPU found! There must be at least 1 GPU for rendering!'
    if args['render_gpu'] is not None:
        gpu_ids = args['render_gpu'].split(',')
        render_gpus = [all_gpus[int(k)] for k in gpu_ids]
    else:
        render_gpus = all_gpus
    config['render_devices'] = tuple(render_gpus)
    config['segment_input'] = args['segment_input']
    config['depth_input'] = args['depth_input']
    config['target_mask_input'] = args['target_mask_input']
    config['success_measure'] = args['success_measure']
    config['multi_target'] = args['multi_target']
    config['object_target'] = args['object_target']
    config['fixed_target'] = args['fixed_target']
    config['mask_feature_dim'] = len(common.all_target_instructions) if ('mask_feature' in args) and args['mask_feature'] else None
    config['outdoor_target'] = args['outdoor_target']

    config['t_max'] = args['t_max']
    config['seed'] = args['seed']
    return config


def gen_data(args):
    np.random.seed(args['seed'])
    random.seed(args['seed'])
    part_id = args['part_id']
    print('>> Data Gen Part#{} Start (house range = {})...'.format(part_id, args['house_range']))
    log_rate = 5
    dur = time.time()

    task = common.create_env(k=args['house_range'],hardness=args['hardness'],
                             max_birthplace_steps=args['max_birthplace_steps'],
                             min_birthplace_grids=args['min_birthplace_grids'],
                             success_measure=args['success_measure'],
                             segment_input=args['segment_input'], depth_input=args['depth_input'],
                             max_steps=-1, render_device=args['device_id'],
                             genRoomTypeMap=args['mask_feature_dim'] is not None,
                             cacheAllTarget=True, use_discrete_action=True,
                             include_object_target=args['object_target'],
                             target_mask_input=args['mask_feature_dim'],
                             include_outdoor_target=args['outdoor_target'],
                             cache_supervision=False,
                             cache_discrete_angles=True)
    n_samples = args['sample_size']
    target = args['fixed_target']
    t_max = args['t_max']
    if t_max <= 0: t_max = None
    data = []
    birth_infos = []

    # logging related
    report_index = set([int(n_samples // log_rate * i) for i in range(1, log_rate)])

    print(' --> Part#%d: data collecting ....' % part_id)

    for i in range(n_samples):
        task.reset(target=target)
        birth_infos.append(task.info)
        data.append(task.gen_supervised_plan(return_numpy_frames=True,
                                             max_allowed_steps=t_max,
                                             mask_feature_dim=args['mask_feature_dim']))  # np_frames, np_act, (optional) np_mask_feat

        if FLAG_SANITY_CHECK:
            assert task._sanity_check_supervised_plan(birth_infos[-1], data[-1][1])

        # logging
        if i in report_index:
            print(" ---> Part#%d: Finished %d / %d, Percent = %.3f, Time Elapsed = %.3f" % (part_id, i + 1, n_samples, (i + 1) / n_samples, time.time() - dur))

    print(" ---> Part#%d: Finished, Time Elapsed = %.3f" % (part_id, time.time() - dur))
    file_name = args['storage_file']
    print(" ---> Part#{}: Dumping to {} ...".format(part_id, file_name))
    with open(file_name, 'wb') as f:
        pickle.dump([args, birth_infos, data], f)
    print(" ---> Part#%d: Done!" % part_id)
    return time.time() - dur


def run(args=None):

    # Process Observation Shape
    common.process_observation_shape(model='rnn',
                                     resolution_level=args['resolution_level'],
                                     segmentation_input=args['segment_input'],
                                     depth_input=args['depth_input'],
                                     target_mask_input=args['target_mask_input'],
                                     history_frame_len=1)
    config = create_data_gen_config(args)

    n_house = args['n_house']
    n_proc = args['n_proc']
    n_part = args['n_partition']
    n_device = len(config['render_devices'])
    total_samples = args['sample_size']

    proc_args = []
    prev_house_id = 0
    for i in range(n_part):
        data_size = total_samples // n_proc
        house_size = n_house // n_proc
        if i < (total_samples % n_proc):
            data_size += 1
            house_size += 1
        cur_config = config.copy()
        cur_config['sample_size'] = data_size
        cur_config['house_range'] = (prev_house_id, prev_house_id + house_size)
        cur_config['part_id'] = i
        cur_config['device_id'] = config['render_devices'][i % n_device]
        cur_config['storage_file'] = os.path.join(args['save_dir'], '/partition%d.pkl' % i)
        proc_args.append((cur_config,))
        prev_house_id += house_size

    from multiprocessing import Pool
    with Pool(n_proc) as pool:
        time_elapsed = pool.starmap(gen_data, proc_args)  # parallel version for initialization
    print('++++++++++ Done +++++++++++')
    print(' >> Accumulative Time Elapsed = %.3f' % sum(time_elapsed))



def parse_args():
    parser = argparse.ArgumentParser("Supervision Data Generator for 3D House Navigation")
    # Special Job Tag
    parser.add_argument("--job-name", type=str, default='')
    # Select Task
    parser.add_argument("--task-name", choices=['roomnav', 'objnav'], default='roomnav')
    # Environment
    parser.add_argument("--env-set", choices=['small', 'train', 'test', 'color'], default='small')
    parser.add_argument("--n-house", type=int, default=1,
                        help="number of houses to train on. Should be no larger than --n-proc")
    parser.add_argument("--seed", type=int, help="random seed")
    parser.add_argument("--hardness", type=float, help="real number from 0 to 1, indicating the hardness of the environment")
    parser.add_argument("--max-birthplace-steps", type=int, help="int, the maximum steps required from birthplace to target")
    parser.add_argument("--min-birthplace-grids", type=int, default=0,
                        help="int, the minimum grid distance of the birthplace towards target. Default 0, namely possible to born with gird_dist=0.")
    parser.add_argument("--segmentation-input", choices=['none', 'index', 'color', 'joint'], default='none', dest='segment_input',
                        help="whether to use segmentation mask as input; default=none; <joint>: use both pixel input and color segment input")
    parser.add_argument("--depth-input", dest='depth_input', action='store_true',
                        help="whether to include depth information as part of the input signal")
    parser.set_defaults(depth_input=False)
    parser.add_argument("--target-mask-input", dest='target_mask_input', action='store_true',
                        help="whether to include target mask 0/1 signal as part of the input signal")
    parser.set_defaults(target_mask_input=False)
    parser.add_argument("--resolution", choices=['normal', 'low', 'tiny', 'high', 'square', 'square_low'],
                        dest='resolution_level', default='normal',
                        help="resolution of visual input, default normal=[120 * 90]")
    #parser.add_argument("--history-frame-len", type=int, default=4,
    #                    help="length of the stacked frames, default=4")
    #parser.add_argument("--max-episode-len", type=int, default=50, help="maximum episode length")
    parser.add_argument("--success-measure", choices=['see-stop', 'stop', 'stay', 'see'], default='see-stop',
                        help="criteria for a successful episode")
    parser.add_argument("--multi-target", dest='multi_target', action='store_true',
                        help="when this flag is set, a new target room will be selected per episode")
    parser.set_defaults(multi_target=False)
    parser.add_argument("--include-object-target", dest='object_target', action='store_true',
                        help="when this flag is set, target can be also an object. Only effective when --multi-target")
    parser.set_defaults(object_target=False)
    parser.add_argument("--include-mask-feature", dest='mask_feature', action='store_true',
                        help="when this flag is set, mast_feature will be fed to the neural network.")
    parser.set_defaults(mask_feature=False)
    parser.add_argument("--fixed-target", type=str, help="fixed training targets: candidate values room, object or any-room/object")
    parser.add_argument("--no-outdoor-target", dest='outdoor_target', action='store_false',
                        help="when this flag is set, we will exclude <outdoor> target")
    parser.set_defaults(outdoor_target=True)
    ########################################################
    # Multi-thread Parameters
    parser.add_argument("--render-gpu", type=str,
                        help="[Data] an integer or a ','-split list of integers, indicating the gpu-id for renderers")
    parser.add_argument("--n-proc", type=int, default=32,
                        help="[Data] number of processes for simulation")
    parser.add_argument("--n-partition", type=int, default=32,
                        help="[Data] number of partitions over houses")

    ########################################################
    # Data Generation Core Parameters
    parser.add_argument("--t-max", type=int, default=-1,
                        help="[Data] maximum number of horizon. <=0 if no constraint")
    parser.add_argument("--sample-size", type=int, default=10000,
                        help="[Data] number of data samples to generate. This will be uniformly distributed over houses")

    ###################################################
    # Checkpointing
    parser.add_argument("--save-dir", type=str, default="./_sup_data_", help="directory in which data samples are stored")

    return parser.parse_args()

if __name__ == '__main__':
    cmd_args = parse_args()

    assert cmd_args.success_measure == 'see-stop', 'currently only support success_measure <see-stop>'

    common.set_house_IDs(cmd_args.env_set, ensure_kitchen=(not cmd_args.multi_target))
    print('>> Environment Set = <%s>, Total %d Houses!' % (cmd_args.env_set, len(common.all_houseIDs)))

    common.ensure_object_targets(cmd_args.object_target)

    if cmd_args.fixed_target is not None:
        allowed_targets = list(common.target_instruction_dict.keys()) + ['any-room']
        if cmd_args.object_target:
            allowed_targets.append('any-object')
        assert cmd_args.fixed_target in allowed_targets, '--fixed-target specified an invalid target <{}>!'.format(cmd_args.fixed_target)
        if not ('any' in cmd_args.fixed_target):
            common.filter_house_IDs_by_target(cmd_args.fixed_target)
            print('[data_gen.py] Filter Houses By Fixed-Target <{}> to N=<{}> Houses...'.format(cmd_args.fixed_target, len(common.all_houseIDs)))

    if cmd_args.n_house > len(common.all_houseIDs):
        print('[data_gen.py] No enough houses! Reduce <n_house> to [{}].'.format(len(common.all_houseIDs)))
        cmd_args.n_house = len(common.all_houseIDs)

    assert cmd_args.n_proc <= cmd_args.n_partition
    assert cmd_args.n_partition <= cmd_args.n_house

    if cmd_args.seed is not None:
        np.random.seed(cmd_args.seed)
        random.seed(cmd_args.seed)

    if not os.path.exists(cmd_args.save_dir):
        print('Directory <{}> does not exist! Creating directory ...'.format(cmd_args.save_dir))
        os.makedirs(cmd_args.save_dir)

    args = cmd_args.__dict__

    if any([args[k] is not None for k in args.keys() if 'rew_shape' in k]):
        common.set_reward_shaping_params(args)

    # store training args
    config_file = args['save_dir']
    if config_file[-1] != '/': config_file = config_file + '/'
    config_file = config_file + 'config_args.json'
    with open(config_file, 'w') as f:
        json.dump(args, f)

    run(args)
