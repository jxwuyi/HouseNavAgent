from headers import *
import common
import utils

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
    all_gpus = common.get_gpus_for_rendering()
    assert (len(all_gpus) > 0), 'No GPU found! There must be at least 1 GPU for rendering!'
    if args['render_gpu'] is not None:
        gpu_ids = args['render_gpu'].split(',')
        render_gpus = [all_gpus[int(k)] for k in gpu_ids]
    else:
        render_gpus = all_gpus
    config['render_devices'] = tuple(render_gpus)
    config['segment_input'] = (args['segment_input'] == 'color')
    config['depth_input'] = args['depth_input']
    config['target_mask_input'] = args['target_mask_input']
    config['multi_target'] = True
    config['object_target'] = args['object_target']
    config['fixed_target'] = args['fixed_target']
    config['mask_feature_dim'] = len(common.all_target_instructions) if ('mask_feature' in args) and args['mask_feature'] else None
    config['outdoor_target'] = True

    config['n_frame'] = args['n_frame']

    config['t_max'] = -1
    config['seed'] = args['seed']
    config['log_rate'] = args['log_rate']
    config['save_dir'] = args['save_dir']
    return config


def gen_data(args):
    np.random.seed(args['seed'])
    random.seed(args['seed'])
    part_id = args['part_id']
    print('>> Data Gen Part#{} Start (house range = {})...'.format(part_id, args['house_range']))
    log_rate = args['log_rate']
    dur = time.time()

    logger = utils.MyLogger(args['save_dir'], clear_file=True, filename=args['log_file'], keep_file_handler=True)

    task = common.create_env(k=args['house_range'],hardness=args['hardness'],
                             max_birthplace_steps=args['max_birthplace_steps'],
                             success_measure='see-stop',
                             segment_input=args['segment_input'], depth_input=args['depth_input'],
                             max_steps=-1, render_device=args['device_id'],
                             genRoomTypeMap=args['mask_feature_dim'] is not None,
                             cacheAllTarget=True, use_discrete_action=True,
                             include_object_target=args['object_target'],
                             target_mask_input=args['mask_feature_dim'],
                             include_outdoor_target=True,
                             discrete_angle=True,
                             cache_supervision=False,
                             cache_discrete_angles=True,
                             multithread_api=True)
    n_samples = args['sample_size']
    neg_rate = args['neg_rate']
    total_samples = n_samples * (1 + neg_rate)
    target = args['fixed_target']
    data = []
    labels = np.zeros(total_samples, dtype=np.uint8)
    labels[: n_samples] = 1
    birth_infos = []

    n_frame = args['n_frame']
    stop_action = common.n_discrete_actions - 1

    def fetch_segmentation_channels(frames):
        if not args['segment_input']:
            return None
        if args['depth_input']:
            return frames[:,:,:3]
        return frames
        

    # logging related
    report_index = set([int(total_samples // log_rate * i) for i in range(1, log_rate)])

    # generate positive samples
    logger.print(' --> Part#%d: positive data collecting ....' % part_id)
    task.reset_hardness(max_birthplace_steps=0)
    for i in range(n_samples):
        while True:
            task.reset(target=target)
            cur_info = task.info
            cur_frames = task._render_panoramic(n_frames=n_frame)  # assume 4 frames
            flag_semantic = False
            for frame in cur_frames:
                task.last_obs = None
                task._cached_seg = None
                task._gen_target_mask(seg_frame=fetch_segmentation_channels(frame))
                if task._is_success(0, grid=None, act=stop_action):
                    flag_semantic = True
                    break
            if flag_semantic:
                break
        data.append(cur_frames)
        birth_infos.append(cur_info)
        # logging
        if i in report_index:
            elap = time.time() - dur
            logger.print(" ---> Part#%d: Total Finished %d / %d, Percent = %.3f, Time Elapsed = %.3fs, Avg Elap = %.4fs" % (part_id, i + 1, total_samples, (i + 1) / total_samples, elap, elap / (i+1)))

    # generate negative samples
    logger.print(' --> Part#%d: negative data collecting ....' % part_id)
    task.reset_hardness(max_birthplace_steps=1000000, min_birth_grid_dist=3)
    n_neg_samples = n_samples * neg_rate
    for i in range(n_neg_samples):
        sample_ptr = i + n_samples
        while True:
            try:
                task.reset(target=target)
                break
            except Exception as e:
                continue
        birth_infos.append(task.info)
        data.append(task._render_panoramic(n_frames=n_frame))  # assume 4 frames
        # logging
        if sample_ptr in report_index:
            elap = time.time() - dur
            logger.print(" ---> Part#%d: Total Finished %d / %d, Percent = %.3f, Time Elapsed = %.3fs, Avg Elap = %.4fs" % (part_id, sample_ptr + 1, total_samples, (sample_ptr + 1) / total_samples, elap, elap / (i+1)))


    logger.print(" ---> Part#%d: Finished, Time Elapsed = %.3f" % (part_id, time.time() - dur))
    file_name = args['storage_file']
    logger.print(" ---> Part#{}: Dumping to {} ...".format(part_id, file_name))
    with open(file_name, 'wb') as f:
        pickle.dump([args, birth_infos, data, labels], f)
    logger.print(" ---> Part#%d: Done!" % part_id)
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
    n_pos_samples = args['sample_size']
    neg_sample_rate = args['neg_rate']
    total_samples = n_pos_samples * (1 + neg_sample_rate)

    proc_args = []
    prev_house_id = 0
    for i in range(n_part):
        data_size = n_pos_samples // n_proc
        house_size = n_house // n_proc
        if i < (total_samples % n_proc):
            data_size += 1
            house_size += 1
        cur_config = config.copy()
        cur_config['sample_size'] = data_size
        cur_config['neg_rate'] = neg_sample_rate
        cur_config['house_range'] = (prev_house_id, prev_house_id + house_size)
        cur_config['part_id'] = i
        cur_config['device_id'] = config['render_devices'][i % n_device]

        cur_config['storage_file'] = os.path.join(args['save_dir'], 'partition%d.pkl' % i)
        cur_config['log_file'] = 'partition%d_log.txt' % i
        proc_args.append((cur_config,))
        prev_house_id += house_size

    if n_proc > 1:
        from multiprocessing import Pool
        with Pool(n_proc) as pool:
            time_elapsed = pool.starmap(gen_data, proc_args)  # parallel version for initialization
    else:
        time_elapsed = []
        for config in proc_args:
            time_elapsed.append(gen_data(*config))
    print('++++++++++ Done +++++++++++')
    print(' >> Accumulative Time Elapsed = %.3f' % sum(time_elapsed))



def parse_args():
    parser = argparse.ArgumentParser("Panoramic Data Generator for 3D House Navigation")
    # Select Task
    parser.add_argument("--task-name", choices=['roomnav', 'objnav'], default='roomnav')
    # Environment
    parser.add_argument("--env-set", choices=['small', 'train', 'test', 'color'], default='small')
    parser.add_argument("--n-house", type=int, default=1,
                        help="number of houses to train on. Should be no larger than --n-proc")
    parser.add_argument("--seed", type=int, help="random seed")
    parser.add_argument("--hardness", type=float, help="real number from 0 to 1, indicating the hardness of the environment")
    parser.add_argument("--max-birthplace-steps", type=int, help="int, the maximum steps required from birthplace to target")
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
    parser.add_argument("--include-object-target", dest='object_target', action='store_true',
                        help="when this flag is set, target can be also an object. Only effective when --multi-target")
    parser.set_defaults(object_target=False)
    parser.add_argument("--fixed-target", type=str, help="fixed training targets: candidate values room, object or any-room/object")
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
    parser.add_argument("--n-frame", type=int, default=4,
                        help="[Data] number of frames per panoramic view")
    parser.add_argument("--sample-size", type=int, default=10000,
                        help="[Data] number of data samples to generate. This will be uniformly distributed over houses")
    parser.add_argument("--neg-rate", type=int, default=1,
                        help="[Data] Negative samples for postive sample")
    parser.add_argument("--log-rate", type=int, default=100,
                        help="[Data] number of times for recording progress in each partition")

    ###################################################
    # Checkpointing
    parser.add_argument("--save-dir", type=str, default="./_sup_data_/panoramic", help="directory in which data samples are stored")

    return parser.parse_args()

if __name__ == '__main__':
    cmd_args = parse_args()

    common.set_house_IDs(cmd_args.env_set)
    print('>> Environment Set = <%s>, Total %d Houses!' % (cmd_args.env_set, len(common.all_houseIDs)))

    common.ensure_object_targets(cmd_args.object_target)

    if cmd_args.fixed_target is not None:
        allowed_targets = list(common.target_instruction_dict.keys())
        assert cmd_args.fixed_target in allowed_targets, '--fixed-target specified an invalid target <{}>!'.format(cmd_args.fixed_target)
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

    assert cmd_args.segment_input in ['none', 'color'], '[Segment-Input] Currently only support <none> and <color>'

    args = cmd_args.__dict__

    # store training args
    config_file = args['save_dir']
    if config_file[-1] != '/': config_file = config_file + '/'
    config_file = config_file + 'config_args.json'
    with open(config_file, 'w') as f:
        json.dump(args, f)

    run(args)
