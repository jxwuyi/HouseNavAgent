from headers import *
import common

import os, time, pickle, argparse


def render_episode(env, images):
    for im in images:
        env.render(im)
        time.sleep(0.5)


def visualize(args, all_stats, config):
    env = common.create_env(config.house)
    env.reset_render()
    for it, stats in enumerate(all_stats):
        if args.only_success and (stats['success'] == 0):
            continue
        if args.only_good and (stats['good'] == 0):
            continue
        if stats['length'] > args.max_episode_len:
            continue
        print('Episode#%d, Length = %d' % (it + 1, stats['length']))
        print(' >> Success = %d' % stats['success'])
        print(' >> Stay in Room = %d' % stats['good'])
        render_episode(env, stats['images'])
        print('Press Any Key To Continue ...')


def parse_args():
    parser = argparse.ArgumentParser("Visualization for 3D House Navigation")
    # config
    parser.add_argument("file", type=str, help="evaluation stats file")
    parser.add_argument("--max-episode-len", type=int, default=2000,
                        help="only display episode with length smaller than this number")
    parser.add_argument("--only-success", action='store_true', default=False,
                        help='Only display those successful runs')
    parser.add_argument("--only-good", action='store_true', default=False,
                        help='Only display those runs where agent reaches the target room')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    assert os.path.exists(args.file), 'Stats File Not Exists!'

    with open(args.file, 'rb') as f:
        [stats, config] = pickle.load(f)

    visualize(args, stats, config)
