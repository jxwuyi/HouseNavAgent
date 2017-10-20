from headers import *
import common

import os, time, pickle, argparse

def render_episode(env, infos):
    images = []
    for info in infos:
        env.set_cam_info(info)
        images.append(env.render(renderMapLoc=info['loc'], display=False))
    return images

def show_episode(env, images):
    for i, im in enumerate(images):
        env.render(im)
        if i == 0:
            time.sleep(0.5)
        else:
            time.sleep(0.3)


def visualize(args, all_stats, config):
    common.resolution = (400, 300)
    common.set_house_IDs(args.env_set)
    env = common.create_env(config.house, hardness=config.hardness)
    env.reset_render()
    print('Resolution = {}'.format(env.resolution))
    total_len = 0
    total_succ = 0
    episode_images = []
    print('Rendering ....')
    elap = time.time()
    for it, stats in enumerate(all_stats):
        if args.only_success and (stats['success'] == 0):
            continue
        if args.only_good and (stats['good'] == 0):
            continue
        if stats['length'] > args.max_episode_len:
            continue
        if 'world_id' in stats:
            env.reset(stats['world_id'])
        episode_images.append((render_episode(env, stats['infos']), stats))
        if len(episode_images) % 10 == 0:
            print(' >>> %d Episode Rendered, Time Elapsed = %.4fs' % (len(episode_images), time.time()-elap))
        if len(episode_images) >= args.max_iters:
            break
    dur = time.time()-elap
    print('Total %d Episodes Rendered (Avg %.4fs per Ep.)' % (len(episode_images), dur / (len(episode_images))))
    if args.save_dir is not None:
        print('Saving to file <{}>'.format(args.save_dir))
        with open(args.save_dir, 'wb') as f:
            pickle.dump(episode_images, f)
    input('>> press any key to continue ...')
    for it, dat in enumerate(episode_images):
        images, stats = dat
        total_len += stats['length']
        total_succ += stats['success']
        print('Episode#%d, Length = %d (Avg len = %.3f)' % (it + 1, stats['length'], total_len/(it+1)))
        print(' >> Target = %s' % (stats['target']))
        print(' >> Success = %d  (Rate = %.3f)' % (stats['success'], total_succ / (it + 1)))
        print(' >> Stay in Room = %d' % stats['good'])
        show_episode(env, images)
        time.sleep(1.5)


def parse_args():
    parser = argparse.ArgumentParser("Visualization for 3D House Navigation")
    # config
    parser.add_argument("file", type=str, help="evaluation stats file")
    parser.add_argument("--env-set", choices=['small', 'train', 'test'], default='test',
                        help="the set of houses. default <test>")
    parser.add_argument("--max-iters", type=int, default=500,
                        help="at most display this number of episodes")
    parser.add_argument("--max-episode-len", type=int, default=2000,
                        help="only display episode with length smaller than this number")
    parser.add_argument("--only-success", action='store_true', default=False,
                        help='Only display those successful runs')
    parser.add_argument("--only-good", action='store_true', default=False,
                        help='Only display those runs where agent reaches the target room')
    parser.add_argument("--save-dir", type=str,
                        help='Set when we need to store all the frames into a file')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    assert os.path.exists(args.file), 'Stats File Not Exists!'

    with open(args.file, 'rb') as f:
        [stats, config] = pickle.load(f)

    visualize(args, stats, config)
