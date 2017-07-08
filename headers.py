import os, sys

time_counter = [0,0,0,0]


if "Apple" in sys.version:
    # own mac PC
    path_to_python_repo = '/Users/yiw/workroom/objrender/python'
elif "Red Hat" in sys.version:
    path_to_python_repo = '/home/yiw/code/objrender/python'
else:
    assert(False)
sys.path.insert(0, path_to_python_repo)


import torch
from torch.autograd import Variable

use_cuda = torch.cuda.is_available()
if use_cuda:
    print('>>> CUDA used!!!')
FloatTensor = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor
IntTensor = torch.cuda.IntTensor if use_cuda else torch.IntTensor
LongTensor = torch.cuda.LongTensor if use_cuda else torch.LongTensor
ByteTensor = torch.cuda.ByteTensor if use_cuda else torch.ByteTensor
Tensor = FloatTensor

# define AgentTrainer Template
class AgentTrainer(object):
    def __init__():
        pass

    def reset_agent(self):
        pass

    def action(self, obs):
        raise NotImplemented()

    def process_experience(self, obs, act, rew, new_obs, done, terminal, info):
        raise NotImplemented()

    def preupdate(self):
        raise NotImplemented()

    def update(self, agents):
        raise NotImplemented()

    def _process_frames(self, raw_frames, volatile=False, merge_dim=True, return_variable=True):
        """
        frames: (batch_size, len, n, m, channel_n) in numpy
        output:
        >> merge_dim=True: (batch_size, len * channel_n, n, m), processed as FloatTensor
           merge_dim=False:(batch_size, len, channel_n, n, m), processed as FloatTensor
        """
        batch_size = raw_frames.shape[0]
        img_h, img_w = raw_frames.shape[2], raw_frames.shape[3]
        chn = raw_frames.shape[1] * raw_frames.shape[4]
        if return_variable:
            frames = Variable(torch.from_numpy(raw_frames), volatile=volatile)
        else:
            frames = torch.from_numpy(raw_frames)
        frames = frames.type(ByteTensor).permute(0, 1, 4, 2, 3)
        if merge_dim: frames = frames.resize(batch_size, chn, img_h, img_w)
        return (frames.type(FloatTensor) - 128.0) / 128.0

    def eval(self):
        self.policy.eval()

    def train(self):
        self.policy.train()

    def save(self, save_dir, version=""):
        if len(version) > 0:
            version = "_" + version
        if save_dir[-1] != '/':
            save_dir += '/'
        filename = save_dir + self.name + version + '.pkl'
        torch.save(self.policy.state_dict(), filename)

    def load(self, save_dir, version=""):
        if os.path.isfile(save_dir) or (version is None):
            filename = save_dir
        else:
            if len(version) > 0:
                version = "_" + version
            if save_dir[-1] != '/':
                save_dir += '/'
                filename = save_dir + self.name + version + '.pkl'
        self.policy.load_state_dict(torch.load(filename))
