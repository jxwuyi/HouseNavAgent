import sys

if "Apple" in sys.version:
    # own mac PC
    path_to_python_repo = '/Users/yiw/workroom/objrender/python'
elif "Red Hat" in sys.version:
    path_to_python_repo = '/home/yiw/code/objrender/python'
else:
    assert(False)
sys.path.insert(0, path_to_python_repo)


import torch
use_cuda = torch.cuda.is_available()
if use_cuda:
    print('>>> CUDA used!!!')
FloatTensor = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if use_cuda else torch.LongTensor
ByteTensor = torch.cuda.ByteTensor if use_cuda else torch.ByteTensor
Tensor = FloatTensor

# define AgentTrainer Template
class AgentTrainer(object):
    def __init__(self, name, policy, obs_shape, act_shape, args):
        raise NotImplemented()

    def action(self, obs):
        raise NotImplemented()

    def process_experience(self, obs, act, rew, new_obs, done, terminal):
        raise NotImplemented()

    def preupdate(self):
        raise NotImplemented()

    def update(self, agents):
        raise NotImplemented()

    def save(self, filename):
        torch.save(self.policy.state_dict(), filename)

    def load(self, filename):
        self.policy.load_state_dict(torch.load(filename))
