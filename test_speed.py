from headers import *
import numpy as np
import common
import utils

import os, sys, time, json

flag_use_pytorch = True


def rand_act(d):
    a = np.random.rand(d)
    a /= np.sum(a)
    return a


np.random.seed(0)
env = common.create_env()
args = common.create_default_args(algo='nop')
trainer = common.create_trainer('nop','random',args)
obs = env.reset()
frames = 10000
print('Start Testing ...')
elap = time.time()
for i in range(frames):
    #obs = np.array(env.api.render(), copy=False)
    if flag_use_pytorch:
        action = trainer.action(obs)
    else:
        action = [rand_act(4), rand_act(2)]
    obs, _, _, _ = env.step(action)
    if (i+1) % 500 == 0:
        print(' --> Finished %d, elapsed = %.2fs' % (i+1, time.time()-elap))
print('Time Elapsed for %d frames = %.5fs' % (frames, time.time()-elap))
