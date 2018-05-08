from headers import *
import numpy as np
import common
import utils

import os, sys, time, json
np.random.seed(0)
env = common.create_env()
args = common.create_default_args(algo='nop')
#policy = common.create_policy((1, 1), (4, 2), name='random')
trainer = common.create_trainer('nop', 'cnn', args)
policy = trainer.policy
obs = env.reset()
frames = 10000
print('Start Testing ...')
elap = time.time()
for i in range(frames):
    #obs = np.array(env.api.render(), copy=False)
    idx = trainer.process_observation(obs)
    if flag_use_pytorch:
        """
        x = [Variable(torch.rand(1,4)).type(FloatTensor), Variable(torch.rand(1,2)).type(FloatTensor)]
        batched_actions = x #[F.softmax(v) for v in x]
        #batched_actions = policy(x)
        if use_cuda:
            actions = [a.cpu() for a in batched_actions]
        else:
            actions = batched_actions
        action = [a[0].data.numpy() for a in actions]
        """
        #action = trainer.action()
        """
        s = Variable(torch.zeros(1,1), volatile=True).type(FloatTensor)
        x = [Variable(torch.rand(s.size(0),4)).type(FloatTensor), Variable(torch.rand(s.size(0),2)).type(FloatTensor)]
        #x = [Variable(torch.rand(s.size(0),4)), Variable(torch.rand(s.size(0),2))]
        y = [F.softmax(v) for v in x]
        #y = policy(Variable(torch.zeros(1,1), volatile=True).type(FloatTensor))
        action = [z.cpu().data.numpy()[0] for z in y]
        """
        action = trainer.action()
    else:
        action = [rand_act(4), rand_act(2)]
    obs, rew, done, info = env.step(action)
    trainer.process_experience(idx, action, rew, done, False)
    if (done) or ((i + 1) % 50 == 0):
        obs = env.reset()
    if (i+1) % 500 == 0:
        print(' --> Finished %d, elapsed = %.2fs' % (i+1, time.time()-elap))
print('Time Elapsed for %d frames = %.5fs' % (frames, time.time()-elap))
