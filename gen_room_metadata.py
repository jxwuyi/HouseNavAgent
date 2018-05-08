from headers import *
import numpy as np
import common
import utils

import os, sys, time, json


env_set_name = 'train'
n_houses = 200

ts = time.time()
common.set_house_IDs(env_set_name)
common.ensure_object_targets()

task = common.create_env(-n_houses, task_name='roomnav',
                        success_measure='see',
                        cacheAllTarget=multi_target,
                        render_device=render_device,
                        use_discrete_action=('dpg' not in algo),
                        include_object_target=True,
                        include_outdoor_target=True,
                        discrete_angle=True)
all_houses = task.env.all_houses

meta_data = [h.all_desired_targetTypes for h in all_houses]

with open(env_set_name+'_set_metadata.pkl', 'wb') as f:
    pickle.dump(meta_data, f)

dur = time.time() - ts
print('Done!  Time Elapsed for %d houses = %.5fs' % (n_houses, dur))
