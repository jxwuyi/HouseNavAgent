import os, sys, json, time

all_rooms = ['outdoor', 'kitchen', 'living_room', 'dining_room', 'bedroom',
             'bathroom', 'office', 'garage']

other = ['all', 'any-object']

obs_signals = ['visual', 'seg']

print('Loading Config ...')
config_file = 'remote_job_dirs.json'
with open(config_file, 'r') as f:
    D = json.load(f)
print(' >> Done!')

all_targets = all_rooms + other

for key in D.keys():
    print('Processing <{}>...'.format(key))
    p = key.split(',')
    if len(p)==2:
        p.append('birth15')
    prefix = './_model_/nips_HRL/' + p[0] + '_large_' + p[2] + '/' + p[1]
    print('  --> prefix = {}'.format(prefix))
    repo = D[key]
    print('  --> Copying ...')
    cmd = 'cp {}/_model_/* {}/'.format(repo, prefix)
    os.system(cmd)
    cmd = 'cp {}/log/* {}/'.format(repo, prefix)
    os.system(cmd)
    cmd = 'cp {}/*.log {}/'.format(repo, prefix)
    os.system(cmd)
    print(' >> Done!')


for obs_sig in obs_signals:
    prefix = './_model_/nips_HRL/' + obs_sig + '_large'
