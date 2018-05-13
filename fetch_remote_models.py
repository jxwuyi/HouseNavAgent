import os, sys, json, time

#save_dir = "./_model_/nips_tune/"  # "./_model_/nips_HRL/"
#config_file = 'remote_job_dirs.json'

save_dir = "./_model_/nips_tune_old/"  # "./_model_/nips_HRL/"
config_file = 'remote_job_dirs_old.json'

print('Loading Config ...')
with open(config_file, 'r') as f:
    D = json.load(f)
print(' >> Done!')

for key in D.keys():
    print('Processing <{}>...'.format(key))
    p = key.split(',')
    if len(p)==2:
        p.append('step15')
    prefix = save_dir + p[0] + '_large_' + p[2]
    if not os.path.exists(prefix):
        print('  --> Creating Repo <{}>...'.format(prefix))
        os.makedirs(prefix)
    prefix = prefix + '/' + p[1]
    if not os.path.exists(prefix):
        print('  --> Creating Repo <{}>...'.format(prefix))
        os.makedirs(prefix)
    else:
        # clear the repo
        print('  --> Clear files under repo <{}>'.format(prefix))
        os.system("rm {}/*".format(prefix))
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

