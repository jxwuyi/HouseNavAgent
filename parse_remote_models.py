import os, sys, json, time

prefix = '/mnt/vol/gfsai-bistro-east/ai-group/bistro/gpu/yiw/'
#dates = ["20180511"]
dates = ['20180510', '20180511']

all_repos = []

for date in dates:
    repo = prefix + date
    lis = [(d, os.path.join(repo, d)) for d in os.listdir(repo) if os.path.isdir(os.path.join(repo, d))]
    all_repos += lis


for d in all_repos:
    print(d)




def parse_key(repo):
    name = repo.split('.')[0]
    term = name.split('_')
    signal = 'visual' if any([t == 'visual' for t in term]) else 'seg'
    if any([t == 'mask' for t in term]):
        signal = signal + '_mask'
    if term[0] == 'joint':
        target = 'any-object'
    else:
        pos_only = [i for i, t in enumerate(term) if t == 'only'][-1]
        target = '_'.join(term[pos_only+1:])
    #birth = [t for t in term if t[:4] == 'step'][0]
    if target != 'any-object':
        return signal+','+target
    else:
        # flag of curriculum
        curr = "c" + term[-1].replace(',', '-')
        return signal+','+target+','+curr

save_dir = 'remote_job_dirs.json'
D = dict()
for name, repo in all_repos:
    key = parse_key(name)
    D[key] = repo
    print('{} --> <{}>'.format(key, repo))

with open(save_dir,'w') as f:
    json.dump(D,f)

