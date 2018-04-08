# HouseNavAgent
Navigation agent in the 3D house environment

# PyTorch Version
Currently only support pytorch 0.1 and 0.2. It will raise run-time error in pytorch 0.3.

# Note and Usage
0. These packages are required: `pytorch, gym, matplotlib, opencv, msgpack, msgpack_numpy`
1. Ensure the path is properly set in `config.json` (see `config.json.example`)
2. See `common.py` for configuring environements
    1. all the house IDs are stored in variable `all_house_IDs`
    2. ensure the path to config files are set correctedly (i.e., `prefix`, `csvFile`, `colorFile`)
    3. see `create_env(k, linearReward, hardness, segment_input)` for creating an instance of environment
        * `k`: indicates the id of house, when `k` < 0, we use a multi-house environment with the first `|k|` houses
        * `segment_input`: 'none' for only using pixel input; otherwise please use 'joint'
3. See `scripts` folder for scripts. For training multi-house policy, please refer to this [script](https://github.com/jxwuyi/HouseNavAgent/blob/master/scripts/script_multihouse_segjoint_20house.sh).
4. For training options, please refer to `train.py`.
5. For evaluation, please refer to 'eval.py'. The evaluation script for multi-house policy is [here](https://github.com/jxwuyi/HouseNavAgent/blob/master/scripts/script_eval_20h_segjoint.sh).
6. For visulizing the evaluation process, please refer to `visualize.py`.
