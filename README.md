# Bayesian Relational Memory for Semantic Visual Navigation (ICCV2019)
This is the source code for our ICCV2019 paper, which implements a visual navigation agent with a Bayesian relational memory over semantic concepts in the [House3D](https://github.com/facebookresearch/House3D) environment.

**_check out our paper [here](https://people.eecs.berkeley.edu/~russell/papers/iccv19-brm.pdf)._**

Bibtex:
```
@inproceedings{wu2019bayesian,
  title={Bayesian Relational Memory for Semantic Visual Navigation},
  author={Wu, Yi and Wu, Yuxin and Tamar, Aviv and Russell, Stuart and Gkioxari, Georgia and Tian, Yuandong},
  booktitle={Proceedings of the 2019 IEEE International Conference on Computer Vision (ICCV)},
  year={2019}
}
```

## PyTorch Version
The required PyTorch version is 0.3.1.

Note: the policies where trained under PyTorch 0.2.0. In order to guarantee the full reproducibility of our results, please switch to PyTorch 0.2.0 (with very little API changes).

It will raise run-time error in pytorch 0.3.0. Make sure to avoid this version! The code will be kept as it is now and no further package upgrade will be performed.

## Environment
For task and environment details, please follow the original [House3D paper](https://arxiv.org/abs/1801.02209).

Our project relies on a customized, [C++ re-implementation](https://github.com/jxwuyi/House3D/tree/C++) of the House3D environment, which is much faster, consumes orders of magnitudes less memory, and provides much more APIs task analysis and auxiliary training.

## Preliminary
0. Python version 3.6
1. These packages are required: `numpy, pytorch=0.3.1, gym, matplotlib, opencv, msgpack, msgpack_numpy`
2. Set the House3D path properly by generating your `config.json` file (see `config.json.example`)
3. See `config.py` and ensure metadata files, `all_house_ids.json` (all house ids) and `all_house_targets.json` (semantic target types), are properly set
4. Stay in the root folder and ensure the following two sanity check scripts can be properly run
    a. training test: `python3 release/scripts/sanity_check/test_train.py`
    b. evaluation test: `python3 release/scripts/sanity_check/test_eval.py`

## Usage
All the scripts and trained policies are stored in [release](https://github.com/jxwuyi/HouseNavAgent/blob/master/release) folder. To run scripts, ensure that the scripts are run from the root folder (i.e., under `HouseNavAgent` folder).
1. For evaluation BRM agents and all related baselines, check the [scripts/eval](https://github.com/jxwuyi/HouseNavAgent/blob/master/release/scripts/eval) folder.
2. For re-training the polices, check the [scripts/train](https://github.com/jxwuyi/HouseNavAgent/blob/master/release/scripts/train) folder

## Important Files
1. To create an environment, refer to [`create_env(...)`](https://github.com/jxwuyi/HouseNavAgent/blob/master/common.py#L600) function in [`common.py`](https://github.com/jxwuyi/HouseNavAgent/blob/master/common.py)
2. For evaluation, refer to [`HRL/eval_HRL.py`](https://github.com/jxwuyi/HouseNavAgent/blob/master/HRL/eval_HRL.py). See the [descriptions](https://github.com/jxwuyi/HouseNavAgent/blob/master/HRL/eval_HRL.py#L279) of all the command line arguments.
3. For parallel A2C training, refer to [`zmq_train.py`](https://github.com/jxwuyi/HouseNavAgent/blob/master/zmq_train.py). See the [descriptions](https://github.com/jxwuyi/HouseNavAgent/blob/master/zmq_train.py#L185) of all the command line arguments.
