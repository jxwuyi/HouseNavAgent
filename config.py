#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: config.py
# Author: Yuxin Wu <ppwwyyxxc@gmail.com>

import os
import json
from functools import lru_cache

@lru_cache()
def get_config(fname=None):
    FNAME = 'config.json'
    def get_fname():
        if os.path.isfile(FNAME):
            return FNAME
        fname = os.path.join(os.path.dirname(__file__), FNAME)
        if os.path.isfile(fname):
            return fname
        raise RuntimeError("Cannot find config.json file!")
    if fname is None:
        fname = get_fname()

    with open(fname) as f:
        obj = json.load(f)
        return obj

@lru_cache()
def get_house_ids(fname=None):
    FNAME = 'all_house_ids.json'
    def get_fname(FNAME):
        if os.path.isfile(FNAME):
            return FNAME
        fname = os.path.join(os.path.dirname(__file__), FNAME)
        if os.path.isfile(fname):
            return fname
        raise RuntimeError("Cannot find all_house_idsjson file!")

    if fname is None:
        fname = get_fname()

    with open(fname) as f:
        obj = json.load(f)
        return obj
