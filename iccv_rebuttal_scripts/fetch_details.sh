#!/bin/bash

python3 fetch_eval_details.py --data /home/jxwuyi/workspace/HouseNavAgent/results/iccv/oracle_fix/g_1000_m_30_term_mask_sd7/mixture_full_eval_history.pkl \
    --log-dir ./ --filename _eval_details_iccv_main.pkl --render-gpu 2

python3 fetch_eval_details.py --data /home/jxwuyi/workspace/HouseNavAgent/results/iccv/additional/oracle/g_1000_m_30_term_mask_sd7000/mixture_full_eval_history.pkl \
    --log-dir ./ --filename _eval_details_iccv_extra.pkl --render-gpu 2 --plan-dist-iters "4:253,5:436"

