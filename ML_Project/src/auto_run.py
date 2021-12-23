# -*- coding: utf-8 -*-
"""
Created on Wed Apr 14 06:03:56 2021

@author: kaibu
"""

import subprocess

program_list = [
                # 'python train_GCN.py --optimize ndcg',
                # 'python train_GCN.py --optimize ndcg --data NYC',
                 'python train_GCN.py --optimize ndcg --data Chi'
                # 'python train_GCN.py --optimize ndcg di',
                # 'python train_GCN.py --optimize ndcg di xndcg',
                # 'python train_GCN.py --optimize ndcg di efpr',
               ]


for program in program_list:
    subprocess.run(program, shell=True)
    print("Finished:" + program)
    print()