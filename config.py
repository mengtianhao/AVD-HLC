# -*- coding: utf-8 -*-
'''
@time: 2025/4/16

@ author:
'''

class Config:

    seed = 20

    # path
    datafolder = '../data/ptbxl/'
    # datafolder = '../data/SPH/'

    experiment = 'exp1'

    # train model name:  MyBackbone, resnet1d_wang, mobilenetv3_small, ...
    model_name = 'MyBackbone'

    batch_size = 64

    max_epoch = 100

    lr = 0.001

    device_num = 0

    checkpoints = 'MyBackbone_exp1_checkpoint_best.pth'

config = Config()
