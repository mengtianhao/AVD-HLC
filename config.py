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

    # exp1, exp1.1, exp1.1.1, exp2, exp3

    experiment = 'exp1'

    # train model name:  MyView, resnet1d_wang, inceptiontime, fcn_wang, lstm, mobilenetv3_small, ...
    model_name = 'MyView'

    batch_size = 32

    max_epoch = 100

    lr = 0.001

    device_num = 0

    checkpoints = 'MyView_exp0_checkpoint_best.pth'

config = Config()
