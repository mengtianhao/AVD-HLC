from dataset import DownLoadECGData, ECGDataset, load_datasets
import numpy as np
import pandas as pd


if __name__ == '__main__':
    datafolder = 'data/SPH/'
    # experiment = 'exp1'
    experiment = 'exp1.1'
    # experiment = 'exp1.1.1'
    # datasets
    train, val, test, num_classes = load_datasets(
        datafolder=datafolder,
        experiment=experiment,
    )
    print('当前分类的标签总数：', end='')
    print(num_classes)
    #
    label_list = []
    if experiment == 'exp1.1':
        label_list = ['A', 'C', 'D', 'E', 'F', 'H', 'I', 'J', 'K', 'L', 'M']
    label_counts = np.zeros(len(label_list)).astype(int)
    # 统计训练集信息
    # print('训练集数量：', end='')
    # print(len(train))
    # for inputs, labels in train:
    #     # print(inputs.shape)
    #     # print(labels.shape)
    #     tmp_labels = labels.cpu().numpy().astype(int)
    #     label_counts = label_counts + tmp_labels
    # print(label_counts)
    # 统计验证集信息
    # print('验证集数量：', end='')
    # print(len(val))
    # for inputs, labels in val:
    #     # print(inputs.shape)
    #     # print(labels.shape)
    #     tmp_labels = labels.cpu().numpy().astype(int)
    #     label_counts = label_counts + tmp_labels
    # print(label_counts)
    # 统计测试集信息
    print('测试集数量：', end='')
    print(len(test))
    for inputs, labels in test:
        # print(inputs.shape)
        # print(labels.shape)
        tmp_labels = labels.cpu().numpy().astype(int)
        label_counts = label_counts + tmp_labels
    print(label_counts)
