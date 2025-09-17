import torch

parent_label = ['CD', 'HYP', 'MI', 'NORM', 'STTC']
sub_label = ['AMI', 'CLBBB', 'CRBBB', 'ILBBB', 'IMI', 'IRBBB', 'ISCA', 'ISCI', 'ISC_', 'IVCD',
             'LAFB/LPFB', 'LAO/LAE', 'LMI', 'LVH', 'NORM', 'NST_', 'PMI', 'RAO/RAE', 'RVH',
             'SEHYP', 'STTC', 'WPW', '_AVB']

# 映射：子标签 → 父标签名
sub_to_parent = {
    'AMI': 'MI',
    'CLBBB': 'CD',
    'CRBBB': 'CD',
    'ILBBB': 'CD',
    'IMI': 'MI',
    'IRBBB': 'CD',
    'ISCA': 'STTC',
    'ISCI': 'STTC',
    'ISC_': 'STTC',
    'IVCD': 'CD',
    'LAFB/LPFB': 'CD',
    'LAO/LAE': 'HYP',
    'LMI': 'MI',
    'LVH': 'HYP',
    'NORM': 'NORM',
    'NST_': 'STTC',
    'PMI': 'MI',
    'RAO/RAE': 'HYP',
    'RVH': 'HYP',
    'SEHYP': 'HYP',
    'STTC': 'STTC',
    'WPW': 'CD',
    '_AVB': 'CD'
}

# 初始化 0 矩阵
E = torch.zeros(len(sub_label), len(parent_label))

# 填充 one-hot
for i, sub in enumerate(sub_label):
    parent = sub_to_parent[sub]
    j = parent_label.index(parent)
    E[i][j] = 1.0

print(E)  # E is (23, 5)
