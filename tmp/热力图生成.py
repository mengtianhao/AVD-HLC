import pickle

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.collections as mcoll
from scipy import ndimage


import torch, time, os
import models, utils
from torch import nn, optim
from dataset_label import load_datasets
from config import config
from sklearn.metrics import roc_auc_score
import numpy as np
import random
import pandas as pd
from utils import find_best_thresholds, find_best_precision, compute_mAP
import warnings

# 检查可用的样式
print("可用的样式:", plt.style.available)

# 使用seaborn风格的替代名称（根据上面打印的结果选择）
# 常见替代名称有：'seaborn-v0_8', 'seaborn', 'seaborn-poster', 'seaborn-whitegrid'等
try:
    plt.style.use('seaborn-v0_8')  # 新版本Matplotlib的seaborn样式名称
except:
    plt.style.use('ggplot')  # 如果seaborn不可用，使用ggplot作为替代

def multicolored_lines(x, y, heatmap, title_name, lead_name):
    fig, ax = plt.subplots(figsize=(12, 4))
    lc = colorline(x, y, heatmap, cmap='rainbow')
    plt.colorbar(lc)
    lc.set_linewidth(2)
    lc.set_alpha(0.8)
    plt.xlim(x.min(), x.max())
    plt.ylim(y.min(), y.max())
    plt.title(f"{title_name} - 导联 {lead_name}")
    plt.grid(False)
    plt.show()


def colorline(x, y, heatmap, cmap='rainbow'):
    z = np.array(heatmap)
    points = np.array([x, y]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    lc = mcoll.LineCollection(segments, array=z, cmap=cmap)
    ax = plt.gca()
    ax.add_collection(lc)
    return lc


class MultiTaskGradCAMpp:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        self._register_hooks()

    def _register_hooks(self):
        def forward_hook(module, input, output):
            self.activations = output.detach()

        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0].detach()

        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_backward_hook(backward_hook)

    def generate_cam(self, inputs, target_p=None, target=None):
        """ 适配多任务模型的Grad-CAM++ """
        # 前向传播 (关闭Train模式)
        outputs = self.model(inputs, target_p, target, Train=False, device=inputs.device)
        main_output = outputs[0]  # 主分类输出

        # 反向传播 (仅针对主输出)
        self.model.zero_grad()
        one_hot = torch.zeros_like(main_output)
        one_hot[0, torch.argmax(main_output)] = 1
        main_output.backward(gradient=one_hot, retain_graph=True)

        # Grad-CAM++计算
        activations = self.activations  # [1, C, T]
        gradients = self.gradients  # [1, C, T]

        # 改进的权重计算
        global_sum = activations.sum(dim=(0, 2), keepdim=True)
        alpha_num = gradients.pow(2)
        alpha_denom = alpha_num * 2 + global_sum * gradients.pow(3)
        alpha_denom = torch.where(alpha_denom != 0.0, alpha_denom, torch.ones_like(alpha_denom))
        alphas = alpha_num / alpha_denom

        weights = F.relu(gradients) * alphas
        weights = weights.sum(dim=(0, 2), keepdim=True)

        # 生成热力图
        cam = (weights * activations).sum(dim=1)  # [1, T]
        cam = F.relu(cam)
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-10)

        return cam.squeeze().cpu().numpy(), main_output.argmax().item()


def visualize_gradcam_plusplus(model1, ecgdata0, target_layer_name, N=1000):
    """
    model1: 第一个PyTorch模型(CNN)
    model2: 第二个PyTorch模型(Transformer)
    ecg_data: 输入的心电数据，形状为(12, 1000)
    target_layer_name: model2中的目标层名称(如'transformer.layers.5.1.fn.fn.net.3')
    N: 信号总时长(秒)
    """
    # 将心电数据转换为tensor
    device = next(model1.parameters()).device
    ecg_data = ecgdata0[0]
    target_p = ecg_data[1]
    target = ecgdata0[2]
    input_tensor = torch.FloatTensor(ecg_data).unsqueeze(0).to(device)  # 添加batch维度

    # 通过model1处理
    with torch.no_grad():
        # logits,intermediate_output = model1(input_tensor)#torch.Size([1, 128, 64])
        output, private_preds, common_preds, view_labels, diff_loss, common_label_preds, parent_out = model(ecg_data,target_p,target,Train=True,device=device)

    # 在model2中找到目标层
    target_layer = None
    for name, module in model1.named_modules():
        if name == target_layer_name:
            target_layer = module
            break

    if target_layer is None:
        raise ValueError(f"在model2中未找到目标层 {target_layer_name}")

    # 初始化Grad-CAM++
    grad_cam = GradCAMpp(model1, target_layer)

    # 生成CAM
    try:
        # 注意: 这里我们处理整个中间输出，而不是单个导联
        cam, outputs = grad_cam.generate_cam(intermediate_output)

        # 调整热力图大小
        heatmap = ndimage.zoom(cam, (1000 / cam.shape[0],))

        # 绘制每个导联
        for lead_idx in range(12):
            x = np.linspace(0, N, 1000)
            y = ecg_data[lead_idx]  # 原始ECG数据
            multicolored_lines(x, y, heatmap, "Grad-CAM++", lead_idx + 1)

    except Exception as e:
        print(f"生成CAM时出错: {str(e)}")

    return outputs


class ECG_GradCAMpp:
    def __init__(self, model):
        self.model = model
        # 注册目标层（所有MyNet的mlla_block输出）
        # self.target_layers = {
        #     f'MyNet{i}': getattr(model, f'MyNet{i}').mlla_block.out_proj
        #     for i in range(1, 7)
        # }
        # 选择layer2的最后一个卷积层作为目标层
        self.target_layers = {
            f'MyNet{i}': getattr(model, f'MyNet{i}').layer2.conv3
            for i in range(1, 7)
        }
        # self.target_layers = {
        #     'MyNet1': model.MyNet1.layer2.conv3,
        #     'MyNet2': model.MyNet2.layer2.conv3,
        #     'MyNet3': model.MyNet3.layer2.conv3,
        #     'MyNet4': model.MyNet4.layer2.conv3,
        #     'MyNet5': model.MyNet5.layer2.conv3,
        #     'MyNet6': model.MyNet6.layer2.conv3
        # }
        self.activations = {}
        self.gradients = {}
        self._register_hooks()

    def _register_hooks(self):
        def forward_hook(net_name):
            def hook(module, input, output):
                self.activations[net_name] = output.detach()

            return hook

        def backward_hook(net_name):
            def hook(module, grad_input, grad_output):
                self.gradients[net_name] = grad_output[0].detach()

            return hook

        for name, layer in self.target_layers.items():
            layer.register_forward_hook(forward_hook(name))
            layer.register_backward_hook(backward_hook(name))

    def generate_cams(self, x, device):
        """
        输入:
          x: ECG信号 [batch, 12, 1000]
          device: 计算设备
        """
        # 确保模型处于eval模式
        self.model.eval()

        # 准备输入（与您的训练代码兼容）
        target_p = None  # 父类标签（Grad-CAM不需要）
        target = None  # 无需真实标签

        # 前向传播（关闭Train模式）
        with torch.set_grad_enabled(True):
            outputs = self.model(x, target_p, target, Train=False, device=device)
            main_output = outputs[0]  # 主分类输出
            # print("main_output::", main_output)
            # 修复维度错误：更健壮的预测类别获取
            if main_output.dim() == 1:  # 处理单维度输出
                pred_class = main_output.argmax(dim=0).item()
                one_hot = torch.zeros_like(main_output)
                one_hot[pred_class] = 1
            else:  # 处理常规[batch, classes]输出
                pred_class = main_output.argmax(dim=1).item()
                one_hot = torch.zeros_like(main_output)
                one_hot[0, pred_class] = 1

            # 反向传播（仅针对主输出）
            self.model.zero_grad()
            main_output.backward(gradient=one_hot, retain_graph=True)

            # 为每个分支生成CAM
            cams = {}
            for net_name in self.target_layers.keys():
                activations = self.activations[net_name]  # [1, 128, 244]
                gradients = self.gradients[net_name]  # [1, 128, 244]

                # Grad-CAM++计算
                weights = F.relu(gradients)

                # 修正维度：沿通道和时间维度平均
                weights = weights.mean(dim=(0, 2), keepdim=True)  # [1, 1, 1]

                # 生成热力图
                cam = (weights * activations).sum(dim=1)  # [1, 244]
                cam = F.relu(cam)

                # 归一化处理
                # cam_min = cam.min()
                # cam_max = cam.max()
                # if cam_max - cam_min > 1e-5:
                #     cam = (cam - cam_min) / (cam_max - cam_min)
                # else:
                #     cam = torch.zeros_like(cam)  # 避免除零错误

                # 修复插值维度错误
                # 正确方式：将cam转换为3D张量 [batch=1, channels=1, time]
                cam = cam.unsqueeze(0)  # [1, 1, 244]
                print("cam_shape:",cam.shape)
                print("x_shape:",x.shape)
                # 上采样到原始长度
                cam = F.interpolate(
                    cam,
                    size=x.shape[-1],  # 自动适配输入长度 (1000)
                    mode='linear',
                    align_corners=False
                )
                cams[net_name] = cam[0, 0].cpu().numpy()  # 提取[1000]数组

            return cams, pred_class


def visualize_ecg_with_cams(ecg_signal, cams, pred_class, fs=500):
    """ 12导联可视化（临床标准视图） """
    lead_names = ['I', 'II', 'III', 'aVR', 'aVL', 'aVF',
                  'V1', 'V2', 'V3', 'V4', 'V5', 'V6']
    time_axis = np.arange(ecg_signal.shape[1]) / fs

    # 导联到MyNet的映射
    lead_to_net = {
        # aVR
        3: 'MyNet1',
        # I + aVL
        0: 'MyNet2', 4: 'MyNet2',
        # V1-V2
        6: 'MyNet3', 7: 'MyNet3',
        # V3-V4
        8: 'MyNet4', 9: 'MyNet4',
        # V5-V6
        10: 'MyNet5', 11: 'MyNet5',
        # II + III + aVF
        1: 'MyNet6', 2: 'MyNet6', 5: 'MyNet6'
    }

    fig, axes = plt.subplots(12, 1, figsize=(15, 18), sharex=True)

    for lead_idx in range(12):
        net_name = lead_to_net.get(lead_idx)
        cam = cams[net_name] if net_name else np.zeros(1000)
        signal = ecg_signal[lead_idx]

        # 绘制ECG信号
        axes[lead_idx].plot(time_axis, signal, 'k', linewidth=1.5, alpha=0.9)

        # 绘制热力图（红色高亮重要区域）
        axes[lead_idx].fill_between(time_axis, signal.min(), signal.max(),
                                    where=cam > 0.3, color='r', alpha=0.2,
                                    transform=axes[lead_idx].get_xaxis_transform())

        # 添加导联标签
        axes[lead_idx].set_ylabel(lead_names[lead_idx], rotation=0, ha='right', va='center')
        axes[lead_idx].grid(True, linestyle='--', alpha=0.6)

    plt.suptitle(f"12-Lead ECG with Grad-CAM++\nPredicted Class: {pred_class}", fontsize=16)
    plt.xlabel("Time (s)")
    plt.tight_layout()
    plt.subplots_adjust(top=0.95, hspace=0.1)
    plt.show()




if __name__=="__main__":
    config.datafolder = 'data/ptbxl/'
    config.experiment = 'exp1.1'
    config.hierarchical = True
    config.seed = 20
    # config.model_name = 'MyNet6View_views_attention'
    config.model_name = 'MyNet6View_views_label'
    config.batch_size = 64
    #获得模型
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # datasets
    train_dataloader, val_dataloader, test_dataloader, num_classes_p, num_classes = load_datasets(
        datafolder=config.datafolder,
        experiment=config.experiment,
        hierarchical=config.hierarchical
    )


    # mode
    model = getattr(models, config.model_name)(num_classes=num_classes, num_classes_p=num_classes_p)
    print('model_name:{}, num_classes={}, parent_num_classes={}'.format(config.model_name, num_classes, num_classes_p))
    model = model.to(device)

    chkpoint = torch.load("./checkpoints/MyNet6View_views_label_exp1.1_checkpoint_best.pth", map_location=device)
    model.load_state_dict(chkpoint["model_state_dict"])
    model.eval()

    # for name, param in model.named_parameters():
    #     print(f"Parameter name: {name},{param.requires_grad}")
    # print()

    # 使用示例
    # ecg_data = np.random.randn(12, 1000)  # 替换为实际心电数据


    # test_dataset = MyDataset_Test(config)
    #42,50,
    # ecg,label=test_dataset[50]
    # ecg_data=ecg.cpu().detach().numpy()
    # print(label)

    # with open('../getdata/data/CPSC/500HZ.npy', 'rb') as f:
    #     data = pickle.load(f)  # 全部的样例,是个张量
    #
    # ecg_data=data[100].cpu().detach().numpy()

    # 初始化Grad-CAM++
    gradcam = ECG_GradCAMpp(model)

    # 获取测试集的第一批数据（包含多个样本）
    test_batch = next(iter(test_dataloader))

    # 如果是元组形式的数据（如ECG信号和标签），取第一个样本
    ecg_input= test_batch[0][0].unsqueeze(0).to(device)  # 信号数据
    print(ecg_input.shape)
    # 生成热力图（与您的训练接口兼容）
    cams, pred_class = gradcam.generate_cams(ecg_input, device)

    # 可视化结果
    ecg_signal = ecg_input.squeeze(0).cpu().numpy()  # 转换为numpy [12, 1000]
    visualize_ecg_with_cams(ecg_signal, cams, pred_class)

    # outputs = visualize_gradcam_plusplus(
    #     model1=model,
    #     ecgdata0=ecgdata0,
    #     # target_layer_name='transformer.layers.5.1.fn.fn.net.3',
    #     target_layer_name='fc',
    #     # target_layer_name='logits',
    #     N=1000  # 总时长(秒)
    # )
