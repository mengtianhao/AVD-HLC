import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.attention import CoordAtt
from .until_module import GradReverse
from .new_attention import Seq_Transformer
from .attention_config import config
from .MLLA1D import MLLABlock
import os
from E import ptb_5t23


token_mapping = {
    'CD': 'conduction disturbance',
    'HYP': 'hypertrophy',
    'MI': 'myocardial infarction',
    'NORM': 'normal ECG',
    'STTC': 'ST-T change',
    'AMI': 'anteroseptal myocardial infarction',
    'CLBBB': 'complete left Bundle Branch Block',
    'CRBBB': 'complete right Bundle Branch Block',
    'ILBBB': 'incomplete left Bundle Branch Block',
    'IMI': 'inferolateral or Inferior myocardial infarction',
    'IRBBB': 'incomplete Right Bundle Branch Block',
    'ISCA': 'anterior or Anteroseptal or Lateral or Anterolateral Ischemia, ST-T change',
    'ISCI': 'inferior or Inferolateral Ischemia, ST-T change',
    'ISC_': 'ischemic ST-T changes',
    'IVCD': 'intraventricular Conduction disturbance',
    'LAFB/LPFB': 'left Anterior or Posterior Fascicular Block',
    'LAO/LAE': 'left Atrial Overload/Enlargement',
    'LMI': 'lateral Myocardial Infarction',
    'LVH': 'left Ventricular Hypertrophy',
    'NST_': 'non-Specific ST-T change',
    'PMI': 'posterior Myocardial Infarction',
    'RAO/RAE': 'right Atrial Overload/Enlargement',
    'RVH': 'right Ventricular Hypertrophy',
    'SEHYP': 'septal Hypertrophy',
    'WPW': 'wolff-Parkinson-White Syndrome',
    '_AVB': 'first or second or third degree Atrioventricular Block',
    '1AVB': 'first degree Atrioventricular Block',
    '2AVB': 'second degree Atrioventricular Block',
    '3AVB': 'third degree Atrioventricular Block',
    'ALMI': 'anterolateral myocardial infarction',
    'ANEUR': 'ST-T changes compatible with ventricular aneurysm',
    'ASMI': 'anteroseptal myocardial infarction',
    'DIG': 'Digitalis Effect, ST-T change',
    'EL': 'electrolyte abnormality',
    'ILMI': 'inferolateral myocardial infarction',
    'INJAL': 'Injury in Anterolateral Leads',
    'INJAS': 'Injury in Anteroseptal Leads',
    'INJIL': 'Injury in Inferolateral leads',
    'INJIN': 'Injury in Inferior Leads',
    'INJLA': 'Injury in Lateral Leads',
    'IPLMI': 'inferoposterolateral myocardial infarction',
    'IPMI': 'inferoposterior myocardial infarction',
    'ISCAL': 'Ischemia in Anterolateral Leads',
    'ISCAN': 'Ischemia in Anterior Leads',
    'ISCAS': 'Ischemia in Anteroseptal Leads',
    'ISCIL': 'Ischemia in Inferolateral Leads',
    'ISCIN': 'Ischemia in Inferior Leads',
    'ISCLA': 'Ischemia in Lateral Leads',
    'LAFB': 'Left Anterior Fascicular Block',
    'LNGQT': 'Long QT Interval',
    'LPFB': 'Left Posterior Fascicular Block',
    'NDT': 'Non-Diagnostic T Wave Changes',
    '1': 'normal ECG',
    '21': 'Sinus tachycardia',
    '22': 'Sinus bradycardia',
    '23': 'Sinus arrhythmia',
    '30': 'Atrial premature complexes',
    '31': 'Atrial premature complexes, nonconducted',
    '36': 'Junctional premature complexes',
    '37': 'Junctional escape complexes',
    '50': 'Atrial fibrillation',
    '51': 'Atrial flutter',
    '54': 'Junctional tachycardia',
    '60': 'Ventricular premature complexes',
    '80': 'Short PR interval',
    '81': 'AV conduction ratio N:D',
    '82': 'Prolonged PR interval',
    '83': 'Second-degree AV block, Mobitz type I Wenckebach',
    '84': 'Second-degree AV block, Mobitz type II',
    '85': '2:1 AV block',
    '86': 'AV block, varying conduction',
    '87': 'AV block, advanced, high-grade',
    '88': 'AV block, complete, third-degree',
    '101': 'Left anterior fascicular block',
    '102': 'Left posterior fascicular block',
    '104': 'Left bundle-branch block',
    '105': 'Incomplete right bundle-branch block',
    '106': 'Right bundle-branch block',
    '108': 'Ventricular preexcitation',
    '120': 'Right axis deviation',
    '121': 'Left axis deviation',
    '125': 'Low voltage',
    '140': 'Left atrial enlargement',
    '142': 'Left ventricular hypertrophy',
    '143': 'Right ventricular hypertrophy',
    '145': 'ST deviation',
    '146': 'ST deviation with T wave change',
    '147': 'T wave abnormality',
    '148': 'Prolonged QT interval',
    '152': 'TU fusion',
    '153': 'ST-T change due to ventricular hypertrophy',
    '155': 'Early repolarization',
    '160': 'Anterior Myocardial infarction',
    '161': 'Inferior Myocardial infarction',
    '165': 'Anteroseptal Myocardial infarction',
    '166': 'Extensive anterior Myocardial infarction',
    'A': 'normal ECG',
    'C': 'Sinus node rhythms and arrhythmias',
    'D': 'Supraventricular arrhythmias',
    'E': 'Supraventricular tachyarrhythmias',
    'F': 'Ventricular arrhythmias',
    'H': 'Atrioventricular conduction',
    'I': 'Intraventricular and intra-atrial conduction',
    'J': 'Axis and voltage',
    'K': 'Chamber hypertrophy or enlargement',
    'L': 'ST segment, T wave, and U wave',
    'M': 'Myocardial infarction',
}

class MyBackbone(nn.Module):

    def __init__(self, num_classes=5, input_channels=12, single_view=False):
        super(MyBackbone, self).__init__()

        self.single_view = single_view

        self.conv1 = nn.Conv1d(input_channels, 64, kernel_size=25, stride=1, padding=0, bias=False)
        self.bn1 = nn.BatchNorm1d(64)
        self.relu = Mish()

        self.layer1 = Res2Block(inplanes=64, planes=128, kernel_size=15, stride=2, atten=True)

        self.layer2 = Res2Block(inplanes=128, planes=128, kernel_size=15, stride=2, atten=True)

        self.avgpool = nn.AdaptiveAvgPool1d(1)

        # GRU
        self.gru = nn.GRU(input_size=128, hidden_size=128, num_layers=1, batch_first=True, bidirectional=False)
        #
        self.mlla_block = MLLABlock(dim=128, input_resolution=244)

        if not self.single_view:
            self.fc = nn.Linear(128, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):

        # print(x.shape)  # [64, 1, 1000] or [64, 2, 1000] or [64, 3, 1000]
        output = self.conv1(x)  # [64, 64, 976]
        output = self.bn1(output)
        output = self.relu(output)

        output = self.layer1(output)  # [64, 128, 488]
        # print(output.shape)

        output = self.layer2(output)  # [64, 128, 244]
        # print(output.shape)

        # GRU
        # output = output.permute(0, 2, 1)  # (batch_size, time_steps, features)
        # output, _ = self.gru(output)  # (batch, seq_len, 2 * hidden_size), (64, 244, 256)
        # output = output.permute(0, 2, 1)  # (64, 256, 244)
        # print(output.shape)

        #
        output = output.permute(0, 2, 1)
        output = self.mlla_block(output)
        output = output.permute(0, 2, 1)
        # print(output.shape)  # [64, 128, 244]
        hidden_state = output
        # print(hidden_state.shape)

        output = self.avgpool(output)  # [64, 128, 1]
        # print(output.shape)

        output = output.view(output.size(0), -1)

        if not self.single_view:
            output = self.fc(output)

        return output, hidden_state


class AdaptiveWeight(nn.Module):
    def __init__(self, plances=32):
        super(AdaptiveWeight, self).__init__()

        self.fc = nn.Linear(plances, 1)
        # self.bn = nn.BatchNorm1d(1)
        self.sig = nn.Sigmoid()

    def forward(self, x):
        out = self.fc(x)
        # out = self.bn(out)
        out = self.sig(out)

        return out


class MyNet6View_views_label_correct(nn.Module):

    def __init__(self, num_classes=23, num_classes_p=5):
        super(MyNet6View_views_label_correct, self).__init__()

        self.num_classes = num_classes
        self.num_classes_p = num_classes_p

        self.MyNet1 = MyNet(input_channels=1, single_view=True)
        self.MyNet2 = MyNet(input_channels=2, single_view=True)
        self.MyNet3 = MyNet(input_channels=2, single_view=True)
        self.MyNet4 = MyNet(input_channels=2, single_view=True)
        self.MyNet5 = MyNet(input_channels=2, single_view=True)
        self.MyNet6 = MyNet(input_channels=3, single_view=True)

        self.fuse_weight_1 = AdaptiveWeight(128)
        self.fuse_weight_2 = AdaptiveWeight(128)
        self.fuse_weight_3 = AdaptiveWeight(128)
        self.fuse_weight_4 = AdaptiveWeight(128)
        self.fuse_weight_5 = AdaptiveWeight(128)
        self.fuse_weight_6 = AdaptiveWeight(128)

        # private_feature_extraction
        self.private_feature_extraction = nn.ModuleList(
            [nn.Sequential(nn.Linear(128, 128),
                           nn.Dropout(p=0.1),
                           nn.Tanh()) for _ in range(6)]
        )

        # common_feature_extraction
        self.common_feature_extraction = nn.Sequential(
            nn.Linear(128, 128),
            nn.Dropout(p=0.3),
            nn.Tanh()
        )

        # view_discriminator
        self.view_discriminator = nn.Sequential(
            nn.Linear(128, 128//2),
            nn.Dropout(p=0.1),
            nn.Tanh(),
            nn.Linear(128//2, 6)
        )

        # common_classifier
        self.common_classfier = nn.Sequential(
            nn.Linear(128, num_classes),
            nn.Dropout(p=0.1),
            nn.Sigmoid()
        )

        # bert降维
        self.bert_projetion = nn.Linear(768, 128)

        self.avgpool_1 = nn.AdaptiveAvgPool1d(1)
        self.avgpool_2 = nn.AdaptiveAvgPool1d(1)
        self.fc_p = nn.Linear(128, num_classes_p)
        # 私有特征和公共特征的自适应平均池化
        self.private_feature_avgpool = nn.ModuleList(
            [nn.Sequential(nn.AdaptiveAvgPool1d(1)) for _ in range(6)]
        )
        self.common_feature_avgpool = nn.AdaptiveAvgPool1d(1)

        self.fc = nn.Linear(128, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    # 正交loss
    def calculate_orthogonality_loss(self, first_feature, second_feature):
        # diff_loss = torch.norm(torch.bmm(first_feature, second_feature.transpose(0, 1)), dim=(0, 1)).pow(2).mean()
        diff_loss = torch.sum(torch.sum(first_feature*second_feature, dim=1) ** 2)
        return diff_loss

    def forward(self, x, target_p, target, Train=False, device='cuda:1'):

        outputs_view = [self.MyNet1(x[:, 3, :].unsqueeze(1)),
                        self.MyNet2(torch.cat((x[:, 0, :].unsqueeze(1), x[:, 4, :].unsqueeze(1)), dim=1)),
                        self.MyNet3(x[:, 6:8, :]),
                        self.MyNet4(x[:, 8:10, :]),
                        self.MyNet5(x[:, 10:12, :]),
                        self.MyNet6(torch.cat((x[:, 1:3, :], x[:, 5, :].unsqueeze(1)), dim=1))]
        hidden_state_view = [outputs_view[i][1] for i in range(6)]
        # print(hidden_state_view[0].shape)  # [64, 128, 244]
        outputs_view = [outputs_view[i][0] for i in range(6)]
        # print(outputs_view[0].shape)  # [64, 128]

        private_views = [self.private_feature_extraction[i](hidden_state_view[i].transpose(2, 1)) for i in range(6)]
        # print(private_views[0].shape)  # [64, 244, 128]
        common_views = [self.common_feature_extraction(hidden_state_view[i].transpose(2, 1)) for i in range(6)]
        # print(common_views[0].shape)  # [64, 244, 128]

        # common_feature = common_views[0] + common_views[1] + common_views[2] + common_views[3] + common_views[4] + common_views[5]  # [64, 244, 128]

        fuse_weight_1 = self.fuse_weight_1(outputs_view[0])
        # print(fuse_weight_1.shape)  # [64, 1]
        fuse_weight_2 = self.fuse_weight_2(outputs_view[1])
        fuse_weight_3 = self.fuse_weight_3(outputs_view[2])
        fuse_weight_4 = self.fuse_weight_4(outputs_view[3])
        fuse_weight_5 = self.fuse_weight_5(outputs_view[4])
        fuse_weight_6 = self.fuse_weight_6(outputs_view[5])

        new_output = (fuse_weight_1.view(-1, 1, 1) * (private_views[0] + common_views[0])
                      + fuse_weight_2.view(-1, 1, 1) * (private_views[1] + common_views[1])
                      + fuse_weight_3.view(-1, 1, 1) * (private_views[2] + common_views[2])
                      + fuse_weight_4.view(-1, 1, 1) * (private_views[3] + common_views[3])
                      + fuse_weight_5.view(-1, 1, 1) * (private_views[4] + common_views[4])
                      + fuse_weight_6.view(-1, 1, 1) * (private_views[5] + common_views[5]))
        # print(new_output.shape)  # [64, 244, 128]

        # 标签约束
        key_list = list(token_mapping.keys())
        bert_embeddings = np.load(os.path.join('./', 'bert_base_label.npy'))  # (107, 768)
        if self.num_classes_p == 5:
            parent_label = ['CD', 'HYP', 'MI', 'NORM', 'STTC']
            sub_label = ['AMI', 'CLBBB', 'CRBBB', 'ILBBB', 'IMI', 'IRBBB', 'ISCA', 'ISCI', 'ISC_', 'IVCD', 'LAFB/LPFB',
                         'LAO/LAE', 'LMI', 'LVH', 'NORM', 'NST_', 'PMI', 'RAO/RAE', 'RVH', 'SEHYP', 'STTC', 'WPW',
                         '_AVB']
            # 父类别标签嵌入
            parent_label_embedding = []
            for label in parent_label:
                label_index = key_list.index(label)
                parent_label_embedding.append(bert_embeddings[label_index])
            parent_label_embedding = np.array(parent_label_embedding)
            # print(parent_label_embedding.shape)  # (5, 768)
            parent_label_embedding = self.bert_projetion(torch.tensor(parent_label_embedding, dtype=torch.float32).to(device))  # [5, 128]
            # 子类别标签嵌入
            sub_label_embedding = []
            for label in sub_label:
                label_index = key_list.index(label)
                sub_label_embedding.append(bert_embeddings[label_index])
            sub_label_embedding = np.array(sub_label_embedding)
            # print(sub_label_embedding.shape)  # (23, 768)
            sub_label_embedding = self.bert_projetion(torch.tensor(sub_label_embedding, dtype=torch.float32).to(device))  # [23, 128]
            # 先验层级约束
            E = torch.tensor(np.array(ptb_5t23), dtype=torch.float32).to(device)
            # 粗粒度标签预测, btd = (64, 244, 128) ld = (5, 128)
            parent_scores = torch.einsum("btd,ld->btl", new_output, parent_label_embedding).transpose(2, 1)  # (64, 244, 5) -> (64, 5, 244)
            parent_a = F.softmax(parent_scores, dim=-1)  # (64, 5, 244)
            # (64, 5, 244) (64, 244, 128)
            parent_context = torch.einsum("blt,btd->bld", parent_a, new_output)  # (64, 5, 128)
            parent_context = self.avgpool_1(parent_context.transpose(2, 1))  # (64, 128, 5) -> (64, 128, 1)
            parent_context = parent_context.view(parent_context.size(0), -1)  # (64, 128)
            x_out_p = self.fc_p(parent_context)  # (64, 5)
            x_out_p_s = torch.sigmoid(x_out_p)  # (64, 5), sigmoid
            # 跨层级标签约束
            O_v1 = parent_scores  # (64, 5, 244)
            O_v1 = O_v1 * x_out_p_s.unsqueeze(-1)
            G_v1 = F.softmax(O_v1, dim=-1)  # (64, 5, 244)
            # btd = (64, 244, 128), ld = (23, 128)
            G_v2 = torch.einsum("btd,ld->btl", new_output, sub_label_embedding).transpose(2, 1)  # (64, 244, 23) -> (64, 23, 244)
            G_v2 = F.softmax(G_v2, dim=-1)
            # E (23, 5), G_v1 (64, 5, 244)
            guided = torch.einsum("lk,bkd->bld", E, G_v1)  # (64, 23, 244)
            alpha = 0.5
            G_v2_prime = G_v2 + alpha * guided  # (64, 23, 244)
            # blt = (64, 23, 244), btd = (64, 244, 128)
            sub_output = torch.einsum("blt,btd->bld", G_v2_prime, new_output)  # (64, 23, 128)
            sub_output = self.avgpool_2(sub_output.transpose(2, 1))  # (64, 128, 23) -> (64, 128, 1)
            sub_output = sub_output.view(sub_output.size(0), -1)  # (64, 128)
            new_output = sub_output

        x_out = self.fc(new_output)
        #
        matrices = []
        for i in range(6):
            matrix = []
            for row in range(private_views[0].shape[0]):
                current_row = [0] * 6
                current_row[i] = 1
                matrix.append(current_row)
            matrices.append(matrix)
        # adversial training
        private_views = [self.private_feature_avgpool[i](private_views[i].transpose(2, 1)).view(new_output.size(0), -1) for i in range(6)]
        common_views = [self.common_feature_avgpool(common_views[i].transpose(2, 1)).view(new_output.size(0), -1) for i in range(6)]
        common_feature = common_views[0] + common_views[1] + common_views[2] + common_views[3] + common_views[4] + common_views[5]  # [64, 128]
        if Train:
            view_labels = [torch.tensor(np.array(mat), dtype=torch.float32).to(device) for mat in matrices]
            private_preds = [self.view_discriminator(private_views[i]) for i in range(6)]
            # print(private_preds[0].shape)  # [64, 6]
            common_preds = [self.view_discriminator(GradReverse.grad_reverse(common_views[i], 1)) for i in range(6)]
            # print(common_preds[0].shape)  # [64, 6]
            diff_loss = sum([self.calculate_orthogonality_loss(private_views[i], common_views[i]) for i in range(6)])
            # common_label_preds
            common_label_preds = self.common_classfier(common_feature)

        if Train:
            return x_out, private_preds, common_preds, view_labels, diff_loss, common_label_preds, x_out_p
        else:
            return x_out
