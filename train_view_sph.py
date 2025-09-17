import torch, time, os
import models, utils
from torch import nn, optim
from dataset import load_datasets
from config import config
from sklearn.metrics import roc_auc_score
import numpy as np
import random
import pandas as pd
from utils import find_thresholds, find_precision, compute_mAP
import warnings

warnings.filterwarnings("ignore")

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def save_checkpoint(best_auc, model, optimizer, epoch):
    print('Model Saving...')
    if config.device_num > 1:
        model_state_dict = model.module.state_dict()
    else:
        model_state_dict = model.state_dict()

    torch.save({
        'model_state_dict': model_state_dict,
        'global_epoch': epoch,
        'optimizer_state_dict': optimizer.state_dict(),
        'best_auc': best_auc,
    }, os.path.join('checkpoints', config.model_name + '_' + config.experiment + '_checkpoint_best.pth'))


def train_epoch(model, optimizer, criterion, train_dataloader):
    model.train()
    loss_meter, it_count = 0, 0
    outputs = []
    targets = []
    for inputs, target in train_dataloader:

        inputs = inputs + torch.randn_like(inputs) * 0.1

        inputs = inputs.to(device)
        target = target.to(device)
        # zero the parameter gradients
        optimizer.zero_grad()
        # forward
        output, private_preds, common_preds, view_labels, diff_loss, common_label_preds = model(inputs, Train=True, device=device)
        # adv loss
        loss_fn = nn.CrossEntropyLoss()
        adv_private_loss_per = [loss_fn(private_preds[i], view_labels[i]) for i in range(6)]
        adv_private_loss = sum(adv_private_loss_per)
        adv_common_loss_per = [loss_fn(common_preds[i], view_labels[i]) for i in range(6)]
        adv_common_loss = sum(adv_common_loss_per)
        # cml
        cml_loss = criterion(common_label_preds, target)

        loss = criterion(output, target) + 0.01 * (adv_private_loss + adv_common_loss) + 5e-6 * diff_loss + 0.5 * cml_loss
        loss.backward()
        optimizer.step()
        loss_meter += loss.item()
        it_count += 1

        output = torch.sigmoid(output)
        for i in range(len(output)):
            outputs.append(output[i].cpu().detach().numpy())
            targets.append(target[i].cpu().detach().numpy())

    auc = roc_auc_score(targets, outputs)
    TPR = utils.compute_TPR(targets, outputs)
    #
    mAP = compute_mAP(targets, outputs)
    best_thresholds, best_f1s = find_thresholds(targets, outputs)
    best_precisions, best_recalls = find_precision(targets, outputs)
    print('train_loss: %.4f,   macro_auc: %.4f,   TPR: %.4f, f1: %.4f, precision: %.4f, recall: %.4f, mAP: %.4f' % (
    loss_meter / it_count, auc, TPR, np.mean(best_f1s), np.mean(best_precisions), np.mean(best_recalls), mAP))
    return loss_meter / it_count, auc, TPR, best_f1s, best_precisions, best_recalls, mAP


def val_epoch(model, criterion, val_dataloader):
    model.eval()
    loss_meter, it_count = 0, 0
    outputs = []
    targets = []
    with torch.no_grad():
        for inputs, target in val_dataloader:

            inputs = inputs + torch.randn_like(inputs) * 0.1

            inputs = inputs.to(device)
            target = target.to(device)
            output = model(inputs)
            loss = criterion(output, target)
            loss_meter += loss.item()
            it_count += 1

            output = torch.sigmoid(output)
            for i in range(len(output)):
                outputs.append(output[i].cpu().detach().numpy())
                targets.append(target[i].cpu().detach().numpy())

        auc = roc_auc_score(targets, outputs)
        TPR = utils.compute_TPR(targets, outputs)

    #
    mAP = compute_mAP(targets, outputs)
    best_thresholds, best_f1s = find_thresholds(targets, outputs)
    best_precisions, best_recalls = find_precision(targets, outputs)
    print('val_loss: %.4f,   macro_auc: %.4f,   TPR: %.4f, f1: %.4f, precision: %.4f, recall: %.4f, mAP: %.4f' % (
    loss_meter / it_count, auc, TPR, np.mean(best_f1s), np.mean(best_precisions), np.mean(best_recalls), mAP))
    return loss_meter / it_count, auc, TPR, best_f1s, best_precisions, best_recalls, mAP


# val and test
def test_epoch(model, criterion, test_dataloader):
    model.eval()
    loss_meter, it_count = 0, 0
    outputs = []
    targets = []
    with torch.no_grad():
        for inputs, target in test_dataloader:

            inputs = inputs + torch.randn_like(inputs) * 0.1

            inputs = inputs.to(device)
            target = target.to(device)
            output = model(inputs)
            loss = criterion(output, target)
            loss_meter += loss.item()
            it_count += 1

            output = torch.sigmoid(output)
            for i in range(len(output)):
                outputs.append(output[i].cpu().detach().numpy())
                targets.append(target[i].cpu().detach().numpy())

        auc = roc_auc_score(targets, outputs)
        TPR = utils.compute_TPR(targets, outputs)

    #
    mAP = compute_mAP(targets, outputs)
    best_thresholds, best_f1s = find_thresholds(targets, outputs)
    best_precisions, best_recalls = find_precision(targets, outputs)
    print('test_loss: %.4f,   macro_auc: %.4f,   TPR: %.4f, f1: %.4f, precision: %.4f, recall: %.4f, mAP: %.4f' % (
    loss_meter / it_count, auc, TPR, np.mean(best_f1s), np.mean(best_precisions), np.mean(best_recalls), mAP))
    return loss_meter / it_count, auc, TPR, best_f1s, best_precisions, best_recalls, mAP


def train(config=config):
    # seed
    setup_seed(config.seed)
    print('torch.cuda.is_available:', torch.cuda.is_available())

    # datasets
    train_dataloader, val_dataloader, test_dataloader, num_classes = load_datasets(
        datafolder=config.datafolder,
        experiment=config.experiment,
    )

    # mode
    model = getattr(models, config.model_name)(num_classes=num_classes)
    print('model_name:{}, num_classes={}'.format(config.model_name, num_classes))
    model = model.to(device)

    # optimizer and loss
    optimizer = optim.Adam(model.parameters(), lr=config.lr)
    criterion = nn.BCEWithLogitsLoss()

    if not os.path.isdir('checkpoints'):
        os.mkdir('checkpoints')

    # =========>train<=========
    for epoch in range(1, config.max_epoch + 1):
        print('#epoch: {}  batch_size: {}  Current Learning Rate: {}'.format(epoch, config.batch_size,
                                                                             config.lr))

        since = time.time()
        train_loss, train_auc, train_TPR, train_all_f1, train_all_precision, train_all_recall, train_mAP = train_epoch(model, optimizer, criterion, train_dataloader)

        val_loss, val_auc, val_TPR, val_all_f1, val_all_precision, val_all_recall, val_mAP = val_epoch(model, criterion, val_dataloader)

        test_loss, test_auc, test_TPR, test_all_f1, test_all_precision, test_all_recall, test_mAP = test_epoch(model, criterion, test_dataloader)

        save_checkpoint(test_auc, model, optimizer, epoch)

        result_list = [[epoch, train_loss, train_auc, train_TPR,
                        val_loss, val_auc, val_TPR, np.mean(val_all_f1), np.mean(val_all_precision), np.mean(val_all_recall), val_mAP,
                        test_loss, test_auc, test_TPR, np.mean(test_all_f1), np.mean(test_all_precision), np.mean(test_all_recall), test_mAP]]

        if epoch == 1:
            columns = ['epoch', 'train_loss', 'train_auc', 'train_TPR',
                       'val_loss', 'val_auc', 'val_TPR', 'val_f1', 'val_precision', 'val_recall', 'val_mAP',
                       'test_loss', 'test_auc', 'test_TPR', 'test_f1', 'test_precision', 'test_recall', 'test_mAP']

        else:
            columns = ['', '', '', '', '', '', '', '', '', ''] + [''] * 8

        dt = pd.DataFrame(result_list, columns=columns)
        dt.to_csv(config.model_name + config.experiment + 'result.csv', mode='a')

        print('time:%s\n' % (utils.print_time_cost(since)))


if __name__ == '__main__':
    config.datafolder = 'data/SPH/'
    config.experiment = 'exp1.1.1'
    config.seed = 20
    config.model_name = 'MyView'
    config.batch_size = 64
    train(config)
