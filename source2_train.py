from utils import save_model_with_checks, setup_seed, save_xlsx, count_parameters
import torch.optim as optim
from sklearn import metrics
import math, copy, time
import baseline_model
# import source_model_1
import xlwt
import source_model2
import os
import numpy as np
from torch.utils.data.dataloader import DataLoader
import torch
import torch.nn as nn
import torch.nn.functional as F
# from irr_model import model
# from model_new import model
import random
from sklearn.model_selection import train_test_split, KFold
import torch.optim as optim
from sklearn import metrics
import math, copy, time


def make_model_baseline_model(temperature=8,N=1, d_model1=6670, d_model2=28, d_ff=128, h1=2, dropout=0.5, device=0):
    c = copy.deepcopy
    attn1 = source_model2.MultiHeadedAttention(h1, d_model1, device)
    ff = source_model2.PositionwiseFeedForward(d_model1, d_ff, dropout)
    position = source_model2.PositionalEncoding(d_model1, dropout)
    model = source_model2.EncoderDecoder(
        source_model2.Encoder(source_model2.EncoderLayer(d_model1, d_model2, c(attn1), c(ff), dropout), N),
        c(position), d_model1, d_model2)

    # Initialize parameters with Glorot / fan_avg.
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
    return model
def configure_model_optimizer_baseline_model(args, model, kfold_index, device, lr, decay):
    """
    配置模型的参数更新和优化器。
    """
    if args.model == 'source_model2':
        # 加载预训练权重
        model.load_state_dict(torch.load('./baseline_model/'+ str(args.site) + '/15_site/' + str(kfold_index) + '.pt'), strict=False)

        # 冻结所有参数
        for param in model.parameters():
            param.requires_grad = False
        #只更新prompt参数
        for name, param in model.named_parameters():
            if "W_A" in name:
                param.requires_grad = True
            if "W_B" in name:
                param.requires_grad = True
            if "W_C" in name:
                param.requires_grad = True
            if "W_D" in name:
                param.requires_grad = True
            if "W_E" in name:
                param.requires_grad = True
            if "W_F" in name:
                param.requires_grad = True

        model.to(device)

        lr = 0.0002  #13个站点
        decay = 0.4

        # lr = 0.00002  #KKI
        # decay = 0.1

        model_param_group = []
        model_param_group.append({"params": model.parameters()})
        optimizer = torch.optim.Adam(model_param_group, lr=lr, betas=(0.9, 0.98), eps=1e-9, weight_decay=decay)

        optimizer = torch.optim.Adam(model_param_group, lr=lr, betas=(0.9, 0.98), eps=1e-9, weight_decay=decay)
    return model, model_param_group, optimizer
def train_and_test_source_train_model(args, path_data, path_data_mask, path_label, lr, weight_decay):
    setup_seed(args.seed)
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")


    X = np.load(path_data)
    X_mask = np.load(path_data_mask)
    Y = np.load(path_label)

    acc_list = []
    precision_list = []
    recall_list = []
    f1_list = []
    auc_list = []
    epoch_list = []
    sen_list = []
    spe_list = []
    loss_all = []

    max_acc = 0
    max_precision = 0
    max_recall = 0
    max_f1 = 0
    max_auc = 0
    max_epoch = 0
    kf = KFold(n_splits=5, random_state=args.seed, shuffle=True)
    kfold_index = 0
    for train_index, test_index in kf.split(X):
        kfold_index += 1
        # if kfold_index!=4:
        #     continue
        X_train, X_test = X[train_index], X[test_index]
        X_mask_train, X_mask_test = X_mask[train_index], X_mask[test_index]
        Y_train, Y_test = Y[train_index], Y[test_index]

        print('X_train{}'.format(X_train.shape))
        print('X_test{}'.format(X_test.shape), )
        print('Y_train{}'.format(Y_train.shape))
        print('Y_test{}'.format(Y_test.shape))

        # model #N=6, d_model1=72, d_model2=116, d_ff=2048, h1=8, h2=10
        model = make_model_baseline_model(temperature=8,N=1, d_model1=6670, d_model2=25, d_ff=128, h1=2,dropout=args.dropout, device=device)

        # #根据选择的模型不同，设定不同模型的初始化和选择更新的参数
        model, model_param_group, optimizer = configure_model_optimizer_baseline_model(args, model,kfold_index,device,lr, weight_decay)

        # 训练模型
        for epoch in range(0, args.epoch_cf + 1):
            model.train()
            idx_batchs = np.random.permutation(int(X_train.shape[0]))
            for i in range(0, int(X_train.shape[0]) // int(args.batch_size)):
                idx_batch = idx_batchs[i * int(args.batch_size):min((i + 1) * int(args.batch_size), X_train.shape[0])]

                train_data_batch = X_train[idx_batch]
                X_mask_train_batch = X_mask_train[idx_batch]
                train_label_batch = Y_train[idx_batch]

                train_data_batch = torch.from_numpy(train_data_batch).float().to(device)
                X_mask_train_batch = torch.from_numpy(X_mask_train_batch).to(device)
                train_label_batch = torch.from_numpy(train_label_batch).long()

                optimizer.zero_grad()

                outputs = model(train_data_batch, src_mask=X_mask_train_batch)
                outputs = outputs.cpu()
                loss = F.cross_entropy(outputs, train_label_batch, reduction='mean')

                loss.backward()
                optimizer.step()

            if epoch % 5 == 0:
                count = 0
                acc = 0
                for i in range(0, int(X_train.shape[0]) // int(args.batch_size)):
                    idx_batch = idx_batchs[
                                i * int(args.batch_size):min((i + 1) * int(args.batch_size), X_train.shape[0])]
                    train_data_batch = X_train[idx_batch]
                    train_label_batch = Y_train[idx_batch]
                    X_mask_train_batch = X_mask_train[idx_batch]

                    X_mask_train_batch = torch.from_numpy(X_mask_train_batch).to(device)
                    train_data_batch = torch.from_numpy(train_data_batch).float().to(device)
                    train_label_batch = torch.from_numpy(train_label_batch).long()

                    outputs = model(train_data_batch, src_mask=X_mask_train_batch)

                    _, indices = torch.max(outputs, dim=1)
                    preds = indices.cpu()
                    acc += metrics.accuracy_score(preds, train_label_batch)
                    count = count + 1
                # 训练的loss输出可能有误
                print('train\tepoch: %d\tloss: %.4f\t\tacc: %.4f' % (epoch, loss.item(), acc / count))
                # 训练集loss下降曲线
                loss_all.append(loss.data.item())

            if epoch % 5 == 0:
                model.eval()
                X_mask_test_batch_dev = torch.from_numpy(X_mask_test).to(device)
                test_data_batch_dev = torch.from_numpy(X_test).float().to(device)

                outputs = model(test_data_batch_dev, src_mask=X_mask_test_batch_dev)

                _, indices = torch.max(outputs, dim=1)
                pre = indices.cpu()
                label_test = Y_test
                acc_test = metrics.accuracy_score(label_test, pre)
                fpr, tpr, _ = metrics.roc_curve(label_test, pre)
                auc = metrics.auc(fpr, tpr)
                tn, fp, fn, tp = metrics.confusion_matrix(label_test, pre).ravel()
                sen = tp / (tp + fn)
                spe = tn / (tn + fp)
                precision = metrics.precision_score(label_test, pre, zero_division=1)
                f1 = metrics.f1_score(label_test, pre)
                reacall = metrics.recall_score(label_test, pre)

                acc_list.append(acc_test)
                precision_list.append(precision)
                recall_list.append(reacall)
                f1_list.append(f1)
                auc_list.append(auc)
                epoch_list.append(epoch)
                sen_list.append(sen)
                spe_list.append(spe)

                print('test result',
                      [kfold_index, epoch, round(acc_test, 4), round(precision, 4), round(reacall, 4), round(f1, 4),
                       round(auc, 4), round(sen, 4), round(spe, 4)])

            if epoch == args.best_epoch and args.save_model == 1:
                if args.model == 'source_model2':
                    save_path = f'./source_model2/{args.site}/{args.site_source}/{kfold_index}.pt'
                    save_model_with_checks(model, save_path)



    ts_result = [['kfold_index', 'prec', 'recall', 'acc', 'F1', 'auc', 'sen', 'spe']]  # 创建一个空列表
    for i in range(5):
        ts_result.append([])
    num = len(acc_list) // 5
    for i in range(num):
        if max_acc < (acc_list[i] + acc_list[i + num] + acc_list[i + 2 * num] + acc_list[i + 3 * num] + acc_list[
            i + 4 * num]) / 5:
            max_acc = (acc_list[i] + acc_list[i + num] + acc_list[i + 2 * num] + acc_list[i + 3 * num] + acc_list[
                i + 4 * num]) / 5
            max_precision = (precision_list[i] + precision_list[i + num] + precision_list[i + 2 * num] + precision_list[
                i + 3 * num] + precision_list[i + 4 * num]) / 5
            max_recall = (recall_list[i] + recall_list[i + num] + recall_list[i + 2 * num] + recall_list[i + 3 * num] +
                          recall_list[i + 4 * num]) / 5
            max_f1 = (f1_list[i] + f1_list[i + num] + f1_list[i + 2 * num] + f1_list[i + 3 * num] + f1_list[
                i + 4 * num]) / 5
            max_auc = (auc_list[i] + auc_list[i + num] + auc_list[i + 2 * num] + auc_list[i + 3 * num] + auc_list[
                i + 4 * num]) / 5
            max_sen = (sen_list[i] + sen_list[i + num] + sen_list[i + 2 * num] + sen_list[i + 3 * num] + sen_list[
                i + 4 * num]) / 5
            max_spe = (spe_list[i] + spe_list[i + num] + spe_list[i + 2 * num] + spe_list[i + 3 * num] + spe_list[
                i + 4 * num]) / 5
            max_epoch = epoch_list[i]

            ts_result[1] = [1, precision_list[i], recall_list[i], acc_list[i], f1_list[i], auc_list[i], sen_list[i],
                            spe_list[i]]
            ts_result[2] = [2, precision_list[i + num], recall_list[i + num], acc_list[i + num], f1_list[i + num],
                            auc_list[i + num], sen_list[i + num], spe_list[i + num]]
            ts_result[3] = [3, precision_list[i + 2 * num], recall_list[i + 2 * num], acc_list[i + 2 * num],
                            f1_list[i + 2 * num], auc_list[i + 2 * num], sen_list[i + 2 * num], spe_list[i + 2 * num]]
            ts_result[4] = [4, precision_list[i + 3 * num], recall_list[i + 3 * num], acc_list[i + 3 * num],
                            f1_list[i + 3 * num], auc_list[i + 3 * num], sen_list[i + 3 * num], spe_list[i + 3 * num]]
            ts_result[5] = [5, precision_list[i + 4 * num], recall_list[i + 4 * num], acc_list[i + 4 * num],
                            f1_list[i + 4 * num], auc_list[i + 4 * num], sen_list[i + 4 * num], spe_list[i + 4 * num]]

    print('{}-{}-{}-{}-{:.4}-{:.4}-{:.4}-{:.4}-{:.4}\n'.format(max_epoch, lr, args.batch_size, weight_decay, max_acc,
                                                               max_precision,
                                                               max_recall, max_f1, max_auc))

    # 保存结果
    save_xlsx(ts_result, args.model, args.site_source, max_epoch, lr, args.batch_size, weight_decay)

    if (ts_result[1][5] == 0.5 or ts_result[2][5] == 0.5 or ts_result[3][5] == 0.5 or ts_result[4][5] == 0.5 or
            ts_result[5][5] == 0.5):
        print("Model not fitting properly: Test AUC == 0.5")
        return 0
    return max_acc * 100
