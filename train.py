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
import baseline_model
# import source_model_1
import xlwt
import target_model
# import source_model_2
import os

def save_model_with_checks(model, save_path):
    # 检查并创建保存路径的目录
    save_dir = os.path.dirname(save_path)
    os.makedirs(save_dir, exist_ok=True)
    # 保存模型
    torch.save(model.state_dict(), save_path)
    print(f"Model saved at {save_path}")
def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
def save_xlsx(ts_result, model, site, max_epoch, lr, batch_size, weight_decay):
    runtime_id = './result/{}/{}-{}_site_result-{}-{}-{}-{}-{}'.format(site, model, site, max_epoch, lr, batch_size,
                                                                       weight_decay,
                                                                       time.strftime('%Y-%m-%d %H-%M-%S',
                                                                                     time.localtime()))
    # 检查并创建目录
    save_dir = os.path.dirname(runtime_id)  # 获取 './result/{site}/' 部分
    os.makedirs(save_dir, exist_ok=True)

    f = xlwt.Workbook('encoding = utf-8')  # 设置工作簿编码
    sheet1 = f.add_sheet('sheet1', cell_overwrite_ok=True)  # 创建sheet工作表
    a = np.average(
        [ts_result[1], ts_result[2], ts_result[3], ts_result[4], ts_result[5]], axis=0).tolist()
    a[0] = 'average'
    ts_result.append(a)
    # print(a)
    for j in range(len(ts_result)):
        for i in range(len(ts_result[j])):
            sheet1.write(j, i, ts_result[j][i])  # 写入数据参数对应 行, 列, 值
    f.save(runtime_id + '.xlsx')  # 保存.xls到当前工作目录
# 统计参数数量
def count_parameters(model):
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_params, trainable_params


def make_model(args,temperature,N=1, d_model1=6670, d_model2=28, d_ff=128, h1=2,dropout=0.5,device=1):
    c = copy.deepcopy
    if args.model == 'baseline_model':
         prompt1 = None
         prompt2 = None
         attn1 = baseline_model.MultiHeadedAttention(h1, d_model1,device)
         ff = baseline_model.PositionwiseFeedForward(d_model1, d_ff, dropout)
         position = baseline_model.PositionalEncoding(d_model1, dropout)
         model = baseline_model.EncoderDecoder(baseline_model.Encoder(baseline_model.EncoderLayer(d_model1, d_model2, c(attn1), c(ff), dropout), N),c(position),d_model1, d_model2)
    elif args.model == 'source_model_1':
        prompt1 = None
        prompt2 = None
        attn1 = source_model_1.MultiHeadedAttention(h1, d_model1, device)
        ff = source_model_1.PositionwiseFeedForward(d_model1, d_ff, dropout)
        position = source_model_1.PositionalEncoding(d_model1, dropout)
        model = source_model_1.EncoderDecoder(source_model_1.Encoder(source_model_1.EncoderLayer(d_model1, d_model2, c(attn1), c(ff), dropout), N), c(position), d_model1, d_model2)
    elif args.model == 'source_model_2':
        prompt1 = None
        prompt2 = source_model_2.Prompt2(d_model1,temperature)
        attn1 = source_model_2.MultiHeadedAttention(h1, d_model1, device)
        ff = source_model_2.PositionwiseFeedForward(d_model1, d_ff, dropout)
        position = source_model_2.PositionalEncoding(d_model1, dropout)
        model = source_model_2.EncoderDecoder(source_model_2.Encoder(source_model_2.EncoderLayer(d_model1, d_model2, c(attn1), c(ff), dropout), N), c(position), d_model1, d_model2)
    elif args.model == 'target_model':
        prompt1 = target_model.Prompt(d_model1,temperature)
        attn1 = target_model.MultiHeadedAttention(h1,d_model1, device)  # h1==注意力机制的头数   d_model1==transformer-encoder输入中，每个token的维度
        ff = target_model.PositionwiseFeedForward(d_model1, d_ff,dropout)  # d_model1==transformer-encoder输入中，每个token的维度   d_ff=FFN隐藏层的维度
        position = target_model.PositionalEncoding(d_model1,dropout)  # 位置编码   d_model1==transformer-encoder输入中，每个token的维度
        model = target_model.EncoderDecoder(target_model.Encoder(target_model.EncoderLayer(d_model1, d_model2, c(attn1), c(ff), dropout), N),c(position),d_model1, d_model2)  # d_model2==凑维度，使得NLP分类维度匹配


        prompt2_source1 = target_model.Prompt2(d_model1, temperature)
        prompt2_source2 = target_model.Prompt2(d_model1, temperature)
        prompt2_source3 = target_model.Prompt2(d_model1, temperature)
        prompt2_target = target_model.Prompt2(d_model1, temperature)

        attn1 = source_model_2.MultiHeadedAttention(h1, d_model1, device)
        ff = source_model_2.PositionwiseFeedForward(d_model1, d_ff, dropout)
        position = source_model_2.PositionalEncoding(d_model1, dropout)
        model1 = source_model_2.EncoderDecoder(
            source_model_2.Encoder(source_model_2.EncoderLayer(d_model1, d_model2, c(attn1), c(ff), dropout), N),
            c(position), d_model1, d_model2)
        prompt2 = target_model.LateFusionDistillationModel(model1 , prompt1, prompt2_source1, prompt2_source2, prompt2_source3,
                                                   prompt2_target)
    
    elif args.model == 'target_model_only_instance':
        prompt1 = None
        prompt2 = source_model_2.Prompt2(d_model1,temperature)
        attn1 = source_model_2.MultiHeadedAttention(h1, d_model1, device)
        ff = source_model_2.PositionwiseFeedForward(d_model1, d_ff, dropout)
        position = source_model_2.PositionalEncoding(d_model1, dropout)
        model = source_model_2.EncoderDecoder(source_model_2.Encoder(source_model_2.EncoderLayer(d_model1, d_model2, c(attn1), c(ff), dropout), N), c(position), d_model1, d_model2)
        
    # Initialize parameters with Glorot / fan_avg.
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
    return model,prompt1,prompt2

def configure_model_optimizer(args,model,prompt1,prompt2,device, kfold_index, lr,decay):
    """
    配置模型的参数更新和优化器。

    Args:
        model (nn.Module): PyTorch模型对象。
        kfold_index (int): 批次。
        args: 模型选择

    Returns:
        model (nn.Module): 配置完成的模型。
        model_param_group: 所需要更新的参数。
    """
    if args.model == 'baseline_model':
        model.to(device)

        lr = 0.00002  # 0.00002#0.00015 # 0.0002
        decay = 0.02

        model_param_group = []
        model_param_group.append({"params": model.parameters()})
        optimizer = torch.optim.Adam(model_param_group, lr=lr, betas=(0.9, 0.98), eps=1e-9, weight_decay=decay)
    elif args.model == 'source_model_1':
        # 加载预训练权重
        model.load_state_dict(torch.load('./baseline_model/' + str(kfold_index) + '.pt'), strict=False)

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
        lr = 0.0005
        decay = 0.02


        model_param_group = []
        model_param_group.append({"params": model.parameters()})
        optimizer = torch.optim.Adam(model_param_group, lr=lr, betas=(0.9, 0.98), eps=1e-9, weight_decay=decay)
    elif args.model == 'source_model_2':
        # 加载预训练权重
        model.load_state_dict(torch.load('./baseline_model/' + str(kfold_index) + '.pt'), strict=False)

        # 冻结所有参数
        for param in model.parameters():
            param.requires_grad = False

        model.to(device)
        prompt2.to(device)
        # lr = 0.0002 # ---NYU  best_epoch=70
        # decay = 0.4


        # lr = 0.18  # ---UCLA  best_epoch=195
        # decay = 0.4


        lr = 0.001  #---UM  best_epoch=15
        decay = 0.05

        # lr = 0.005  #---USM  best_epoch=120
        # decay = 0.5

        model_param_group = []
        model_param_group.append({"params": prompt2.parameters()})
        optimizer = torch.optim.Adam(model_param_group, lr=lr, betas=(0.9, 0.98), eps=1e-9, weight_decay=decay)
    elif args.model == 'target_model':



        # 定义源站点和目标站点映射
        site_mapping = {
            "UCLA": ["NYU", "UM", "USM"],
            "NYU": ["UCLA", "UM", "USM"],
            "UM": ["UCLA", "NYU", "USM"],
            "USM": ["UCLA", "NYU", "UM"]
        }

        # 根据目标站点选择源站点
        source_sites = site_mapping[args.site]

        # 加载预训练权重
        model.load_state_dict(torch.load(f'./baseline_model/{kfold_index}.pt'), strict=False)
        # 加载prompt2中的预训练权重
        prompt2.model1.load_state_dict(torch.load('./baseline_model/' + str(kfold_index) + '.pt'), strict=False)
        
        # 加载 Attention 权重
        weights = ['W_A', 'W_B', 'W_C', 'W_D', 'W_E', 'W_F']
        for i, site in enumerate(source_sites, start=1):
            # print(site)
            state_dict = torch.load(f'./source_model_1/{site}/{kfold_index}.pt')
            for weight in weights:
                attr_name = f'{weight}_{i}'
                if hasattr(model, attr_name):  # 确保模型有该属性
                    # 修改 .data 确保更新张量值
                    getattr(model, attr_name).data = state_dict[weight]
                else:
                    print(f"Warning: Attribute {attr_name} does not exist in model.")



        for p in model.parameters():
            p.requires_grad = False
        for name, param in model.named_parameters():
            if "W_A_4" in name:
                param.requires_grad = True
            if "W_B_4" in name:
                param.requires_grad = True
            if "W_C_4" in name:
                param.requires_grad = True
            if "W_D_4" in name:
                param.requires_grad = True
            if "W_E_4" in name:
                param.requires_grad = True
            if "W_F_4" in name:
                param.requires_grad = True

        # for name, value in prompt2.state_dict().items():
        #     print(f"Parameter name: {name}, Shape: {value.shape}")

        # 加载每个源站点的模型权重并赋值给对应的 prompt2_source
        for i, site in enumerate(source_sites, start=1):
            # 加载源站点的模型权重
            state_dict = torch.load(f'./source_model_2/{site}/{kfold_index}.pt')

            # 选择对应的 prompt2 模块
            prompt_source = getattr(prompt2, f"model{i}", None)
            if prompt_source is None:
                raise ValueError(f"Prompt source model{i} not found in prompt2.")

            # 检查 state_dict 是否包含所有必要的 key
            required_keys = ['attn_W_down.weight', 'attn_W_up.weight', 'layer_norm.weight', 'layer_norm.bias']
            missing_keys = [key for key in required_keys if key not in state_dict]
            if missing_keys:
                raise KeyError(f"Missing keys in state_dict for site {site}: {missing_keys}")

            # 将加载的权重赋值到 prompt_source 模块中
            prompt_source.attn_W_down.weight.data = state_dict['attn_W_down.weight']
            prompt_source.attn_W_up.weight.data = state_dict['attn_W_up.weight']
            prompt_source.layer_norm.weight.data = state_dict['layer_norm.weight']
            prompt_source.layer_norm.bias.data = state_dict['layer_norm.bias']

        for p in prompt2.parameters():
            p.requires_grad = False

        for name, param in prompt2.named_parameters():
            if "student_model" in name:
                param.requires_grad = True

        for p in prompt1.parameters():
            p.requires_grad = True

        model.to(device)
        prompt1.to(device)
        prompt2.to(device)

        # # 定义学习率
        prompt_lr = 0.0000851
        model_lr = 0.0001

        decay = decay


        # 创建参数组列表
        model_param_group = []
        # 将 prompt_model 的参数添加到参数组列表中，并设置对应的学习率
        model_param_group.append({"params": prompt1.parameters(), "lr": prompt_lr})
        # 将 prompt_model 的参数添加到参数组列表中，并设置对应的学习率
        model_param_group.append({"params": prompt2.student_model.parameters(), "lr": lr})#lr
        # 将 model 的参数添加到参数组列表中，并设置对应的学习率
        model_param_group.append({"params": model.parameters(), "lr": model_lr})
        # 创建优化器
        optimizer = torch.optim.Adam(model_param_group, betas=(0.9, 0.98), eps=1e-9, weight_decay=decay)






    elif args.model == 'target_model_only_instance':

        # 加载预训练权重
        model.load_state_dict(torch.load(f'./baseline_model/{kfold_index}.pt'), strict=False)





        # 加载源站点的模型权重
        state_dict1 = torch.load(f'./source_model_3/{args.site}/{kfold_index}.pt')
        # 将加载的权重赋值到对应的 prompt_source 模块中
        prompt2.attn_W_down.weight.data = state_dict1['attn_W_down.weight']
        prompt2.attn_W_up.weight.data = state_dict1['attn_W_up.weight']
        prompt2.layer_norm.weight.data = state_dict1['layer_norm.weight']
        prompt2.layer_norm.bias.data = state_dict1['layer_norm.bias']

        # lr = 0.0001 # ---NYU  --UCLA
        # decay = 0.015

        model.to(device)
        prompt2.to(device)

        # 定义学习率
        lr=lr
        decay = decay

        # 创建参数组列表
        model_param_group = []
        # 将 prompt_model 的参数添加到参数组列表中，并设置对应的学习率
        model_param_group.append({"params": prompt2.parameters(), "lr": lr})
        # 创建优化器
        optimizer = torch.optim.Adam(model_param_group, betas=(0.9, 0.98), eps=1e-9, weight_decay=decay)
    return model, model_param_group,optimizer


def train_and_test(args, path_data,path_data_mask, path_label, lr, weight_decay, temperature):
    setup_seed(args.seed)
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    # print(device)

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
    loss_all=[]

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
        model,prompt1,prompt2 = make_model(args,temperature,N=1, d_model1=6670, d_model2=25, d_ff=128, h1=2, dropout=args.dropout,device=device)
        # #选择baseline_model,source_model,target_model模型

        # #根据选择的模型不同，设定不同模型的初始化和选择更新的参数
        model,model_param_group,optimizer=configure_model_optimizer(args, model,prompt1,prompt2,device, kfold_index, lr,weight_decay)



        # 统计 prompt1、prompt2 和 model 的参数
        # for model_name, component in [("Prompt1", prompt1), ("Prompt2", prompt2), ("Model", model)]:
        #     total, trainable = count_parameters(component)
        #     print(f"{model_name} - Total Parameters: {total}, Trainable Parameters: {trainable}")


        #训练模型
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
                if args.model == 'target_model':
                    outputs,src_loss = model(train_data_batch, src_mask=X_mask_train_batch, prompt1=prompt1, prompt2=prompt2)
                    outputs = outputs.cpu()
                    loss = F.cross_entropy(outputs, train_label_batch, reduction='mean')
                    # print('src_loss',src_loss)
                    # print('loss',loss)
                    loss = loss + 0.3 * src_loss  # src_loss是蒸馏损失
                else:
                     outputs,_ = model(train_data_batch, src_mask=X_mask_train_batch, prompt1=prompt1,
                                                  prompt2=prompt2)
                     outputs = outputs.cpu()
                     loss = F.cross_entropy(outputs, train_label_batch, reduction='mean')

                loss.backward()
                optimizer.step()


            if epoch % 5 == 0:
                count = 0
                acc = 0
                for i in range(0, int(X_train.shape[0]) // int(args.batch_size)):
                    idx_batch = idx_batchs[i * int(args.batch_size):min((i + 1) * int(args.batch_size), X_train.shape[0])]
                    train_data_batch = X_train[idx_batch]
                    train_label_batch = Y_train[idx_batch]
                    X_mask_train_batch = X_mask_train[idx_batch]

                    X_mask_train_batch = torch.from_numpy(X_mask_train_batch).to(device)
                    train_data_batch = torch.from_numpy(train_data_batch).float().to(device)
                    train_label_batch = torch.from_numpy(train_label_batch).long()


                    outputs, _ = model(train_data_batch, src_mask=X_mask_train_batch, prompt1=prompt1,
                                        prompt2=prompt2)
                        
                    _, indices = torch.max(outputs, dim=1)
                    preds = indices.cpu()
                    acc += metrics.accuracy_score(preds, train_label_batch)
                    count = count + 1
                # 训练的loss输出可能有误
                print('train\tepoch: %d\tloss: %.4f\t\tacc: %.4f' % (epoch, loss.item(), acc / count))
                #训练集loss下降曲线
                loss_all.append(loss.data.item())
            # 保存特征
            if args.model == 'source_model_2' and epoch==50 and kfold_index==1:
                X_mask_train_batch_dev = torch.from_numpy(X_mask_train).to(device)
                train_data_batch_dev = torch.from_numpy(X_train).float().to(device)
                outputs, outputs_CC = model(train_data_batch_dev, src_mask=X_mask_train_batch_dev, prompt1=prompt1,
                                            prompt2=prompt2)

                _, indices = torch.max(outputs, dim=1)
                pre = indices.cpu()
                label_train = Y_train
                acc_test = metrics.accuracy_score(label_train, pre)
                fpr, tpr, _ = metrics.roc_curve(label_train, pre)
                auc = metrics.auc(fpr, tpr)
                tn, fp, fn, tp = metrics.confusion_matrix(label_train, pre).ravel()
                sen = tp / (tp + fn)
                spe = tn / (tn + fp)
                precision = metrics.precision_score(label_train, pre, zero_division=1)
                f1 = metrics.f1_score(label_train, pre)
                reacall = metrics.recall_score(label_train, pre)

                # 打印结果
                print(f"Accuracy: {acc_test}")
                print(f"AUC: {auc}")
                print(f"Sensitivity (Recall): {sen}")
                print(f"Specificity: {spe}")
                print(f"Precision: {precision}")
                print(f"F1 Score: {f1}")

                file_name = str(args.site) + "_X" + ".npy"
                np.save(file_name, outputs_CC.cpu().detach().numpy())

                file_name1 = str(args.site) + "_Y" + ".npy"
                np.save(file_name1, Y_train)


            if epoch % 5 == 0:
                    model.eval()
                    X_mask_test_batch_dev = torch.from_numpy(X_mask_test).to(device)
                    test_data_batch_dev = torch.from_numpy(X_test).float().to(device)

                    outputs, _ = model(test_data_batch_dev, src_mask=X_mask_test_batch_dev, prompt1=prompt1,
                                        prompt2=prompt2)

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
                if args.model == 'baseline_model':
                    save_path = f'./baseline_model/{kfold_index}.pt'
                    save_model_with_checks(model, save_path)

                elif args.model == 'source_model_1':
                    save_path = f'./source_model_1/{args.site}/{kfold_index}.pt'
                    save_model_with_checks(model, save_path)

                elif args.model == 'source_model_2':
                    save_path = f'./source_model_2/{args.site}/{kfold_index}.pt'
                    save_model_with_checks(prompt2, save_path)

                elif args.model == 'target_model':
                    # 保存主模型
                    save_path_main = f'./target_model/{args.site}/{kfold_index}.pt'
                    save_model_with_checks(model, save_path_main)

                    # 保存 prompt1
                    save_path_prompt1 = f'./target_model/{args.site}/prompt1/{kfold_index}.pt'
                    save_model_with_checks(prompt1, save_path_prompt1)

                    # 保存 prompt2
                    save_path_prompt2 = f'./target_model/{args.site}/prompt2/{kfold_index}.pt'
                    save_model_with_checks(prompt2, save_path_prompt2)

                elif args.model == 'target_model_only_instance':
                    save_path = f'./target_model_only_instance/{args.site}/{kfold_index}.pt'
                    save_model_with_checks(prompt2, save_path)
        # 判断训练集拟合情况   acc / count
        # if max(train_acc_list) < 0.8:
        #     print("Model not fitting properly: Train ACC < 0.8")
        #     return 0  # 直接返回不合法的结果

    ts_result = [['kfold_index', 'prec', 'recall', 'acc', 'F1', 'auc','sen','spe']]  # 创建一个空列表
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

            ts_result[1] = [1, precision_list[i], recall_list[i], acc_list[i], f1_list[i], auc_list[i],sen_list[i],spe_list[i]]
            ts_result[2] = [2, precision_list[i + num], recall_list[i + num], acc_list[i + num], f1_list[i + num],
                            auc_list[i + num],sen_list[i+ num],spe_list[i+ num]]
            ts_result[3] = [3, precision_list[i + 2 * num], recall_list[i + 2 * num], acc_list[i + 2 * num],
                            f1_list[i + 2 * num], auc_list[i + 2 * num],sen_list[i+ 2 * num],spe_list[i+ 2 * num]]
            ts_result[4] = [4, precision_list[i + 3 * num], recall_list[i + 3 * num], acc_list[i + 3 * num],
                            f1_list[i + 3 * num], auc_list[i + 3 * num],sen_list[i+ 3 * num],spe_list[i+ 3 * num]]
            ts_result[5] = [5, precision_list[i + 4 * num], recall_list[i + 4 * num], acc_list[i + 4 * num],
                            f1_list[i + 4 * num], auc_list[i + 4 * num],sen_list[i+ 4 * num],spe_list[i+ 4 * num]]

    print('{}-{}-{}-{}-{:.4}-{:.4}-{:.4}-{:.4}-{:.4}\n'.format(max_epoch, lr, args.batch_size, weight_decay, max_acc, max_precision,
                                                               max_recall, max_f1, max_auc))

    # 保存结果
    save_xlsx(ts_result,args.model,args.site,max_epoch, lr, args.batch_size, weight_decay)

    if (ts_result[1][5]==0.5 or ts_result[2][5]==0.5 or ts_result[3][5]==0.5 or ts_result[4][5]==0.5 or ts_result[5][5]==0.5):
        print("Model not fitting properly: Test AUC == 0.5")
        return 0
    return max_acc * 100



