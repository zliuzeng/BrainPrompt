import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
# from irr_model import model
# from model_new import model
import random
from sklearn.model_selection import train_test_split, KFold
import torch.optim as optim
from sklearn import metrics
import math, copy, time
import source_model
import xlwt
from utils import save_model_with_checks, setup_seed, save_xlsx, count_parameters
from scipy.special import softmax


# 定义蒸馏损失
class KnowledgeDistillationLoss(nn.Module):
    def __init__(self, temperature=2.0, alpha=0.7):
        super(KnowledgeDistillationLoss, self).__init__()
        self.temperature = temperature
        self.alpha = alpha

    def forward(self, student_output, teacher_output):
        # 使用KL散度计算蒸馏损失
        soft_student = F.log_softmax(student_output / self.temperature, dim=1)
        soft_teacher = F.softmax(teacher_output / self.temperature, dim=1)
        distillation_loss = F.kl_div(soft_student, soft_teacher, reduction='batchmean') * (self.temperature ** 2)
        return distillation_loss


class LateFusionDistillationModel(nn.Module):
    def __init__(self, model, student_model, teacher_models,num_sources,weights=None):
        """
        初始化Late Fusion Distillation Model。

        model: 主模型，baseline模型
        prompt1: 提示1
        student_model: 目标领域的学生模型
        teacher_models: 包含多个教师模型的列表
        weights: 教师模型的融合权重列表（可选，默认为均匀权重）
        """
        super(LateFusionDistillationModel, self).__init__()
        self.model = model
        self.student_model = student_model
        self.teacher_models = teacher_models  # 包含多个教师模型的列表

        # 如果没有提供权重，则默认为均匀权重
        if weights is None:
            self.weights = [1.0 / num_sources] * num_sources
        else:
            self.weights = weights

        self.criterion = KnowledgeDistillationLoss()  # 定义蒸馏损失

    def forward(self, x, src_mask):
        # 获取每个教师模型的预测结果
        teacher_preds = []
        for i, teacher_model in enumerate(self.teacher_models):
            pred = self.model(x, src_mask, teacher_model)
            teacher_preds.append(pred)

        # 融合所有教师模型的预测
        fused_pred = sum(self.weights[i] * teacher_preds[i] for i in range(len(teacher_preds)))

        # 获取学生模型的输出
        student_output = self.model(x, src_mask, self.student_model)

        # 计算蒸馏损失
        distillation_loss = self.criterion(student_output, fused_pred)

        return student_output, distillation_loss


def make_model(temperature=8, N=6, d_model1=72, d_model2=116, d_ff=2048, h1=8, dropout=0.5, device=0, num_prompts=3):
    """
    Helper: Construct a model from hyperparameters.
    该函数支持动态数量的 `prompt2_source`，如3个、10个或16个。
    """
    c = copy.deepcopy

    # 创建 `num_prompts` 个 Prompt2 对象
    prompt2_sources = [source_model.Prompt2(d_model1, temperature) for _ in range(num_prompts)]
    prompt2_target = source_model.Prompt2(d_model1, temperature)

    attn1 = source_model.MultiHeadedAttention(h1, d_model1, device)
    ff = source_model.PositionwiseFeedForward(d_model1, d_ff, dropout)
    position = source_model.PositionalEncoding(d_model1, dropout)
    model = source_model.EncoderDecoder(
        source_model.Encoder(source_model.EncoderLayer(d_model1, d_model2, c(attn1), c(ff), dropout), N),
        c(position), d_model1, d_model2)

    # 初始化参数，使用Glorot / fan_avg初始化
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)

    return model, prompt2_sources, prompt2_target


# 定义站点映射函数，根据目标站点生成源站点列表
def generate_site_mapping(target_site, num_sources):

        # 假设源站点与目标站点不同，且源站点数目有限制
        if target_site == "NYU":
            all_sites = [
                "UCLA", "MaxMun", "Trinity", "Caltech", "KKI",
                "SDSU", "SBL", "Pitt", "Olin", "Yale",
                "Stanford", "USM", "Leuven", "CMU", "UM"
            ]#第一种
            # all_sites =[
            #     "UCLA", "Caltech", "Trinity", "MaxMun", "SDSU", "KKI", "Pitt", "SBL",
            #     "Stanford", "USM", "Yale", "Olin", "Leuven", "CMU", "UM"
            # ]#第二种    1.修改
            available_sites = [site for site in all_sites if site != "NYU"]

        # 根据num_sources选择源站点
        source_sites = available_sites[:num_sources]
        return source_sites

site1='NYU'
num_sources = 3  # 3.修改  1-15
#目标域的数据--NYU
X = np.load(f'./data/correlation//{site1}_X1.npy')
X_mask = np.load(f'./data/correlation//{site1}_X1_mask.npy')
Y = np.load(f'./data/correlation//{site1}_Y1.npy')

best_epoch=50  # Hyperparameter: needs to be adjusted
seed = 123
setup_seed(seed)
epochs = 50
batch_size = 16
drop_out = 0.5
lr =0.000051
weight_decay = 0.1
result = []
loss_all_5=[]
acc_list = []
precision_list = []
recall_list = []
f1_list = []
auc_list = []
epoch_list = []
sen_list = []
spe_list = []
max_acc = 0
max_precision = 0
max_recall = 0
max_f1 = 0
max_auc = 0
max_epoch = 0
training_time_result=[]
kf = KFold(n_splits=5, random_state=seed, shuffle=True)
kfold_index = 0
for train_index, test_index in kf.split(X):
    loss_all = []
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
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


    # 1.定义模型


    # model #N=6, d_model1=72, d_model2=116, d_ff=2048, h1=8, h2=10
    model,prompt2_sources,prompt2_target = make_model(temperature=8, N=1, d_model1=6670, d_model2=25, d_ff=128, h1=2,
                                         dropout=drop_out, device=device, num_prompts=num_sources)

    model.load_state_dict(torch.load('./baseline_model/' + str(site1) +'/15_site/'+ str(kfold_index) + '.pt'), strict=False)


    #2.目标站点对应的top k个源站点--导入参数
    source_sites = generate_site_mapping(site1, num_sources)

    # 加载每个源站点的模型权重并赋值给对应的 prompt2_source
    for i, site in enumerate(source_sites, start=1):
        # 加载源站点的模型权重
        state_dict = torch.load(f'./source_model/{site1}/{site}/{kfold_index}.pt')

        # print(len(prompt2_sources))
        # print('i',i)
        # 根据当前的 i 值选择对应的 prompt2_source
        prompt_source = prompt2_sources[i-1]

        # 将加载的权重赋值到对应的 prompt_source 模块中---未定义，之后在定义
        prompt_source.attn_W_down.weight.data = state_dict['attn_W_down.weight']
        prompt_source.attn_W_up.weight.data = state_dict['attn_W_up.weight']
        prompt_source.layer_norm.weight.data = state_dict['layer_norm.weight']
        prompt_source.layer_norm.bias.data = state_dict['layer_norm.bias']


    model.to(device)
    for prompt_source in prompt2_sources:
        prompt_source.to(device)
    prompt2_target.to(device)
    prompt2_target.to(device)

    # 3.定义每个站点数的权重数据--导入权重

    weights = [
        0.9444, 0.9254, 0.928, 0.9374, 0.9223,
        0.9111, 0.9105, 0.9037, 0.8442, 0.8729,
        0.7963, 0.7941, 0.7982, 0.7596, 0.709
    ] #--cos相似度 第一种

    # weights = [
    #     0.8593, 0.8388, 0.8298, 0.8231, 0.824,
    #     0.8159, 0.8147, 0.8122, 0.7212, 0.7178,
    #     0.7866, 0.7451, 0.7307, 0.7008, 0.6953
    # ]#--cos相似度 第二种 #2.修改

    # 获取top_k的余弦相似度值
    top_k_values = sorted(weights, reverse=True)[:num_sources]
    print(f"Top {num_sources} Cosine Similarity Values: {top_k_values}")

    # 使用 scipy.special.softmax 计算 softmax
    softmax_values = softmax(np.array(top_k_values))

    # 输出 softmax 结果
    print(f"\nSoftmax values (probabilities) for top {num_sources}: {np.round(softmax_values, 5)}")


    selected_weights = softmax_values

    # 输出选择的权重
    print(f"Selected weights for {num_sources} sources: {selected_weights}")

    # 创建融合模型（使用动态数量的 prompt2_source）  model, student_model, teacher_models,num_sources,weights=None
    fusion_model = LateFusionDistillationModel(model, prompt2_target,prompt2_sources,num_sources,selected_weights)
    fusion_model.to(device)


    # 假设 fusion_model.student_model 是你的 student_model,weights=selected_weights
    student_model_params = fusion_model.student_model.parameters()
    # 定义优化器
    optimizer = optim.Adam(student_model_params, lr=0.001,weight_decay=0.01)


    total_loss = 0
    correct = 0
    total = 0
    # train
    for epoch in range(0, epochs + 1):
            fusion_model.train()
            idx_batchs = np.random.permutation(int(X_train.shape[0]))
            for i in range(0, int(X_train.shape[0]) // int(batch_size)):
                idx_batch = idx_batchs[i * int(batch_size):min((i + 1) * int(batch_size), X_train.shape[0])]

                train_data_batch = X_train[idx_batch]
                X_mask_train_batch = X_mask_train[idx_batch]
                train_label_batch = Y_train[idx_batch]

                train_data_batch = torch.from_numpy(train_data_batch).float().to(device)
                X_mask_train_batch = torch.from_numpy(X_mask_train_batch).to(device)
                train_label_batch = torch.from_numpy(train_label_batch).long()

                optimizer.zero_grad()
                # 获取学生模型的输出和蒸馏损失
                student_output, distillation_loss = fusion_model(train_data_batch, src_mask=X_mask_train_batch)
                # 反向传播
                distillation_loss.backward()
                optimizer.step()

            count = 0
            acc = 0
            if epoch % 5 == 0:
                count = 0
                acc = 0
                for i in range(0, int(X_train.shape[0]) // int(batch_size)):
                    idx_batch = idx_batchs[i * int(batch_size):min((i + 1) * int(batch_size), X_train.shape[0])]
                    train_data_batch = X_train[idx_batch]
                    train_label_batch = Y_train[idx_batch]
                    X_mask_train_batch = X_mask_train[idx_batch]

                    X_mask_train_batch = torch.from_numpy(X_mask_train_batch).to(device)
                    train_data_batch = torch.from_numpy(train_data_batch).float().to(device)
                    train_label_batch = torch.from_numpy(train_label_batch).long()

                    # 获取学生模型的输出和蒸馏损失
                    student_output, distillation_loss = fusion_model(train_data_batch, src_mask=X_mask_train_batch)

                    _, indices = torch.max(student_output, dim=1)
                    preds = indices.cpu()
                    acc += metrics.accuracy_score(preds, train_label_batch)
                    count = count + 1
                # 训练的loss输出可能有误
                print('train\tepoch: %d\tloss: %.4f\t\tacc: %.4f' % (epoch, distillation_loss.item(), acc / count))
                loss_all.append(distillation_loss.data.item())

            if epoch % 5 == 0:
                fusion_model.eval()
                X_mask_test_batch_dev = torch.from_numpy(X_mask_test).to(device)
                test_data_batch_dev = torch.from_numpy(X_test).float().to(device)
                # 获取学生模型的输出和蒸馏损失
                student_output, distillation_loss = fusion_model(test_data_batch_dev, src_mask=X_mask_test_batch_dev)

                _, indices = torch.max(student_output, dim=1)
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

            if epoch == best_epoch:

               save_path='./source_model_1/' + site1 + '/' + str(num_sources) + '/' + str(kfold_index) + '.pt'
               save_model_with_checks(fusion_model.student_model, save_path)


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

print('{}-{}-{}-{}-{:.4}-{:.4}-{:.4}-{:.4}-{:.4}\n'.format(max_epoch, lr, batch_size, weight_decay, max_acc,
                                                               max_precision,
                                                               max_recall, max_f1, max_auc))

# 保存结果
save_xlsx(ts_result, 'model', site1, max_epoch, lr, batch_size, weight_decay)
