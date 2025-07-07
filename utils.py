# utils.py
import os
import torch
import numpy as np
import random
import xlwt
import time

def save_model_with_checks(model, save_path):
    save_dir = os.path.dirname(save_path)
    os.makedirs(save_dir, exist_ok=True)
    torch.save(model.state_dict(), save_path)
    print(f"Model saved at {save_path}")

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def save_xlsx(ts_result, model, site, max_epoch, lr, batch_size, weight_decay,num_sources):
    runtime_id = './result/{}/{}/{}-{}_site_result-{}-{}-{}-{}-{}'.format(
        site, num_sources, model, site, max_epoch, lr, batch_size, weight_decay,
        time.strftime('%Y-%m-%d %H-%M-%S', time.localtime())
    )
    save_dir = os.path.dirname(runtime_id)
    os.makedirs(save_dir, exist_ok=True)

    f = xlwt.Workbook('encoding = utf-8')
    sheet1 = f.add_sheet('sheet1', cell_overwrite_ok=True)
    a = np.average([ts_result[1], ts_result[2], ts_result[3], ts_result[4], ts_result[5]], axis=0).tolist()
    a[0] = 'average'
    ts_result.append(a)
    for j in range(len(ts_result)):
        for i in range(len(ts_result[j])):
            sheet1.write(j, i, ts_result[j][i])
    f.save(runtime_id + '.xlsx')

def count_parameters(model):
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_params, trainable_params
