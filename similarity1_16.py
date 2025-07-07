#保存每个站点的训练集结果--已完成
#平均--求cos  average_cos_calculation_16


from scipy.special import softmax
import numpy as np
from scipy.spatial.distance import cosine

import numpy as np
from scipy.spatial.distance import cosine



import numpy as np

# 假设加载了16个站点的数据
KKI_X = np.load('./data/similarity/KKI_X.npy')
Leuven_X = np.load('./data/similarity/Leuven_X.npy')
MaxMun_X = np.load('./data/similarity/MaxMun_X.npy')
Pitt_X = np.load('./data/similarity/Pitt_X.npy')
Trinity_X = np.load('./data/similarity/Trinity_X.npy')
Yale_X = np.load('./data/similarity/Yale_X.npy')
UM_X = np.load('./data/similarity/UM_X.npy')
UCLA_X = np.load('./data/similarity/UCLA_X.npy')
USM_X = np.load('./data/similarity/USM_X.npy')
NYU_X = np.load('./data/similarity/NYU_X.npy')
Caltech_X = np.load('./data/similarity/Caltech_X.npy')
CMU_X = np.load('./data/similarity/CMU_X.npy')
Olin_X = np.load('./data/similarity/Olin_X.npy')
SBL_X = np.load('./data/similarity/SBL_X.npy')
SDSU_X = np.load('./data/similarity/SDSU_X.npy')
Stanford_X = np.load('./data/similarity/Stanford_X.npy')

# 设置样本数
num_samples = 110


# 计算每个站点的平均值并存储
KKI_X_avg = np.mean(KKI_X[:num_samples], axis=0)
KKI_X_avg_final = np.mean(KKI_X_avg, axis=0)

Leuven_X_avg = np.mean(Leuven_X[:num_samples], axis=0)
Leuven_X_avg_final = np.mean(Leuven_X_avg, axis=0)

MaxMun_X_avg = np.mean(MaxMun_X[:num_samples], axis=0)
MaxMun_X_avg_final = np.mean(MaxMun_X_avg, axis=0)

Pitt_X_avg = np.mean(Pitt_X[:num_samples], axis=0)
Pitt_X_avg_final = np.mean(Pitt_X_avg, axis=0)

Trinity_X_avg = np.mean(Trinity_X[:num_samples], axis=0)
Trinity_X_avg_final = np.mean(Trinity_X_avg, axis=0)

Yale_X_avg = np.mean(Yale_X[:num_samples], axis=0)
Yale_X_avg_final = np.mean(Yale_X_avg, axis=0)

UM_X_avg = np.mean(UM_X[:num_samples], axis=0)
UM_X_avg_final = np.mean(UM_X_avg, axis=0)

UCLA_X_avg = np.mean(UCLA_X[:num_samples], axis=0)
UCLA_X_avg_final = np.mean(UCLA_X_avg, axis=0)

USM_X_avg = np.mean(USM_X[:num_samples], axis=0)
USM_X_avg_final = np.mean(USM_X_avg, axis=0)

NYU_X_avg = np.mean(NYU_X[:num_samples], axis=0)
NYU_X_avg_final = np.mean(NYU_X_avg, axis=0)

Caltech_X_avg = np.mean(Caltech_X[:num_samples], axis=0)
Caltech_X_avg_final = np.mean(Caltech_X_avg, axis=0)

CMU_X_avg = np.mean(CMU_X[:num_samples], axis=0)
CMU_X_avg_final = np.mean(CMU_X_avg, axis=0)

Olin_X_avg = np.mean(Olin_X[:num_samples], axis=0)
Olin_X_avg_final = np.mean(Olin_X_avg, axis=0)

SBL_X_avg = np.mean(SBL_X[:num_samples], axis=0)
SBL_X_avg_final = np.mean(SBL_X_avg, axis=0)

SDSU_X_avg = np.mean(SDSU_X[:num_samples], axis=0)
SDSU_X_avg_final = np.mean(SDSU_X_avg, axis=0)

Stanford_X_avg = np.mean(Stanford_X[:num_samples], axis=0)
Stanford_X_avg_final = np.mean(Stanford_X_avg, axis=0)


# 站点名称列表
sites = ['KKI', 'Leuven', 'MaxMun', 'Pitt', 'Trinity', 'Yale', 'UM', 'UCLA', 'USM', 'NYU',
         'Caltech', 'CMU', 'Olin', 'SBL', 'SDSU', 'Stanford']

# 站点平均值字典
site_avg = {
    'KKI': KKI_X_avg_final,
    'Leuven': Leuven_X_avg_final,
    'MaxMun': MaxMun_X_avg_final,
    'Pitt': Pitt_X_avg_final,
    'Trinity': Trinity_X_avg_final,
    'Yale': Yale_X_avg_final,
    'UM': UM_X_avg_final,
    'UCLA': UCLA_X_avg_final,
    'USM': USM_X_avg_final,
    'NYU': NYU_X_avg_final,
    'Caltech': Caltech_X_avg_final,
    'CMU': CMU_X_avg_final,
    'Olin': Olin_X_avg_final,
    'SBL': SBL_X_avg_final,
    'SDSU': SDSU_X_avg_final,
    'Stanford': Stanford_X_avg_final
}

# 定义计算余弦相似度并进行softmax的函数
def calculate_top_k_similarity(site_avg, target_site, sites, top_k=3):
    # 存储余弦相似度
    cos_sim_values = []

    # 计算余弦相似度并保存
    for site in sites:
        if site != target_site:  # 排除目标站点自己
            cos_sim = 1 - cosine(site_avg[target_site], site_avg[site])
            cos_sim_values.append(cos_sim)
            print(f"Cosine Similarity between {target_site} and {site}: {cos_sim}")

    # 获取top_k的余弦相似度值
    top_k_values = sorted(cos_sim_values, reverse=True)[:top_k]
    print(f"Top {top_k} Cosine Similarity Values: {top_k_values}")

    # 使用 scipy.special.softmax 计算 softmax
    softmax_values = softmax(np.array(top_k_values))

    # 输出 softmax 结果
    print(f"\nSoftmax values (probabilities) for top {top_k}: {np.round(softmax_values, 5)}")

    return top_k_values, softmax_values

# 调用函数，设置top_k=3  top_k为所需要修改的参数
top_k_values, softmax_values = calculate_top_k_similarity(site_avg, 'NYU', sites, top_k=3)



