#保存每个站点的训练集结果--已完成
#根据聚类--求cos  cluster_cosine_calculation


import numpy as np
from scipy.spatial.distance import cosine
from sklearn.cluster import KMeans

from scipy.special import softmax

# 聚类数量
n_clusters = 1  # 我们只聚为一类

# 计算两个质心之间的余弦相似度
def compute_cosine_similarity(centroid1, centroid2):
    similarity = 1 - cosine(centroid1, centroid2)  # 余弦相似度
    return similarity
# 定义处理单个文件的函数
def process_file(file_name, n_clusters=1):
    # 导入数据
    data = np.load(file_name)  # 读取文件 (78, 25, 6670)

    # 数据重塑：将每个样本展平成一维向量 (78, 25 * 6670)
    reshaped_data = data.reshape(data.shape[0], -1)  # 变为 (78, 25 * 6670)

    # 使用KMeans进行聚类
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    kmeans.fit(reshaped_data)

    # 获取质心
    centroids = kmeans.cluster_centers_

    return centroids
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

# 导入并处理每个文件，获取质心
USM_X_centroids = process_file('./data/similarity/USM_X.npy')
UCLA_X_centroids = process_file('./data/similarity/UCLA_X.npy')
UM_X_centroids = process_file('./data/similarity/UM_X.npy')
NYU_X_centroids = process_file('./data/similarity/NYU_X.npy')
KKI_X_centroids = process_file('./data/similarity/KKI_X.npy')
Leuven_X_centroids = process_file('./data/similarity/Leuven_X.npy')
MaxMun_X_centroids = process_file('./data/similarity/MaxMun_X.npy')
Pitt_X_centroids = process_file('./data/similarity/Pitt_X.npy')
Trinity_X_centroids = process_file('./data/similarity/Trinity_X.npy')
Yale_X_centroids = process_file('./data/similarity/Yale_X.npy')
Caltech_X_centroids = process_file('./data/similarity/Caltech_X.npy')
CMU_X_centroids = process_file('./data/similarity/CMU_X.npy')
Olin_X_centroids = process_file('./data/similarity/Olin_X.npy')
SBL_X_centroids = process_file('./data/similarity/SBL_X.npy')
SDSU_X_centroids = process_file('./data/similarity/SDSU_X.npy')
Stanford_X_centroids = process_file('./data/similarity/Stanford_X.npy')

# 站点名称列表
sites = [
    'USM', 'UCLA', 'UM', 'NYU',
    'KKI', 'Leuven', 'MaxMun', 'Pitt',
    'Trinity', 'Yale', 'Caltech', 'CMU',
    'Olin', 'SBL', 'SDSU', 'Stanford'
]

# 将站点与质心对应
site_avg = {
    'USM': USM_X_centroids,
    'UCLA': UCLA_X_centroids,
    'UM': UM_X_centroids,
    'NYU': NYU_X_centroids,
    'KKI': KKI_X_centroids,
    'Leuven': Leuven_X_centroids,
    'MaxMun': MaxMun_X_centroids,
    'Pitt': Pitt_X_centroids,
    'Trinity': Trinity_X_centroids,
    'Yale': Yale_X_centroids,
    'Caltech': Caltech_X_centroids,
    'CMU': CMU_X_centroids,
    'Olin': Olin_X_centroids,
    'SBL': SBL_X_centroids,
    'SDSU': SDSU_X_centroids,
    'Stanford': Stanford_X_centroids
}

import numpy as np
from scipy.spatial.distance import cosine
from scipy.special import softmax


# 定义计算余弦相似度并进行 softmax 的函数
def calculate_cosine_similarity_and_softmax(target_site, site_avg, sites, top_k=3):
    # 存储余弦相似度
    cos_sim_values = []

    # 计算余弦相似度并保存
    for site in sites:
        if site != target_site:  # 排除目标站点自己
            cos_sim = 1 - cosine(site_avg[target_site].flatten(), site_avg[site].flatten())
            cos_sim_values.append(cos_sim)
            print(f"Cosine Similarity between {target_site} and {site}: {cos_sim}")

    # 获取top_k的余弦相似度值
    top_k_values = sorted(cos_sim_values, reverse=True)[:top_k]
    print(f"Top {top_k} Cosine Similarity Values: {top_k_values}")

    # 使用 scipy.special.softmax 计算 softmax
    softmax_values = softmax(np.array(top_k_values))

    # 输出 softmax 结果
    print(f"\nSoftmax values (probabilities) for top {top_k}: {np.round(softmax_values, 5)}")

    return cos_sim_values, softmax_values, top_k_values


# 调用函数，设置 top_k=3
cos_sim_values, softmax_values, top_k_values = calculate_cosine_similarity_and_softmax('NYU', site_avg, sites, top_k=3)





