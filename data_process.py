import warnings
import pandas as pd
import os
import tqdm
import numpy as np
import scipy.io as sio
warnings.filterwarnings("ignore")
import pickle
import torch
def get_key(file_name):
    file_name = file_name.split('_')
    key = ''
    for i in range(len(file_name)):
        if file_name[i] == 'rois':
            key = key[:-1]
            break
        else:
            key += file_name[i]
            key += '_'
    return key
def load_data(path1, path2):
    all = {}
    labels = {}
    all_data = []
    label = []
    for filename in path1:
        a = np.loadtxt(filename)
        a = a.transpose()
        all[filename] = a
        all_data.append(a)
        data = pd.read_csv(path2)
        for i in range(len(data)):
            if os.path.basename(get_key(filename)) == data['FILE_ID'][i]:
                if int(data['DX_GROUP'][i]) == 2:
                    labels[filename] = int(data['DX_GROUP'][i]-1)
                    label.append(int(data['DX_GROUP'][i]-1))
                else:
                    labels[filename] = 0
                    label.append(0)
                break
    label = np.array(label)
    return all_data, label
def compute_correlation(matrix):
    num_slices, dim1,_  = matrix.shape
    correlation_matrices = np.zeros((num_slices, dim1, dim1))
    for i in range(num_slices):
        for j in range(dim1):
            correlation_matrices[i] = np.corrcoef(matrix[i][j], rowvar=False)
    return correlation_matrices
#设置滑动的大小和步长
def sliding_window(sample, window_size=100, step=10):
    """
    Apply sliding window on the second dimension of the sample.
    Discard the last part if it's smaller than the window size.
    """
    if sample.shape[1] < window_size:
        return None  # Discard the sample if it's too small for even one window

    windows = []
    for start in range(0, sample.shape[1] - window_size + 1, step):
        end = start + window_size
        windows.append(sample[:, start:end])

    # Handle the remaining part if it's larger than step
    # if sample.shape[1] % step != 0 and sample.shape[1] > len(windows) * step:
    #         last_start = sample.shape[1] - window_size
    #         windows.append(sample[:, last_start:])
    return windows

#超参数需要设置--设置填充到哪个位置
def process_samples(samples,lables):
    """
    Process a list of samples, applying sliding window and generating correlation matrices.
    """
    processed = []
    processed_labels=[]
    for i,sample in enumerate(samples):
        # print(sample.shape)
        windows = sliding_window(sample)
        if windows  is not None:
            correlation_matrices = np.array([np.corrcoef(w) for w in windows])
            # print(correlation_matrices.shape[0])
            # print(correlation_matrices.shape)
            # Pad with zeros if less than 25 slices  #超参数需要设置  28--时序最长为320--存在12个小站点中
            if correlation_matrices.shape[0] < 25:
                padded = np.zeros((25, correlation_matrices.shape[1], correlation_matrices.shape[2]))
                padded[:correlation_matrices.shape[0], :, :] = correlation_matrices
                correlation_matrices = padded
                # print(correlation_matrices.shape)
            # if correlation_matrices.shape[0] > 26:
            #     print('False')

            processed.append(correlation_matrices)
            processed_labels.append(lables[i])

    # Combine all processed samples into a single array
    return np.array(processed), np.array(processed_labels)

def get_filtered_files(directory, prefixes):
    """
    遍历目录并筛选以指定前缀开头的文件。

    :param directory: 要遍历的目录路径
    :param prefixes: 前缀列表（如 ['NYU', 'UCLA']）
    :return: 筛选文件的完整路径列表
    """
    filtered_files = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            # 检查文件是否以指定前缀开头
            if any(file.startswith(prefix) for prefix in prefixes):
                filtered_files.append(os.path.join(root, file))
    return filtered_files

#将x转换为上三角矩阵--四维矩阵-->三维矩阵
def triu_x(x):
    #确保 x 是上三角矩阵
    x = torch.triu(x)
    #初始化一个空的 tensor 用于存储结果
    result = torch.zeros((x.size(0),x.size(1),(x.size(2)*(x.size(3)-1))//2))
    # 遍历每个矩阵，提取非零元素
    for i in range(x.size(0)):
        for j in range(x.size(1)):
            #获取当前矩阵的非零元素
            non_zero_elements = x[i, j][x[i, j] != 0]
            #获取当前矩阵的不为1的元素
            non_zero_elements = non_zero_elements[non_zero_elements != 1]
            #获取当前矩阵的前6670个元素
            selected_elements = non_zero_elements[:((x.size(2)*(x.size(3)-1))//2)]
            result[i, j, :selected_elements.numel()] = selected_elements
    return result
#找到mask的操作
def find_padded_parts(matrix):
    """
    For a given 4D matrix, determine which slices in the second dimension are padded.
    Return a boolean matrix of shape (140, 25, 25) indicating original (True) or padded (False) parts.
    """
    original = np.any(matrix != 0, axis=(2, 3))  # Check for non-zero values in the last two dimensions
    padded_matrix = np.repeat(original[:, :, np.newaxis], 25, axis=2)
    padded_matrix = torch.from_numpy(padded_matrix)
    return padded_matrix

#文件选择 17个站点  NYU UM USM UCLA
#  ['KKI','Leuven','MaxMun','Pitt','Trinity','Yale',"UM", "UCLA", "USM", "NYU",'Caltech','CMU','OHSU','Olin','SBL','SDSU','Stanford'] "UM", "UCLA", "USM", "NYU"

# prefixes = ['KKI','Leuven','MaxMun','Pitt','Trinity','Yale','Caltech','CMU','OHSU','Olin','SBL','SDSU','Stanford'] #12个站点
# i='12_site'


# prefixes = ['NYU']
# i='NYU'

# prefixes = ['UM']
# i='UM'

# prefixes = ['USM']
# i='USM'

#prefixes = ['UCLA']
#i='UCLA'

# "NYU", "UCLA"
#16个站点的数据分析
prefixes = ["NYU","UM", "USM",'KKI','Leuven','MaxMun','Pitt','Trinity','Yale','Caltech','CMU','Olin','SBL','SDSU','Stanford']
print(len(prefixes))#应该为 15
i="UCLA"


# 设置文件夹路径和前缀
directory = ".//ABIDE-1035/ABIDE_pcp/cpac/filt_global"  # 文件夹路径
#1.获取筛选符合条件的站点
filtered_files = get_filtered_files(directory, prefixes)
#站点数据+路径
print(f"Found {len(filtered_files)} files with prefixes {prefixes}")
label_path = './/ABIDE-1035/ABIDE_pcp/Phenotypic_V1_0b_preprocessed1.csv'
#2.导入数据  #raw_data:列表，列表里面是np.array的单个样本地址    labels：标签
raw_data, labels = load_data(filtered_files, label_path)
#3.划分时间窗+计算相关性pcc
transformed_data,process_lables=process_samples(raw_data,labels)
#4.去除nan,inf
transformed_data=np.nan_to_num(transformed_data, nan=0, posinf=1, neginf=1)
print('transformed_data',transformed_data.shape)
# print('process_lables',process_lables)
#5.维度转化
#将numpy的x转换为torch
X = torch.from_numpy(transformed_data).float()
#保存mask
mask = find_padded_parts(transformed_data)

#矩阵维度：(392, 86, 200, 200)-->(392, 86, X)
X1=triu_x(X)
numpy_array = X1.numpy()
print('numpy_array',numpy_array.shape)


# 目标目录
directory = './data/correlation/'+ str(i) + '/'

# 检查目录是否存在，如果不存在则创建
if not os.path.exists(directory):
    os.makedirs(directory)
#6.保存文件
np.save('./data/correlation/' + str(i) + '/' + str(i) +  '_15_site_X1.npy', numpy_array)
np.save('./data/correlation/' + str(i) + '/' + str(i) +  '_15_site_X1_mask.npy', mask)
np.save('./data/correlation/' + str(i) + '/' + str(i) +  '_15_site_Y1.npy', process_lables)



