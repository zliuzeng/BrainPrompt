#ABIDE1 自闭症数据集 下载
from nilearn import datasets

# quality_checked的选择影响数据集的数量
# quality_checked = True下载后是871个数据，quality_checked = False则是1035个数据。

abide_dataset = datasets.fetch_abide_pcp('ABIDE-871/',
                                 derivatives=['rois_aal'],
                                 pipeline='cpac',
                                 band_pass_filtering=True,
                                 global_signal_regression=True,
                                 quality_checked=False)
