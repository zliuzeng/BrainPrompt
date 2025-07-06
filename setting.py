import argparse


def get_args():
    parser = argparse.ArgumentParser(description="HGnn")
    parser.add_argument("--gpu", type=str, default=0, help='gpu设备号')
    parser.add_argument("--seed", type=int, default=123)

    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--kFold", type=int, default=5)
    parser.add_argument("--epoch_cf", type=int, default=200)
    #保存模型的选择
    parser.add_argument("--best_epoch", type=int, default=5)
    #0:不保存  1：保存
    parser.add_argument("--save_model", type=int, default=0)

    #选择模型 num_sources
    parser.add_argument('--model', type=str, choices=['target_model','source_model', 'baseline_model'], default='target_model',
                        help="Choose the model to use: 'target_model','source_model' or 'baseline'")
    #选择不同的数据集
    parser.add_argument('--site', type=str, choices=['NYU', 'UCLA', 'UM', 'USM'], default='NYU',
                        help="Choose the site to use:'NYU', 'UCLA', 'UM', 'USM' ")

    #选择top k源域的类目标提示
    parser.add_argument('--num_sources', type=str, choices=['15', '10', '3'],default='3',
                        help="Choose the num_sources to use: '15', '10', '3'")

    # 选择哪个源域进行source prompt训练  --只在source_model训练的时候有用
    parser.add_argument('--site_source', type=str, choices=['NYU', 'UCLA', 'UM', 'USM'], default='SBL',
                        help=["NYU","UM",'UCLA', "USM",'KKI','Leuven','MaxMun','Pitt','Trinity','Yale','Caltech','CMU','Olin','SBL','SDSU','Stanford'])
    # 重要超参
    parser.add_argument("--lr", type=float, default=0.0001)
    parser.add_argument("--decay", type=float, default=0.1)

    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument('--temperature', type=int,default=8)


    #目标域 lr
    parser.add_argument("--prompt_lr", type=float, default=0.0000851)
    parser.add_argument("--model_lr", type=float, default=0.0001)


    args = parser.parse_args()

    return args
