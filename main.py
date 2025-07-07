import setting
import optuna
import test
import os
args = setting.get_args()
import baseline_train
import source1_train
import target_train
import source2_train
import source12_train

def objective(trial):
    # 建议超参数值  model_lr
    lr = trial.suggest_categorical("lr", [0.0000851,0.00001,0.00002,0.00003,0.00004,0.00006,0.0008,0.0006,0.0004,0.0002,0.001,0.003,0.005,0.008,0.1,0.12,0.14,0.16,0.18,0.2,0.25,0.3, 0.5])


    weight_decay = trial.suggest_categorical("weight_decay", [0.55,0.5,0.45,0.4,0.35,0.3,0.25,0.2,0.15,0.1,0.05,0.01])

    #target中调节的参数
    temperature = trial.suggest_categorical('temperature', [10, 9, 8, 7, 6, 5, 4, 3, 2, 1])

    if args.model=='baseline_model': #训练baseline模型
        # 选择不同的数据集
        path_list = f'./data/correlation//{args.site}//{args.site}_15_site_X1.npy'
        path_list_mask = f'./data/correlation//{args.site}//{args.site}_15_site_X1_mask.npy'
        path_label = f'./data/correlation//{args.site}//{args.site}_15_site_Y1.npy'

        print('path_label',path_label)
        # 训练 baseline模型
        acc = baseline_train.train_and_test_baseline_model(args, path_list, path_list_mask, path_label, lr,
                                                           weight_decay)
    elif args.model=='source_model1':#训练mask prompt
        # 选择不同的数据集
        path_list = f'./data/correlation//{args.site_source}_X1.npy'
        path_list_mask = f'./data/correlation//{args.site_source}_X1_mask.npy'
        path_label = f'./data/correlation//{args.site_source}_Y1.npy'

        print('path_label', path_label)
        # 训练 source_model
        acc = source1_train.train_and_test_source_train_model(args, path_list, path_list_mask, path_label, lr,
                                                           weight_decay)
    elif args.model=='source_model2':  #训练lora微调
        # 选择不同的数据集
        path_list = f'./data/correlation//{args.site_source}_X1.npy'
        path_list_mask = f'./data/correlation//{args.site_source}_X1_mask.npy'
        path_label = f'./data/correlation//{args.site_source}_Y1.npy'

        print('path_label', path_label)
        # 训练 source_model
        acc = source2_train.train_and_test_source_train_model(args, path_list, path_list_mask, path_label, lr,
                                                           weight_decay)
    elif args.model=='source_model12': #训练mask prompt+lora微调
        # 选择不同的数据集
        path_list = f'./data/correlation//{args.site_source}_X1.npy'
        path_list_mask = f'./data/correlation//{args.site_source}_X1_mask.npy'
        path_label = f'./data/correlation//{args.site_source}_Y1.npy'

        print('path_label', path_label)
        # 训练 source_model
        acc = source12_train.train_and_test_source_train_model(args, path_list, path_list_mask, path_label, lr,
                                                               weight_decay)
    elif args.model=='target_model':  #训练mask prompt+instance p
        # 选择不同的数据集
        path_list = f'./data/correlation//{args.site}_X1.npy'
        path_list_mask = f'./data/correlation//{args.site}_X1_mask.npy'
        path_label = f'./data/correlation//{args.site}_Y1.npy'

        print('path_label', path_label)

        prompt_lr = trial.suggest_categorical("prompt_lr",
                                       [0.0000851, 0.00001, 0.00002, 0.00003, 0.00004, 0.00006, 0.0008, 0.0006, 0.0004,
                                        0.0002, 0.001, 0.003, 0.005, 0.008])


        # 训练 target_model
        acc = target_train.train_and_test_target_train_model(args, path_list, path_list_mask, path_label, lr,prompt_lr,
                                                                       weight_decay,temperature)
        # acc = target_train_no_combin.train_and_test_target_train_model(args, path_list, path_list_mask, path_label, lr,
        #                                                                weight_decay)


    # acc = train.train_and_test(args, path_list,path_list_mask, path_label, lr,weight_decay,temperature)
    # acc = test.train_and_test(args, path_list, path_list_mask, path_label, lr, weight_decay, A, temperature)

    return acc
if __name__ == '__main__':
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=200)
    print(study.best_params)
