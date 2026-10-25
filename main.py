import argparse
import os
import numpy as np
from datetime import datetime
import torch
import random
from train_test_pre import Detector
from utils_pyg import read_json, Dataset
import warnings
from test import Tester


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True

warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser()
parser.add_argument('--trials', type=int, default=5)
parser.add_argument('--model', type=str, default='OFA_GLAD')
parser.add_argument('--shot', type=int, default=10)
parser.add_argument('--json_dir', type=str, default='./params')
parser.add_argument('--path', type=str, default='OFA_GLAD')
parser.add_argument('--device', type=str, default='cuda:0')
parser.add_argument('--datadir', type=str, default ='dataset', help='Directory where benchmark is located')
parser.add_argument('--max-nodes', type=int, default=0,
                    help='Maximum number of nodes (ignore graghs with nodes exceeding the number.')
parser.add_argument('--batch_size', default=500, type=int, help='Batch size.')
parser.add_argument('--batch_size_test', default=500, type=int, help='Batch size of test.')
parser.add_argument('--input_dim', type=int, default=32)

args = parser.parse_known_args()[0]
print(args)

datasets_test = ['AIDS', 'BZR', 'COX2', 'IMDB-BINARY', 'REDDIT-BINARY', 'PROTEINS_full', 'DD', 'Tox21_HSE',
                 'Tox21_MMP', 'Tox21_PPAR-gamma','Tox21_ER']
datasets_train = ['Tox21_p53', 'NCI1', 'COLLAB', 'ENZYMES']

model = args.model
model_result = {'name': model}

work_dir = 'results/' + '/' + args.model + '/' + args.path + '/' + 'shot' + str(args.shot) + '/' + datetime.now().strftime('%d_%B_%Y_%Hh') + '/'


train_config = {
    'device': args.device,
    'epochs_pre': 20,
    'epochs': 61,
    'testdsets': datasets_test,
    'traindsets': datasets_train,
    'model_path': os.path.join(work_dir, 'snapshots/'),

# 新增
    'few_seed': args.few_seed,
    'anomaly': args.anomaly,
}
print('model_path:', train_config['model_path'])
os.makedirs(train_config['model_path'], exist_ok=True)
result_path = os.path.join(work_dir, 'results/')
os.makedirs(result_path, exist_ok=True)

save_dir = os.path.join(work_dir, 'error/')
os.makedirs(save_dir, exist_ok=True)

train_dataset_name = 'data_train_loader_tr4_shot{}.pt'.format(args.shot)
test_dataset_name = 'data_test_loader_tr4_shot{}.pt'.format(args.shot)

data_train_loader = [Dataset(name, args, is_train=True) for name in datasets_train]
data_test_loader = [Dataset(name, args, is_train=False) for name in datasets_test]

output_dir = './Dataloader/'
os.makedirs(output_dir, exist_ok=True)
torch.save(data_train_loader, os.path.join(output_dir, train_dataset_name))
torch.save(data_test_loader, os.path.join(output_dir, test_dataset_name))
print("Data loaders '{}' have been saved successfully.".format(train_dataset_name))

# 加载已保存的预处理数据
output_dir = './Dataloader/'
data_train_loader = torch.load(os.path.join(output_dir, train_dataset_name))
data_test_loader = torch.load(os.path.join(output_dir, test_dataset_name))
print("Data loaders '{}' have been loaded successfully.".format(train_dataset_name))

model_config = read_json(model, args.shot, args.json_dir)

if model_config is None:
    model_config = {
        "model": "OFA_GLAD",
        "lr1": 5e-4,
        "lr2": 5e-4,
        "wd1": 5e-5,
        "wd2": 5e-5,
        "input_dim": args.input_dim,
        "hidden_dim": 256,
        "embedding_dim": 256,
        "prompt_dim": 256,
        "num_layers": 3,
        "num_prompt": args.shot,
        "bn": False,
        "norm": False,
        "ln": False,
        "bias": False,
        "drop_1": 0.3,
        "drop_2": 0.3,
        "lammda": 1,
        "alpha": 1,
        "beta": 1,
        }
    print('use default model config')
    print(model_config)
else:
    print('use saved best model config')
    print(model_config)

# Initialize dictionaries to store scores for each test dataset
auc_dict, ap_dict, pre_dict, rec_dict = {}, {}, {}, {}
for t in range(args.trials):
    seed = t
    set_seed(seed)
    print("Model {}, Trial {}".format(model, seed))
    train_config['seed'] = seed
    data = {'train': data_train_loader, 'test': data_test_loader}

    detector = Detector(train_config, model_config, data, t)
    detector.pretrain()
    test_score_list = detector.train()

    # Aggregate scores for each test dataset
    for test_data_name, test_score in test_score_list.items():
        if test_data_name not in auc_dict:
            auc_dict[test_data_name] = []
            ap_dict[test_data_name] = []
            pre_dict[test_data_name] = []
            rec_dict[test_data_name] = []
        auc_dict[test_data_name].append(test_score['AUROC'])
        ap_dict[test_data_name].append(test_score['AP'])
        pre_dict[test_data_name].append(test_score['Precision@K'])
        rec_dict[test_data_name].append(test_score['Recall@K'])
        print(f"Test on {test_data_name}, AUC is {test_score['AUROC']}, AP is {test_score['AP']}, "
                          f"PRE is {test_score['Precision@K']}, REC is {test_score['Recall@K']}")

# # Calculate mean and standard deviation for each test dataset
auc_mean_dict, auc_std_dict, ap_mean_dict, ap_std_dict, pre_mean_dict, pre_std_dict, rec_mean_dict, rec_std_dict= {}, {}, {}, {}, {}, {}, {}, {}
for test_data_name in auc_dict:
    auc_mean_dict[test_data_name] = np.mean(auc_dict[test_data_name])
    auc_std_dict[test_data_name] = np.std(auc_dict[test_data_name])
    ap_mean_dict[test_data_name] = np.mean(ap_dict[test_data_name])
    ap_std_dict[test_data_name] = np.std(ap_dict[test_data_name])
    pre_mean_dict[test_data_name] = np.mean(pre_dict[test_data_name])
    pre_std_dict[test_data_name] = np.std(pre_dict[test_data_name])
    rec_mean_dict[test_data_name] = np.mean(rec_dict[test_data_name])
    rec_std_dict[test_data_name] = np.std(rec_dict[test_data_name])
# Output the results for each test dataset
for test_data_name in auc_mean_dict:
    str_result = 'AUROC:{:.2f}+-{:.2f}, AP:{:.2f}+-{:.2f}, PRE:{:.2f}+-{:.2f}, REC:{:.2f}+-{:.2f}'.format(
        auc_mean_dict[test_data_name] * 100,
        auc_std_dict[test_data_name] * 100,
        ap_mean_dict[test_data_name] * 100,
        ap_std_dict[test_data_name] * 100,
        pre_mean_dict[test_data_name] * 100,
        pre_std_dict[test_data_name] * 100,
        rec_mean_dict[test_data_name] * 100,
        rec_std_dict[test_data_name] * 100
    )
    print('-' * 40 + test_data_name + '-' * 40)
    print('str_result', str_result)

test_data_names = list(auc_dict.keys())

auc_values = np.array([auc_dict[name] for name in test_data_names])
ap_values = np.array([ap_dict[name] for name in test_data_names])
pre_values = np.array([pre_dict[name] for name in test_data_names])
rec_values = np.array([rec_dict[name] for name in test_data_names])

np.savetxt(result_path + 'auc_values.txt', auc_values.T)
np.savetxt(result_path + 'auc_values.txt',
           auc_values.T,
           fmt='%.6f',
           delimiter='\t',
           header='\t'.join(test_data_names),
           comments='')
np.savetxt(result_path + 'ap_values.txt',
           ap_values.T,
           fmt='%.6f',
           delimiter='\t',
           header='\t'.join(test_data_names),
           comments='')
np.savetxt(result_path + 'pre_values.txt',
           pre_values.T,
           fmt='%.6f',
           delimiter='\t',
           header='\t'.join(test_data_names),
           comments='')
np.savetxt(result_path + 'rec_values.txt',
           rec_values.T,
           fmt='%.6f',
           delimiter='\t',
           header='\t'.join(test_data_names),
           comments='')
