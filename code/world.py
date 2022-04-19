import os
from os.path import join
import torch
from parse import parse_args
import multiprocessing

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
args = parse_args()

ROOT_PATH = "./"
CODE_PATH = join(ROOT_PATH, 'code')
DATA_PATH = join(ROOT_PATH, 'data')
BOARD_PATH = join(CODE_PATH, 'runs')
FILE_PATH = join(CODE_PATH, 'checkpoints')
import sys

sys.path.append(join(CODE_PATH, 'sources'))

if not os.path.exists(FILE_PATH):
    os.makedirs(FILE_PATH, exist_ok=True)

config = {}
all_dataset = ['MIND', 'yelp2018', 'amazon-book']
all_models = ['lgn', 'kgc', 'sgl', 'sgl-rgat']
config['bpr_batch_size'] = args.bpr_batch
config['latent_dim_rec'] = args.recdim
config['lightGCN_n_layers'] = args.layer
config['dropout'] = args.dropout
config['keep_prob'] = args.keepprob
config['A_n_fold'] = args.a_fold
config['test_u_batch_size'] = args.testbatch
config['multicore'] = args.multicore
config['lr'] = args.lr
config['decay'] = args.decay
config['pretrain'] = args.pretrain
config['A_split'] = False

GPU = torch.cuda.is_available()
device = torch.device('cuda' if GPU else "cpu")

# RGAT \ NO
kgcn = "RGAT"
use_trans = True
entity_num_per_item = 10
# WEIGHTED (-MIX) \ RANDOM \ NO
uicontrast = "WEIGHTED"
kgcontrast = True
kgc_joint = True
use_kgc_pretrain = False
kgc_temp = 0.2
kg_p_drop = 0.5
ui_p_drop = 0.1
ssl_reg = 0.1
CORES = multiprocessing.cpu_count() // 2
seed = args.seed


test_verbose = 1
test_start_epoch = 20
early_stop_cnt = 5

dataset = args.dataset
if dataset == 'MIND':
    config['lr'] = 5e-4
    config['decay'] = 1e-3

    uicontrast = "WEIGHTED-MIX"
    ssl_reg = 0.06
    kgc_temp = 0.2
    kg_p_drop = 0.5
    ui_p_drop = 0.4

    mix_ratio = 1 - ui_p_drop - 0
    test_start_epoch = 1

elif dataset == 'amazon-book':
    uicontrast = "WEIGHTED"
    ui_p_drop = 0.1
    test_start_epoch = 15

elif dataset == 'yelp2018':
    uicontrast = "WEIGHTED"
    ui_p_drop = 0.1
    test_start_epoch = 25

model_name = args.model
if dataset not in all_dataset:
    raise NotImplementedError(
        f"Haven't supported {dataset} yet!, try {all_dataset}")
if model_name not in all_models:
    raise NotImplementedError(
        f"Haven't supported {model_name} yet!, try {all_models}")

if model_name == 'lgn':
    kgcn = "NO"
    use_trans = False
    # WEIGHTED \ RANDOM \ ITEM-BI \ PGRACE \NO
    uicontrast = "NO"
    kgcontrast = False
    kgc_joint = False
    use_kgc_pretrain = False
elif model_name == 'sgl':
    kgcn = "NO"
    use_trans = False
    # WEIGHTED \ RANDOM \ ITEM-BI \ PGRACE \NO
    uicontrast = "RANDOM"
    kgcontrast = False
    kgc_joint = False
    use_kgc_pretrain = False
    ui_p_drop = 0.1
    test_start_epoch = 5

TRAIN_epochs = args.epochs
LOAD = args.load
SAVE = args.save
test_file = "/" + args.test_file
PATH = args.path
topks = eval(args.topks)
tensorboard = args.tensorboard
comment = args.comment
# let pandas shut up
from warnings import simplefilter

simplefilter(action="ignore", category=FutureWarning)


def cprint(words: str):
    print(words)
    # print(f"\033[0;30;43m{words}\033[0m")