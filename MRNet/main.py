import random
import numpy as np

import torch
from torch.utils.data import DataLoader
from transformers import AutoConfig, WavLMModel

try: 
    import egg_exp
except:
    import sys
    sys.path.append('/exp_lib')
from egg_exp import log, dataset, loss, model, framework, signal_processing, data_augmentation
import arguments
import data_processing
import train

# get arguments
args, system_args, experiment_args = arguments.get_args()

# set reproducible
random.seed(args['rand_seed'])
np.random.seed(args['rand_seed'])
torch.manual_seed(args['rand_seed'])
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = True

# logger
builder = log.LoggerList.Builder(args['name'], args['project'], args['tags'], args['description'], args['path_scripts'], args)
builder.use_local_logger(args['path_log'])
#builder.use_neptune_logger(args['neptune_user'], args['neptune_token'])
#builder.use_wandb_logger(args['wandb_entity'], args['wandb_api_key'], args['wandb_group'])
logger = builder.build()
logger.log_arguments(experiment_args)

# data loader
fma_large = dataset.FMA(args['path_FMA'], args['FMA_size'])
train_set = data_processing.TrainSet(args, fma_large.train_set, args['p_gaussian_noise'])
train_loader = DataLoader(
    train_set,
    num_workers=args['num_workers'],
    batch_size=args['batch_size'],
    pin_memory=True,
    shuffle=True,
    drop_last=True
)
val_set = data_processing.ValidationSet(args, fma_large.val_set)
val_loader = DataLoader(
    val_set,
    num_workers=args['num_workers'],
    batch_size=args['batch_size'],
    pin_memory=True,
)
eval_set = data_processing.EvaluationSet(args, fma_large.test_set)
eval_loader = DataLoader(
    eval_set,
    num_workers=args['num_workers'] * 2,
    batch_size=1,
    pin_memory=True,
)

# DA
ga = data_augmentation.GaussianNoise(args['p_gaussian_noise'], args['gaussian_mean'], args['gaussian_std'])

# model
ssl = WavLMModel.from_pretrained(
    args['huggingface_url'],
    from_tf=bool(".ckpt" in args['huggingface_url']),
    config=AutoConfig.from_pretrained(args['huggingface_url']),
    revision="main",
    ignore_mismatched_sizes=False, 
)

mlfa_net = model.MLFANet(args['hidden_size'], args['C'], args['embed_size'], args['hidden_num'])

# criterion
cce_loss = loss.CCE(
    args['embed_size'], len(fma_large.class_weight), class_weight=fma_large.class_weight
)

# framework
mgc_framework = framework.MusicGenreClassificationFramework(
    ssl=ssl,
    train_ssl=False,
    model=mlfa_net,
    data_augmentation=[ga],
    criterion=cce_loss
)
mgc_framework.cuda()
logger.log_text('num_param', f'{mgc_framework.get_num_trainable_parameters()}')

# optimizer
optimizer = torch.optim.AdamW(
    mgc_framework.get_parameters(), 
    lr=args['lr'], 
    weight_decay=args['weight_decay'],
    amsgrad=True
)

# lr scheduler
lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
    optimizer,
    T_0=args['T_0'],
    T_mult=args['T_mult'],
    eta_min=args['lr_min']
)
logger.log_text('w', f'{mgc_framework.modules["model"].w}')
logger.log_text('w_all', f'{mgc_framework.modules["model"].w_all}')

#=============================================
#                   Train 
#=============================================
best_acc = 0
best_state_dict = mgc_framework.copy_state_dict()
acc_sum = 0

for epoch in range(1, args['epoch'] + 1):
    # log lr
    for p_group in optimizer.param_groups:
        lr = p_group['lr']
        logger.log_metric('lr', lr, epoch)

    lr_scheduler.step(epoch)
    
    # train
    train.train(epoch, mgc_framework, train_loader, optimizer, logger)
    logger.log_text('w', f'{mgc_framework.modules["model"].w}')
    logger.log_text('w_all', f'{mgc_framework.modules["model"].w_all}')

    # validation
    corrects, _, total_samples = train.test_inference(fma_large, mgc_framework, val_loader)
    accuracy = sum(corrects) / sum(total_samples) * 100
    logger.log_metric(f"val/acc", accuracy, epoch)
    acc_sum += accuracy
    if epoch % 5 == 0:
        logger.log_metric(f"val/acc_avg5", acc_sum / 5, epoch)
        acc_sum = 0
        
    if best_acc < accuracy:
        best_acc = accuracy
        best_state_dict = mgc_framework.copy_state_dict()

# evaluation
mgc_framework.load_state_dict(best_state_dict)
corrects, predicts, total_samples = train.test_inference(fma_large, mgc_framework, eval_loader)

# log eval accuracy
keys = list(fma_large.genre_dict.keys())
'''
for i in range(len(corrects)):
    logger.log_metric(f"eval/acc_{keys[i]}", corrects[i] / total_samples[i] * 100, epoch)
'''
logger.log_metric(f"eval/acc", sum(corrects) / sum(total_samples) * 100) 

# log eval f1-score
f1_sum = 0
for i in range(len(corrects)):
    if predicts[i] == 0:
        f1 = 0
    else:
        precision = corrects[i]/predicts[i]
        recall = corrects[i]/total_samples[i]
        if precision + recall == 0:
            f1 = 0
        else:
            f1 = (2 * precision * recall) / (precision + recall)
    f1_sum += f1
    # logger.log_metric(f"eval/f1_score_{keys[i]}", f1, epoch)   
logger.log_metric(f"eval/macro_F1", f1_sum / len(corrects))