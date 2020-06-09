import sys
import os
import numpy as np
import random
import math

import torch
import torch.optim as optim
import torchvision.transforms as transforms
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
from torch.nn.parallel import DataParallel
import pickle
import time
import argparse

from baseline.dataset import add_transforms
from baseline.dataset.Dataset import AttDataset
from baseline.model.DeepMAR import DeepMAR_ResNet50
from baseline.model.DeepMAR import DeepMAR_ResNet50_ExtractFeature 
from baseline.utils.evaluate import attribute_evaluate
from baseline.utils.utils import str2bool
from baseline.utils.utils import transfer_optim_state
from baseline.utils.utils import time_str
from baseline.utils.utils import save_ckpt, load_ckpt
from baseline.utils.utils import load_state_dict 
from baseline.utils.utils import ReDirectSTD
from baseline.utils.utils import adjust_lr_staircase
from baseline.utils.utils import set_devices
from baseline.utils.utils import AverageMeter
from baseline.utils.utils import to_scalar 
from baseline.utils.utils import may_set_mode 
from baseline.utils.utils import may_mkdir 
from baseline.utils.utils import set_seed
import pandas as pd


def build_graph_from_correlations(file_path, coeff_threshold=0.4, with_weights=False, reweighted=False, p=0.2):
    corrs = pd.read_csv(file_path, index_col=0)



    # an edge should be where the value is above the threshold
    # adjacent nodes are marked by a 1
    if coeff_threshold is not None and with_weights:
        adjacency = corrs.to_numpy()
        adjacency[adjacency < coeff_threshold] = 0
    elif coeff_threshold is not None:
        adjacency = np.where(corrs.to_numpy() >= coeff_threshold, 1, 0)
    elif with_weights:
        adjacency = corrs.to_numpy()

    # reweigh edge correlations to prevent oversmoothing
    if reweighted:
        corrs = pd.DataFrame(data=adjacency)
        i = 0
        for _, row in corrs.iterrows():
            sum = row.sum(axis=0)
            self_connection = corrs.iloc[i, i]
            for j, value in enumerate(row):
                if sum == self_connection:
                    reweight = self_connection
                else:
                    reweight = p / (sum - self_connection) if i != j else 1 - p
                corrs.iloc[i, j] = reweight * corrs.iloc[i, j]
            i = i + 1
        adjacency = corrs.to_numpy()

    return adjacency

def build_edge_index_from_adjacency(adjacency_matrix, self_connections=False):
    edge_index = []
    for i, row in enumerate(adjacency_matrix):
        for j, column in enumerate(row):
            if self_connections:
                if adjacency_matrix[i][j] != 0:
                    edge_index.append([i, j])
            else:
                if i != j and adjacency_matrix[i][j] != 0:
                    edge_index.append([i, j])
    edge_index = torch.Tensor(edge_index)
    return torch.t(edge_index)

def get_edge_weights_from_adjacency(adjacency_matrix, self_connections=False):
    edge_weights = []
    for i, row in enumerate(adjacency_matrix):
        for j, column in enumerate(row):
            if self_connections:
                if adjacency_matrix[i][j] != 0:
                    edge_weights.append(adjacency_matrix[i][j])
            else:
                if i != j and adjacency_matrix[i][j] != 0:
                    edge_weights.append(adjacency_matrix[i][j])
    edge_weights = torch.Tensor(edge_weights)
    return edge_weights

def get_adjacency_and_weights(input_file, coeff_threshold=None, with_weights=False, reweighted=False, self_connections=True, p=0.2):

    adjacency_matrix = build_graph_from_correlations(input_file,
                                                     coeff_threshold=coeff_threshold,
                                                     with_weights=with_weights,
                                                     reweighted=reweighted,
                                                     p=p)
    edge_weights = get_edge_weights_from_adjacency(adjacency_matrix, self_connections)
    edge_index = build_edge_index_from_adjacency(adjacency_matrix, self_connections)
    return (edge_index, edge_weights)

def relation_aware_loss(output, labels, weight=None):
    output, intermediate = output
    criterion_out = F.binary_cross_entropy_with_logits
    criterion_intermediate = F.binary_cross_entropy_with_logits
    lambda_1 = 1
    lambda_2 = 1
    loss_1 = criterion_out(output.float(), labels.float(), weight=weight)
    loss_2 = criterion_intermediate(intermediate.float(), labels.float(), weight=weight)
    return lambda_1 * loss_1 + lambda_2 * loss_2

class Config(object):
    def __init__(self):
        
        parser = argparse.ArgumentParser()
        parser.add_argument('-d', '--sys_device_ids', type=eval, default=(0,))
        parser.add_argument('--set_seed', type=str2bool, default=False)
        ## dataset parameter
        parser.add_argument('--dataset', type=str, default='peta',
                choices=['peta','rap', 'pa100k', 'rap2'])
        parser.add_argument('--split', type=str, default='trainval',
                            choices=['trainval', 'train'])
        parser.add_argument('--test_split', type=str, default='test')
        parser.add_argument('--partition_idx', type=int, default=0)
        parser.add_argument('--resize', type=eval, default=(224, 224))
        parser.add_argument('--mirror', type=str2bool, default=True)
        parser.add_argument('--batch_size', type=int, default=32)
        parser.add_argument('--workers', type=int, default=2)
        # model
        parser.add_argument('--num_att', type=int, default=35)
        parser.add_argument('--pretrained', type=str2bool, default=True)
        parser.add_argument('--last_conv_stride', type=int, default=2, choices=[1,2])
        parser.add_argument('--drop_pool5', type=str2bool, default=True)
        parser.add_argument('--drop_pool5_rate', type=float, default=0.5)

        parser.add_argument('--sgd_weight_decay', type=float, default=0.0005)
        parser.add_argument('--sgd_momentum', type=float, default=0.9)
        parser.add_argument('--new_params_lr', type=float, default=0.001)
        parser.add_argument('--finetuned_params_lr', type=float, default=0.001)
        parser.add_argument('--staircase_decay_at_epochs', type=eval,
                            default=(51, 101))
        parser.add_argument('--staircase_decay_multiple_factor', type=float,
                            default=0.1)
        parser.add_argument('--total_epochs', type=int, default=150)
        parser.add_argument('--weighted_entropy', type=str2bool, default=True)
        # utils
        parser.add_argument('--resume', type=str2bool, default=False)
        parser.add_argument('--ckpt_file', type=str, default='')
        parser.add_argument('--load_model_weight', type=str2bool, default=False)
        parser.add_argument('--model_weight_file', type=str, default='')
        parser.add_argument('--test_only', type=str2bool, default=False)
        parser.add_argument('--exp_dir', type=str, default='')
        parser.add_argument('--exp_subpath', type=str, default='deepmar_resnet50')
        parser.add_argument('--log_to_file', type=str2bool, default=True)
        parser.add_argument('--steps_per_log', type=int, default=20)
        parser.add_argument('--epochs_per_val', type=int, default=10)
        parser.add_argument('--epochs_per_save', type=int, default=50)
        parser.add_argument('--run', type=int, default=1)
        parser.add_argument('--coeff_threshold', type=float, default=0.4)
        parser.add_argument('--p', type=float, default=0.2)
        args = parser.parse_args()
        
        # gpu ids
        self.sys_device_ids = args.sys_device_ids
        # random
        self.set_seed = args.set_seed
        if self.set_seed:
            self.rand_seed = 0
        else: 
            self.rand_seed = None
        # run time index
        self.run = args.run
        # Dataset #
        datasets = dict()
        datasets['peta'] = '/home/ubuntu/code/pedestrian-attribute-recognition-pytorch/dataset/peta/peta_dataset.pkl'
        datasets['rap'] = './dataset/rap/rap_dataset.pkl'
        datasets['pa100k'] = './dataset/pa100k/pa100k_dataset.pkl'
        datasets['rap2'] = './dataset/rap2/rap2_dataset.pkl'
        partitions = dict()
        partitions['peta'] = '/home/ubuntu/code/pedestrian-attribute-recognition-pytorch/dataset/peta/peta_partition.pkl'
        partitions['rap'] = './dataset/rap/rap_partition.pkl'
        partitions['pa100k'] = './dataset/pa100k/pa100k_partition.pkl'
        partitions['rap2'] = './dataset/rap2/rap2_partition.pkl'
        
        self.dataset_name = args.dataset
        if not args.dataset in datasets or not args.dataset in partitions:
            print("Please select the right dataset name.")
            raise ValueError
        else:
            self.dataset = datasets[args.dataset]
            self.partition = partitions[args.dataset]
        self.partition_idx = args.partition_idx
        self.split = args.split
        self.test_split = args.test_split
        self.resize = args.resize
        self.mirror = args.mirror
        self.mean = [0.485, 0.456, 0.406]
        self.std = [0.229, 0.224, 0.225]
        self.batch_size = args.batch_size
        self.workers = args.workers
        # optimization
        self.sgd_momentum = args.sgd_momentum
        self.sgd_weight_decay = args.sgd_weight_decay
        self.new_params_lr = args.new_params_lr
        self.finetuned_params_lr = args.finetuned_params_lr
        self.staircase_decay_at_epochs = args.staircase_decay_at_epochs
        self.staircase_decay_multiple_factor = args.staircase_decay_multiple_factor
        self.total_epochs = args.total_epochs
        self.weighted_entropy = args.weighted_entropy

        # utils
        self.resume = args.resume
        self.ckpt_file = args.ckpt_file
        if self.resume:
            if self.ckpt_file == '':
                print('Please input the ckpt_file if you want to resume training')
                raise ValueError
        self.load_model_weight = args.load_model_weight
        self.model_weight_file = args.model_weight_file
        if self.load_model_weight:
            if self.model_weight_file == '':
                print('Please input the model_weight_file if you want to load model weight')
                raise ValueError
        self.test_only = args.test_only
        self.exp_dir = args.exp_dir
        self.exp_subpath = args.exp_subpath
        self.log_to_file = args.log_to_file
        self.steps_per_log = args.steps_per_log
        self.epochs_per_val = args.epochs_per_val
        self.epochs_per_save = args.epochs_per_save
        self.run = args.run

        self.coeff_threshold = args.coeff_threshold
        self.p = args.p
        
        # for model
        model_kwargs = dict()
        model_kwargs['num_att'] = args.num_att
        model_kwargs['last_conv_stride'] = args.last_conv_stride
        model_kwargs['drop_pool5'] = args.drop_pool5
        model_kwargs['drop_pool5_rate'] = args.drop_pool5_rate
        self.model_kwargs = model_kwargs
        # for evaluation
        self.test_kwargs = dict()

        if self.exp_dir == '':
            self.exp_dir = os.path.join('exp',
                '{}'.format(self.exp_subpath),
                '{}'.format(self.dataset_name),
                'partition{}'.format(self.partition_idx),
                'run{}'.format(self.run))
        self.stdout_file = os.path.join(self.exp_dir, \
            'log', 'stdout_{}.txt'.format(time_str()))
        self.stderr_file = os.path.join(self.exp_dir, \
            'log', 'stderr_{}.txt'.format(time_str()))
        may_mkdir(self.stdout_file)

### main function ###
cfg = Config()

# log
if cfg.log_to_file:
    ReDirectSTD(cfg.stdout_file, 'stdout', False)
    ReDirectSTD(cfg.stderr_file, 'stderr', False)

# dump the configuration to log.
import pprint
print('-' * 60)
print('cfg.__dict__')
pprint.pprint(cfg.__dict__)
print('-' * 60)

# set the random seed
if cfg.set_seed:
    set_seed( cfg.rand_seed )
# init the gpu ids
set_devices(cfg.sys_device_ids)

# dataset 
normalize = transforms.Normalize(mean=cfg.mean, std=cfg.std)
transform = transforms.Compose([
        transforms.Resize(cfg.resize),
        transforms.Pad((24, 24)),
        transforms.RandomCrop(cfg.resize),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(), # 3*H*W, [0, 1]
        normalize,
        transforms.RandomErasing(),]) # normalize with mean/std
# by a subset of attributes 
train_set = AttDataset(
    dataset = cfg.dataset, 
    partition = cfg.partition,
    split = cfg.split,
    partition_idx= cfg.partition_idx,
    transform = transform)

num_att = len(train_set.dataset['selected_attribute'])
cfg.model_kwargs['num_att'] = num_att

train_loader = torch.utils.data.DataLoader(
    dataset = train_set,
    batch_size = cfg.batch_size,
    shuffle = True,
    num_workers = cfg.workers,
    pin_memory = True,
    drop_last = False)

test_transform = transforms.Compose([
        transforms.Resize(cfg.resize),
        transforms.ToTensor(),
        normalize,])
test_set = AttDataset(
    dataset = cfg.dataset,
    partition = cfg.partition,
    split = cfg.test_split,
    partition_idx = cfg.partition_idx,
    transform = test_transform)
### Att model ###
model = DeepMAR_ResNet50(**cfg.model_kwargs)

# Wrap the model after set_devices, data parallel
model_w = torch.nn.DataParallel(model)

# using the weighted cross entropy loss
if cfg.weighted_entropy:
    rate = np.array(train_set.partition['weight_' + cfg.split][cfg.partition_idx])
    rate = rate[train_set.dataset['selected_attribute']].tolist()
else:
    rate = None
# compute the weight of positive and negative
if rate is None:
    weight_pos = [1 for i in range(num_att)]
    weight_neg = [1 for i in range(num_att)]
else:
    if len(rate) != num_att:
        print("the length of rate should be equal to %d" % (num_att))
        raise ValueError
    weight_pos = []
    weight_neg = []
    for idx, v in enumerate(rate):
        weight_pos.append(math.exp(1.0 - v))
        weight_neg.append(math.exp(v))
criterion = relation_aware_loss

# Optimizer
finetuned_params = []
new_params = []
for p in model.parameters():

    finetuned_params.append(p)
param_groups = [{'params': finetuned_params, 'lr': cfg.finetuned_params_lr},
                {'params': new_params, 'lr': cfg.new_params_lr}]

optimizer = optim.SGD(
    param_groups,
    momentum = cfg.sgd_momentum,
    weight_decay = cfg.sgd_weight_decay)
# bind the model and optimizer
modules_optims = [model, optimizer]

# load model weight if necessary
if cfg.load_model_weight:
    map_location = (lambda storage, loc:storage)
    ckpt = torch.load(cfg.model_weight_file, map_location=map_location)
    model.load_state_dict(ckpt['state_dicts'][0], strict=False)

### Resume or not ###
if cfg.resume:
    # store the model, optimizer, epoch 
    start_epoch, scores = load_ckpt(modules_optims, cfg.ckpt_file)
else:
    start_epoch = 0

model_w = torch.nn.DataParallel(model)
model_w.cuda()
transfer_optim_state(state=optimizer.state, device_id=0)

# cudnn.benchmark = True
# for evaluation
feat_func_att = DeepMAR_ResNet50_ExtractFeature(model=model_w)

def attribute_evaluate_subfunc(feat_func, test_set, edge_index, edge_weights, **test_kwargs):
    """ evaluate the attribute recognition precision """
    result = attribute_evaluate(feat_func, test_set, edge_index, edge_weights, **test_kwargs)
    print('-' * 60)
    print('Evaluation on %s set:' % (cfg.test_split))
    print('Label-based evaluation: \n mA: %.4f'%(np.mean(result['label_acc'])))
    print('Instance-based evaluation: \n Acc: %.4f, Prec: %.4f, Rec: %.4f, F1: %.4f' \
        %(result['instance_acc'], result['instance_precision'], result['instance_recall'], result['instance_F1']))
    print('-' * 60)

# print the model into log
print(model)
# test only
if cfg.test_only:
    print('test with feat_func_att')
    attribute_evaluate_subfunc(feat_func_att, test_set, **cfg.test_kwargs)
    sys.exit(0)
     
# training
for epoch in range(start_epoch, cfg.total_epochs):
    # adjust the learning rate
    adjust_lr_staircase(
        optimizer.param_groups,
        [cfg.finetuned_params_lr, cfg.new_params_lr],
        epoch + 1,
        cfg.staircase_decay_at_epochs,
        cfg.staircase_decay_multiple_factor)
    
    may_set_mode(modules_optims, 'train')
    # recording loss
    loss_meter = AverageMeter()
    dataset_L = len(train_loader)
    ep_st = time.time()

    basepath_results = "/home/ubuntu/results/"

    input_file = os.path.join(basepath_results, "correlations_peta.csv")

    edge_index, edge_weights = get_adjacency_and_weights(input_file,
                                                         with_weights=True,
                                                         coeff_threshold=cfg.coeff_threshold,
                                                         reweighted=True,
                                                         self_connections=False,
                                                         p=cfg.p)
    edge_index = edge_index.cuda()
    edge_weights = edge_weights.cuda()
    edge_index = edge_index.long()

    for step, (imgs, targets) in enumerate(train_loader):
         
        step_st = time.time()
        imgs_var = Variable(imgs).cuda()
        targets_var = Variable(targets).cuda()

        score = model_w(imgs_var, edge_index, edge_weights)

        # compute the weight
        weights = torch.zeros(targets_var.shape)
        for i in range(targets_var.shape[0]):
            for j in range(targets_var.shape[1]):
                if targets_var.data.cpu()[i, j] == -1:
                    weights[i, j] = weight_neg[j]
                elif targets_var.data.cpu()[i, j] == 1:
                    weights[i, j] = weight_pos[j]
                else:
                    weights[i, j] = 0

        # loss for the attribute classification, average over the batch size
        targets_var[targets_var == -1] = 0
        loss = criterion(score, targets_var, weight=Variable(weights.cuda()))*num_att

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        ############
        # step log #
        ############
        loss_meter.update(to_scalar(loss))

        if (step+1) % cfg.steps_per_log == 0 or (step+1)%len(train_loader) == 0:
            log = '{}, Step {}/{} in Ep {}, {:.2f}s, loss:{:.4f}'.format( \
            time_str(), step+1, dataset_L, epoch+1, time.time()-step_st, loss_meter.val)
            print(log)

    ##############
    # epoch log  #
    ##############
    log = 'Ep{}, {:.2f}s, loss {:.4f}'.format(
        epoch+1, time.time() - ep_st, loss_meter.avg)
    print(log)

    # model ckpt
    if (epoch + 1) % cfg.epochs_per_save == 0 or epoch+1 == cfg.total_epochs:
        ckpt_file = os.path.join(cfg.exp_dir, 'model', 'ckpt_epoch%d.pth'%(epoch+1))
        save_ckpt(modules_optims, epoch+1, 0, ckpt_file)

    ##########################
    # test on validation set #
    ##########################
    if (epoch + 1) % cfg.epochs_per_val == 0 or epoch+1 == cfg.total_epochs:
        print('att test with feat_func_att')
        attribute_evaluate_subfunc(feat_func_att, test_set, edge_index, edge_weights, **cfg.test_kwargs)
