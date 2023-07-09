import json
import os
import random
import numpy as np
# import pandas as pd
import data_process
import torch
import random
import torch.nn as nn
from torch.nn import MSELoss
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import AutoConfig
from evaluation import extract_relation_emb, evaluate
from model import REMatchingModel
from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument("--seed", type=int, default=42,
                    help="random seed, should match the splited dataset if you don not wanna use original splited")
parser.add_argument("--gamma", type=float, default=0.06, help="Loss function: margin factor gamma")
parser.add_argument("--alpha", type=float, default=0.33,
                    help="Similarity: balance entity and context weights in single sample")
parser.add_argument("--lambda_", type=float, default=0.5, help='Reversal rate in Gradient Reversal Layer')
parser.add_argument("--num_negsample", type=int, default=7,
                    help='the number of negative sample per positive sample')
parser.add_argument("--entity_way", type=str, choices=['tmp', 'keyword'], default='tmp',
                    help='the description entity represent way')
# file_path
parser.add_argument("--dataset_path", type=str, default='../data', help='where data stored')
parser.add_argument("--rel2id_path", type=str, default='../data/rel2id')
parser.add_argument("--relation_description", type=str,
                    default='relation_description_addsynonym_processed.json',
                    help='relation description of manual design')
parser.add_argument("--ckpt_save_path", type=str, default='/root/ckpt/')
# model and cuda config
parser.add_argument("--ckpt_name", type=str, default='')
parser.add_argument("--visible_device", type=str, default='4', help='which devices this model run')
parser.add_argument("--pretrained_model", type=str, default='bert-base-uncased', help='huggingface pretrained model')

args = parser.parse_args()
args.dataset = args.ckpt_name[:args.ckpt_name.find('_')]
args.ckpt_load_path = args.ckpt_save_path + args.ckpt_name
args.rel_split_seed = args.ckpt_load_path[args.ckpt_load_path.find('split_')+6:args.ckpt_load_path.find('_un')]
args.unseen = int(args.ckpt_load_path[args.ckpt_load_path.find('unseen_')+7:args.ckpt_load_path.find('_f1')])
print(f'dataset:{args.dataset}, rel_split_seed : {args.rel_split_seed}, unseen: {str(args.unseen)}')

# dataset ,relation_description, rel2id path
args.dataset_file = os.path.join(args.dataset_path, args.dataset, f'{args.dataset}_dataset.json')
args.relation_description_file = os.path.join(args.dataset_path, args.dataset, 'relation_description',
                                              args.relation_description)
args.rel2id_file = os.path.join(args.rel2id_path, f'{args.dataset}_rel2id',
                                f'{args.dataset}_rel2id_{str(args.unseen)}_{args.rel_split_seed}.json')

# cuda set
os.environ['CUDA_VISIBLE_DEVICES'] = args.visible_device


def set_seed(seed: int):
    """Sets the relevant random seeds."""
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# fix seed
set_seed(args.seed)

# load rel2id and id2rel
with open(args.rel2id_file, 'r', encoding='utf-8') as r2id:
    relation2idx = json.load(r2id)
    train_relation2idx, test_relation2idx = relation2idx['train'], relation2idx['test']
    train_idx2relation, test_idx2relation, = dict((v, k) for k, v in train_relation2idx.items()), \
                                             dict((v, k) for k, v in test_relation2idx.items())

train_label, test_label = list(train_relation2idx.keys()), list(test_relation2idx.keys())

# load rel_description
with open(args.relation_description_file, 'r', encoding='utf-8') as rd:
    relation_desc = json.load(rd)
    train_desc = [i for i in relation_desc if i['relation'] in train_label]
    test_desc = [i for i in relation_desc if i['relation'] in test_label]
# load data
with open(args.dataset_file, 'r', encoding='utf-8') as d:
    raw_data = json.load(d)
    test_data = [i for i in raw_data if i['relation'] in test_label]

# print info
print('there are {} kinds of relation in test.'.format(len(set(test_label))))

# load description
train_rel2vec, test_rel2vec = data_process.generate_attribute(args, train_desc, test_desc)

# load model
config = AutoConfig.from_pretrained(args.pretrained_model, num_labels=len(set(train_label)))
config.pretrained_model = args.pretrained_model
config.margin = args.gamma
config.alpha = args.alpha
config.lambda_ = args.lambda_
model = REMatchingModel.from_pretrained(args.pretrained_model, config=config)
model = model.cuda()
model.load_state_dict(torch.load(args.ckpt_load_path))

# To evaluate the inference time
test_batchsize = 100
testset = data_process.FewRelDataset(args, 'test', test_data, test_rel2vec, test_relation2idx)
testloader = DataLoader(testset, batch_size=test_batchsize,
                        collate_fn=data_process.create_mini_batch, shuffle=False)

test_y_attr, test_y, test_y_e1, test_y_e2 = [], [], [], []

for i, test in enumerate(test_data):
    label = int(test_relation2idx[test['relation']])
    test_y.append(label)

for i in test_label:
    test_y_attr.append(test_rel2vec[i][0])
    test_y_e1.append(test_rel2vec[i][1])
    test_y_e2.append(test_rel2vec[i][2])

test_y, test_y_attr, test_y_e1, test_y_e2 = np.array(test_y), np.array(test_y_attr), np.array(test_y_e1), np.array(
    test_y_e2)

print('============== EVALUATION ON Test DATA ==============')
preds, e1_hs, e2_hs = extract_relation_emb(model, testloader, 'test')
test_pt, test_rt, test_f1t = evaluate(args, preds.cpu(), e1_hs.cpu(), e2_hs.cpu(), test_y_attr, test_y_e1,
                                      test_y_e2, test_y, test_idx2relation)

print(f'[test] (macro) final precision: {test_pt:.4f}, recall: {test_rt:.4f}, f1 score: {test_f1t:.4f}')
