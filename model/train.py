import json
import os
import random
import numpy as np
# import pandas as pd
import data_process
import time
import torch
import random
import torch.nn as nn
from torch.nn import MSELoss
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import AutoConfig
from evaluation import extract_relation_emb, evaluate
from model import REMatchingModel
from transformers import get_linear_schedule_with_warmup
from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--gamma", type=float, default=0.06, help="Loss function: margin factor gamma")
parser.add_argument("--alpha", type=float, default=0.33,
                    help="Similarity: balance entity and context weights in single sample")
parser.add_argument("--batch_size", type=int, default=128)
parser.add_argument("--epochs", type=int, default=5, help='training epochs')
parser.add_argument("--lr", type=float, default=2e-6, help='learning rate')
parser.add_argument("--num_negsample", type=int, default=7,
                    help='Number of negative cases for each positive case')
parser.add_argument("--warm_up", type=float, default=0.1, help='warm_up rate')
parser.add_argument("--unseen", type=int, default=15, help='Number of unseen class')
parser.add_argument("--entity_way", type=str, choices=['tmp', 'keyword'], default='tmp',
                    help='Representation of the described entity')
# file_path
parser.add_argument("--dataset_path", type=str, default='../data', help='where data stored')
parser.add_argument("--dataset", type=str, default='fewrel', choices=['fewrel', 'wikizsl'],
                    help='original dataset')
parser.add_argument("--rel2id_path", type=str, default='../data/rel2id')
parser.add_argument("--rel_split_seed", type=str, default='ori')
parser.add_argument("--relation_description", type=str,
                    default='relation_description_addsynonym_processed.json',
                    help='relation descriptions of manual design')
# model and cuda config
parser.add_argument("--visible_device", type=str, default='0', help='the device on which this model will run')
parser.add_argument("--pretrained_model", type=str, default='bert-base-uncased', help='huggingface pretrained model')
args = parser.parse_args()
# ckpt_save_path, dataset ,relation_description, rel2id path

args.ckpt_save_path = f'../ckpt/{args.dataset}_split_{args.rel_split_seed}_unseen_{str(args.unseen)}'

args.dataset_file = os.path.join(args.dataset_path, args.dataset, f'{args.dataset}_dataset.json')
args.relation_description_file = os.path.join(args.dataset_path, args.dataset, 'relation_description',
                                              args.relation_description)
args.rel2id_file = os.path.join(args.rel2id_path, f'{args.dataset}_rel2id',
                                f'{args.dataset}_rel2id_{str(args.unseen)}_{args.rel_split_seed}.json')
print('ckpt_save_path:', args.ckpt_save_path)
print('dataset_file_path : ', args.dataset_file)
print('relation_description_file_path : ', args.relation_description_file)
print('rel2id_file_path : ', args.rel2id_file)

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


# set seed
set_seed(args.seed)

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
    training_data = [i for i in raw_data if i['relation'] in train_label]
    test_data = [i for i in raw_data if i['relation'] in test_label]

# print info
print(
    f'args_config: alpha:{args.alpha}, lr:{args.lr},seed:{args.seed},epochs:{args.epochs}')
print('there are {} kinds of relation in test.'.format(len(set(test_label))))
print('the lengths of test data is {} '.format(len(test_data)))

# load description
train_rel2vec, test_rel2vec = data_process.generate_attribute(args, train_desc, test_desc)

# load model
config = AutoConfig.from_pretrained(args.pretrained_model, num_labels=len(set(train_label)))
config.pretrained_model = args.pretrained_model
config.margin = args.gamma
config.alpha = args.alpha
model = REMatchingModel.from_pretrained(args.pretrained_model, config=config)
model = model.cuda()

# multi gpus
if torch.cuda.device_count() > 1:
    print('let use {} GPUs'.format(torch.cuda.device_count()))
    model = torch.nn.DataParallel(model)

trainset = data_process.FewRelDataset(args, 'train', training_data, train_rel2vec, train_relation2idx)
trainloader = DataLoader(trainset, batch_size=args.batch_size, collate_fn=data_process.create_mini_batch, shuffle=True)

# To evaluate the inference time
test_batchsize = 10 * args.unseen

testset = data_process.FewRelDataset(args, 'test', test_data, test_rel2vec, test_relation2idx)
testloader = DataLoader(testset, batch_size=test_batchsize,
                        collate_fn=data_process.create_mini_batch, shuffle=False)

train_y_attr, test_y_attr, test_y, test_y_e1, test_y_e2, train_y_e1, train_y_e2 = [], [], [], [], [], [], []

for i, test in enumerate(test_data):
    label = int(test_relation2idx[test['relation']])
    test_y.append(label)

for i in test_label:
    test_y_attr.append(test_rel2vec[i][0])
    test_y_e1.append(test_rel2vec[i][1])
    test_y_e2.append(test_rel2vec[i][2])

for i in train_label:
    train_y_attr.append(train_rel2vec[i][0])
    train_y_e1.append(train_rel2vec[i][1])
    train_y_e2.append(train_rel2vec[i][2])

train_y_attr, train_y_e1, train_y_e2 = np.array(train_y_attr), np.array(train_y_e1), np.array(train_y_e2)
test_y, test_y_attr, test_y_e1, test_y_e2 = np.array(test_y), np.array(test_y_attr), np.array(test_y_e1), np.array(
    test_y_e2)

# optimizer and scheduler
model.train()
optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
num_training_steps = len(trainset) * args.epochs // args.batch_size
warmup_steps = num_training_steps * args.warm_up
scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, num_training_steps)

test_pt, test_rt, test_f1t = 0.0, 0.0, 0.0
for epoch in range(args.epochs):
    print(f'============== TRAIN ON THE {epoch + 1}-th EPOCH ==============')
    running_loss = 0.0
    classifier_loss = 0.0
    out_sentence_embs = None
    e1_hs = None
    e2_hs = None
    train_y = None
    for step, data in enumerate(trainloader):
        tokens_tensors, marked_e1, marked_e2, marked_head, marked_tail, \
        attention_mask, relation_emb, relation_head_emb, relation_tail_emb, labels_ids = [t.cuda() for t in data]
        optimizer.zero_grad()

        outputs, out_sentence_emb, e1_h, e2_h = model(input_ids=tokens_tensors,
                                                      attention_mask=attention_mask,
                                                      e1_mask=marked_e1,
                                                      e2_mask=marked_e2,
                                                      marked_head=marked_head,
                                                      marked_tail=marked_tail,
                                                      input_relation_emb=relation_emb,
                                                      input_relation_head_emb=relation_head_emb,
                                                      input_relation_tail_emb=relation_tail_emb,
                                                      labels=labels_ids,
                                                      num_neg_sample=args.num_negsample
                                                      )

        loss = outputs[0].sum()
        loss = loss / args.batch_size
        rev_loss = outputs[1].mean()
        rev_loss.backward(retain_graph=True)
        loss.backward()
        optimizer.step()
        scheduler.step()

        running_loss += loss.item()
        classifier_loss += rev_loss.item()
        if step % 100 == 0:
            print(f'[step {step}]' + '=' * (step // 100))
            print('running_loss:{}'.format(running_loss / (step + 1)))
            print('classifier_loss:{}'.format(classifier_loss / (step + 1)))

    if epoch == args.epochs - 1:
        print('============== EVALUATION ON Test DATA ==============')
        preds, e1_hs, e2_hs = extract_relation_emb(model, testloader, 'test')
        test_pt, test_rt, test_f1t = evaluate(args, preds.cpu(), e1_hs.cpu(), e2_hs.cpu(), test_y_attr, test_y_e1,
                                              test_y_e2, test_y)
        torch.save(model.state_dict(), args.ckpt_save_path + f'_f1_{test_f1t}')
    print("* " * 20)
print(f'[test] final precision: {test_pt:.4f}, recall: {test_rt:.4f}, f1 score: {test_f1t:.4f}')
