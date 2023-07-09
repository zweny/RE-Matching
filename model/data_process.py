import json
import torch
import numpy as np
import re
# import pandas as pd
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
from transformers import AutoTokenizer
from sentence_transformers import SentenceTransformer


def inverse_tokenize(tokens):
    r"""
    Convert tokens to sentence.
    Untokenizing a text undoes the tokenizing operation, restoring
    punctuation and spaces to the places that people expect them to be.
    Ideally, `untokenize(tokenize(text))` should be identical to `text`,
    except for line breaks.
    Watch out!
    Default punctuation add to the word before its index,
    it may raise inconsistency bug.
    :param list[str]r tokens: target token list
    :return: str
    """
    assert isinstance(tokens, list)
    text = ' '.join(tokens)
    step1 = text.replace("`` ", '"') \
        .replace(" ''", '"') \
        .replace('. . .', '...')
    step2 = step1.replace(" ( ", " (").replace(" ) ", ") ")
    step3 = re.sub(r' ([.,:;?!%]+)([ \'"`])', r"\1\2", step2)
    step4 = re.sub(r' ([.,:;?!%]+)$', r"\1", step3)
    step5 = step4.replace(" '", "'").replace(" n't", "n't").replace(
        "can not", "cannot")
    step6 = step5.replace(" ` ", " '")
    step7 = step6.replace('do nt', 'dont').replace('Do nt', 'Dont')
    step8 = step7.replace(' - ', '-')
    return step8.strip()


def generate_attribute(args, train_description_file, val_description_file):
    # get fixed and pre-trained embedding encoder
    encoder_dict = {'bert-base-uncased': 'stsb-bert-base', 'roberta-base': 'stsb-roberta-base-v2',
                    'distilbert-base-uncased': 'stsb-distilbert-base', 'distilroberta-base': 'stsb-distilroberta-base-v2'}
    encoder = SentenceTransformer(encoder_dict[args.pretrained_model])
    # get description sentence
    train_description_sentences = [inverse_tokenize(single_data['description']) for single_data in
                                   train_description_file]
    val_description_sentences = [inverse_tokenize(single_data['description']) for single_data in val_description_file]

    # get description's head-tail entity
    if args.entity_way == 'tmp':
        train_head_entity_sentences = ['the entity types including ' + ' and '.join(single_data['h']) for single_data in
                                       train_description_file]
        train_tail_entity_sentences = ['the entity types including ' + ' and '.join(single_data['t']) for single_data in
                                       train_description_file]
        val_head_entity_sentences = ['the entity types including ' + ' and '.join(single_data['h']) for single_data in
                                     val_description_file]
        val_tail_entity_sentences = ['the entity types including ' + ' and '.join(single_data['t']) for single_data in
                                     val_description_file]

    elif args.entity_way == 'keyword':
        train_head_entity_sentences = [' '.join(single_data['h']) for single_data in
                                       train_description_file]
        train_tail_entity_sentences = [' '.join(single_data['t']) for single_data in
                                       train_description_file]
        val_head_entity_sentences = [' '.join(single_data['h']) for single_data in
                                     val_description_file]
        val_tail_entity_sentences = [' '.join(single_data['t']) for single_data in
                                     val_description_file]
    # get description's context embeddings
    train_description_sentence_embeddings = encoder.encode(train_description_sentences)
    val_description_sentence_embeddings = encoder.encode(val_description_sentences)

    # get description's entity embeddings
    train_head_entity_embeddings = encoder.encode(train_head_entity_sentences)
    train_tail_entity_embeddings = encoder.encode(train_tail_entity_sentences)
    val_head_entity_embeddings = encoder.encode(val_head_entity_sentences)
    val_tail_entity_embeddings = encoder.encode(val_tail_entity_sentences)

    # get relations
    train_description_relations = [single_data['relation'] for single_data in
                                   train_description_file]
    val_description_relations = [single_data['relation'] for single_data in val_description_file]

    train_rel2vec, val_rel2vec = {}, {}

    for rid, embedding, head_emb, tail_emb in zip(train_description_relations, train_description_sentence_embeddings,
                                                  train_head_entity_embeddings, train_tail_entity_embeddings):
        train_rel2vec[rid] = [embedding.astype('float32'), head_emb.astype('float32'), tail_emb.astype('float32')]
    for rid, embedding, head_emb, tail_emb in zip(val_description_relations, val_description_sentence_embeddings,
                                                  val_head_entity_embeddings, val_tail_entity_embeddings):
        val_rel2vec[rid] = [embedding.astype('float32'), head_emb.astype('float32'), tail_emb.astype('float32')]
    return train_rel2vec, val_rel2vec


def mark_fewrel_entity(new_pos, new_entity_h, new_entity_t, sent_len):
    mark_head = np.array([0] * sent_len)
    mark_tail = np.array([0] * sent_len)
    mark_head[new_entity_h[0]:new_entity_h[1]] = 1
    mark_tail[new_entity_t[0]:new_entity_t[1]] = 1
    marked_e1 = np.array([0] * sent_len)
    marked_e2 = np.array([0] * sent_len)
    marked_e1[new_pos[0]] = 1
    marked_e2[new_pos[1]] = 1
    return torch.tensor(marked_e1, dtype=torch.long), torch.tensor(marked_e2, dtype=torch.long), torch.tensor(mark_head,
                                                                                                              dtype=torch.long), torch.tensor(
        mark_tail, dtype=torch.long)


def create_mini_batch(samples):
    # all of here are positive samples
    tokens_tensors = [s[0] for s in samples]
    attention_mask = [s[1] for s in samples]
    marked_e1_tensor = [s[2] for s in samples]
    marked_e2_tensor = [s[3] for s in samples]
    mark_head = [s[4] for s in samples]
    mark_tail = [s[5] for s in samples]
    relation_emb = [s[6] for s in samples]
    relation_head_emb = [s[7] for s in samples]
    relation_tail_emb = [s[8] for s in samples]
    # new_pos = [s[8] for s in samples]
    if samples[0][9] is not None:
        label_ids = torch.stack([s[9] for s in samples])
    else:
        label_ids = None

    tokens_tensors = pad_sequence(tokens_tensors,
                                  batch_first=True)
    marked_e1_tensor = pad_sequence(marked_e1_tensor,
                                    batch_first=True)
    marked_e2_tensor = pad_sequence(marked_e2_tensor,
                                    batch_first=True)
    marked_head = pad_sequence(mark_head, batch_first=True)
    marked_tail = pad_sequence(mark_tail, batch_first=True)

    attention_mask = pad_sequence(attention_mask, batch_first=True)
    relation_emb = torch.tensor(relation_emb)
    relation_head_emb = torch.tensor(relation_head_emb)
    relation_tail_emb = torch.tensor(relation_tail_emb)

    return tokens_tensors, marked_e1_tensor, marked_e2_tensor, marked_head, marked_tail, attention_mask, relation_emb, relation_head_emb, relation_tail_emb, label_ids


class FewRelDataset(Dataset):
    def __init__(self, args, mode, data, rel2vec, relation2idx):
        assert mode in ['train', 'test']
        self.mode = mode
        self.data = data
        self.rel2vec = rel2vec
        self.relation2idx = relation2idx
        self.len = len(data)
        self.tokenizer = AutoTokenizer.from_pretrained(
            args.pretrained_model)
        self.max_length = 128
        # in bert-base ,1001 is the index of ' #',1030 is the index of ' @'
        # in roberta ,849 is the index of ' #' ,787 is the index of ' @'
        if args.pretrained_model in ['bert-base-uncased', 'distilbert-base-uncased']:
            self.head_mark_ids = 1001
            self.tail_mark_ids = 1030
        elif args.pretrained_model in ['roberta-base', 'distilroberta-base']:
            self.head_mark_ids = 849
            self.tail_mark_ids = 787

    def __getitem__(self, idx):
        single_data = self.data[idx]
        pos1 = single_data['h']['pos'][0]
        pos1_end = single_data['h']['pos'][1]
        pos2 = single_data['t']['pos'][0]
        pos2_end = single_data['t']['pos'][1]
        words = single_data['token']

        if pos1 < pos2:
            new_words = words[:pos1] + ['#'] + words[pos1:pos1_end] + ['#'] + words[pos1_end:pos2] \
                        + ['@'] + words[pos2:pos2_end] + ['@'] + words[pos2_end:]

        else:
            new_words = words[:pos2] + ['@'] + words[pos2:pos2_end] + ['@'] + words[pos2_end:pos1] \
                        + ['#'] + words[pos1:pos1_end] + ['#'] + words[pos1_end:]

        sentence = " ".join(new_words)
        tokens_info = self.tokenizer(sentence)
        tokens_ids = tokens_info['input_ids']
        attention_mask = torch.tensor(tokens_info['attention_mask'])
        # for roberta
        if pos2 == 0:
            tokens_ids[1] = self.tail_mark_ids
        elif pos1 == 0:
            tokens_ids[1] = self.head_mark_ids

        new_head_pos = tokens_ids.index(self.head_mark_ids)
        new_tail_pos = tokens_ids.index(self.tail_mark_ids)
        new_head_end_pos = tokens_ids.index(self.head_mark_ids, new_head_pos + 1)
        new_tail_end_pos = tokens_ids.index(self.tail_mark_ids, new_tail_pos + 1)
        new_pos = (new_head_pos, new_tail_pos)
        new_entity_h = (new_head_pos + 1, new_head_end_pos)
        new_entity_t = (new_tail_pos + 1, new_tail_end_pos)

        marked_e1, marked_e2, mark_head, mark_tail = mark_fewrel_entity(new_pos, new_entity_h, new_entity_t,
                                                                        len(tokens_ids))
        relation_emb, relation_head_emb, relation_tail_emb = self.rel2vec[single_data['relation']]
        tokens_ids = torch.tensor(tokens_ids)
        label_idx_tensor = None
        if self.mode == 'train':
            label_idx = int(self.relation2idx[single_data['relation']])
            label_idx_tensor = torch.tensor(label_idx)
        elif self.mode == 'test':
            label_idx_tensor = None

        return (
            tokens_ids, attention_mask, marked_e1, marked_e2, mark_head, mark_tail, relation_emb, relation_head_emb,
            relation_tail_emb, label_idx_tensor)

    def __len__(self):
        return self.len
