import json
import re
import ast

import pandas as pd
import dask.dataframe as dd
import torch
from torch.utils.data import Dataset
from allennlp.modules.elmo import Elmo
from allennlp.modules.elmo import batch_to_ids

import model.constant as C

DIGIT_PATTERN = re.compile('\d')

def bio_to_bioes(labels):
    """Convert a sequence of BIO labels to BIOES labels.
    :param labels: A list of labels.
    :return: A list of converted labels.
    """
    label_len = len(labels)
    labels_bioes = []
    for idx, label in enumerate(labels):
        next_label = labels[idx + 1] if idx < label_len - 1 else 'O'
        if label == 'O':
            labels_bioes.append('O')
        elif label.startswith('B-'):
            if next_label.startswith('I-'):
                labels_bioes.append(label)
            else:
                labels_bioes.append('S-' + label[2:])
        else:
            if next_label.startswith('I-'):
                labels_bioes.append(label)
            else:
                labels_bioes.append('E-' + label[2:])
    return labels_bioes


def mask_to_distance(mask, mask_len, decay=.1):
    if 1 not in mask:
        return [0] * mask_len
    start = mask.index(1)
    end = mask_len - list(reversed(mask)).index(1)
    dist = [0] * mask_len
    for i in range(start):
        dist[i] = max(0, 1 - (start - i - 1) * decay)
    for i in range(end, mask_len):
        dist[i] = max(0, 1 - (i - end) * decay)
    return dist

class FetDataset(Dataset):
    def __init__(self, training_file_path,tokens_field,entities_field,sentence_field,label_stoi,gpu=False):
        self.gpu = gpu
        self.pad = C.PAD_INDEX
        self.entities_field = entities_field
        self.tokens_field = tokens_field
        self.sentence_field = sentence_field
        self.label_stoi = label_stoi
        self.label_size = len(label_stoi)
        # self.data = dd.read_parquet(training_file_path,engine='fastparquet')
        self.data= pd.read_parquet(training_file_path, engine="fastparquet")
    def __getitem__(self, idx):
        # data_transformed = self.data.loc[idx].compute()
        # data_transformed = data_transformed.to_dict('records')
        # record = data_transformed[0]
        record = self.data.iloc[idx]
        record_dict = {"tokens":record[self.tokens_field],"entities":record[self.entities_field],"sentence":record[self.sentence_field]}
        instance = self.process_instance(record_dict,self.label_stoi)
        # instance = ast.literal_eval(record['instance'])
        return instance

    def __len__(self):
        return len(self.data)

    def process_instance(self,inst, label_stoi):
        tokens = inst['tokens']
        tokens = [C.TOK_REPLACEMENT.get(t, t) for t in tokens]
        seq_len = len(tokens)
        char_ids = batch_to_ids([tokens])[0].tolist()
        labels_nbz, men_mask, ctx_mask, men_ids, mentions = [], [], [], [], []
        annotations = inst['entities']
        anno_num = len(annotations)
        for annotation in annotations:
            mention_id = annotation['mention_id']
            labels = annotation['labels']
            labels = [l.replace('geograpy', 'geography') for l in labels]
            start = annotation['start']
            end = annotation['end']

            men_ids.append(mention_id)
            mentions.append(annotation['mention'])
            labels = [label_stoi[l] for l in labels if l in label_stoi]
            labels_nbz.append(labels)
            men_mask.append([1 if i >= start and i < end else 0
                                for i in range(seq_len)])
            ctx_mask.append([1 if i < start or i >= end else 0
                                for i in range(seq_len)])
        return (char_ids, labels_nbz, men_mask, ctx_mask, men_ids, mentions, anno_num,
                seq_len)

    def batch_process(self, batch):
        
        # Process the batch
        seq_lens = [x[-1] for x in batch]
        max_seq_len = max(seq_lens)

        batch_char_ids = []
        batch_labels = []
        batch_men_mask = []
        batch_dist = []
        batch_ctx_mask = []
        batch_gathers = []
        batch_men_ids = []
        batch_mentions = []

        for inst_idx, inst in enumerate(batch):

            char_ids, labels, men_mask, ctx_mask, men_ids, mentions, anno_num, seq_len = inst

            # Elmo Character ids
            batch_char_ids.append(char_ids + [[self.pad] * C.ELMO_MAX_CHAR_LEN
                                              for _ in range(max_seq_len - seq_len)])
            # Instance labels
            for ls in labels:
                batch_labels.append([1 if l in ls else 0
                                     for l in range(self.label_size)])
            # mention masks
            for mask in men_mask:
                batch_men_mask.append(mask + [self.pad] * (max_seq_len - seq_len))
                batch_dist.append(mask_to_distance(mask, seq_len)
                                  + [self.pad] * (max_seq_len - seq_len))
            #context masks
            for mask in ctx_mask:
                batch_ctx_mask.append(mask + [self.pad] * (max_seq_len - seq_len))

            batch_gathers.extend([inst_idx] * anno_num)

            batch_men_ids.extend(men_ids)
            batch_mentions.extend(mentions)

        return (batch_char_ids, batch_labels, batch_men_mask, batch_ctx_mask,
                batch_dist, batch_gathers, batch_men_ids, batch_mentions)
