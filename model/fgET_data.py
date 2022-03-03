import json
import re
import torch
import model.constant as C
from torch.utils.data import Dataset
from allennlp.modules.elmo import batch_to_ids
import dask.dataframe as dd

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

def numberize(inst, label_stoi):
    tokens = inst['tokens']
    tokens = [C.TOK_REPLACEMENT.get(t, t) for t in tokens]
    seq_len = len(tokens)
    char_ids = batch_to_ids([tokens])[0].tolist()
    labels_nbz, men_mask, ctx_mask, men_ids = [], [], [], []
    annotations = inst['annotations']
    anno_num = len(annotations)
    for annotation in annotations:
        mention_id = annotation['mention_id']
        labels = annotation['labels']
        labels = [l.replace('geograpy', 'geography') for l in labels]
        start = annotation['start']
        end = annotation['end']

        men_ids.append(mention_id)
        labels = [label_stoi[l] for l in labels if l in label_stoi]
        labels_nbz.append(labels)
        men_mask.append([1 if i >= start and i < end else 0
                            for i in range(seq_len)])
        ctx_mask.append([1 if i < start or i >= end else 0
                            for i in range(seq_len)])
    return (char_ids, labels_nbz, men_mask, ctx_mask, men_ids, anno_num,
            seq_len)

class FetDataset(Dataset):
    def __init__(self, training_file_path,word_tokens_field,tags_field,gpu=False):
        self.gpu = gpu
        self.word_tokens_field = word_tokens_field
        self.tags_field = tags_field
        self.data = dd.read_parquet(training_file_path,engine='fastparquet')
    def __getitem__(self, idx):
        data_transformed = self.data.loc[idx].compute()
        data_transformed = data_transformed.to_dict('records')
        record = data_transformed[0]
        sample = (record[self.word_tokens_field],record[self.tags_field])
        return sample

    def __len__(self):
        return len(self.data)