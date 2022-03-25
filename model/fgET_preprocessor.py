import re

from allennlp.modules.elmo import Elmo
from allennlp.modules.elmo import batch_to_ids
import torch

import model.constant as C

DIGIT_PATTERN = re.compile('\d')

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

class PreProcessor():
    def __init__(self,elmo_option,
                 elmo_weight,
                 elmo_dropout=.5,
                 gpu=False):

        self.elmo = Elmo(elmo_option, elmo_weight, 1,
                         dropout=elmo_dropout)

        #freeze Elmo model
        for param in self.elmo.parameters():
            param.requires_grad = False

        self.elmo_dim = self.elmo.get_output_dim()

        if gpu:
            self.elmo.cuda()

        # self.elmo._elmo._modules['_elmo_lstm']._elmo_lstm.stateful = False

        self.pad = C.PAD_INDEX

    def process_instance(self,inst, label_stoi):
        tokens = inst['tokens']
        sentence = inst['sentence']
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
            
        return (char_ids, labels_nbz, men_mask, ctx_mask, men_ids, mentions,sentence, anno_num,
                seq_len)

    def get_elmo_embeddings(self,elmo_ids):
        '''
        All pre-trained models expect input images normalized in the same way,
        i.e. mini-batches of 3-channel RGB images of shape (3 x H x W), where H and W are expected to be at least 224.
        The images have to be loaded in to a range of [0, 1]
        and then normalized using mean = [0.485, 0.456, 0.406] and std = [0.229, 0.224, 0.225].
        '''
        if torch.cuda.is_available():
            elmo_ids = torch.cuda.LongTensor(elmo_ids)
        else:
            elmo_ids = torch.LongTensor(elmo_ids)
        elmo_outputs = self.elmo(elmo_ids)['elmo_representations'][0]
        _, seq_len, feat_dim = elmo_outputs.size()
        gathers = gathers.unsqueeze(-1).unsqueeze(-1).expand(-1, seq_len, feat_dim)
        elmo_outputs = torch.gather(elmo_outputs, 0, gathers)

        return elmo_outputs

    def batch_process(self, batch):
            
        # Process the batch
        seq_lens = [x[-1] for x in batch]
        max_seq_len = max(seq_lens)

        batch_char_ids = []
        batch_elmo_embeddings = []
        batch_labels = []
        batch_men_mask = []
        batch_dist = []
        batch_ctx_mask = []
        batch_gathers = []
        batch_men_ids = []
        batch_mentions = []
        batch_sentences = []

        for inst_idx, inst in enumerate(batch):

            char_ids, labels, men_mask, ctx_mask, men_ids, mentions,sentence, anno_num, seq_len = inst

            # Elmo Character ids
            batch_char_ids.append(char_ids + [[self.pad] * C.ELMO_MAX_CHAR_LEN
                                                for _ in range(max_seq_len - seq_len)])
            batch_elmo_embeddings = self.get_elmo_embeddings(batch_char_ids)
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
            batch_mentions.extend(sentence)

        return (batch_elmo_embeddings, batch_labels, batch_men_mask, batch_ctx_mask,
                batch_dist, batch_gathers, batch_men_ids, batch_mentions,batch_sentences)