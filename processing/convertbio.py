from collections import defaultdict
from typing import Dict


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

def bio_to_bioul(labels):
    """Convert a sequence of BIO labels to BIOUL labels.
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
                labels_bioes.append('U-' + label[2:])
        else:
            if next_label.startswith('I-'):
                labels_bioes.append(label)
            else:
                labels_bioes.append('L-' + label[2:])
    return labels_bioes

def main():
    bioes_tags = [
                "O",
                "O",
                "O",
                "B-NORP",
                "O",
                "O",
                "O",
                "O",
                "O",
                "O",
                "B-CARDINAL",
                "B-NORP",
                "O",
                "O",
                "O",
                "O",
                "O",
                "O",
                "O",
                "O",
                "O",
                "B-CARDINAL",
                "O",
                "O",
                "O",
                "B-CARDINAL",
                "O",
                "O",
                "B-LOC",
                "I-LOC",
                "I-LOC",
                "I-LOC",
                "O",
                "O",
                "O",
                "O"
            ]

    tags  = bio_to_bioul(bioes_tags)
    print(tags)

if __name__ == '__main__':
    main()