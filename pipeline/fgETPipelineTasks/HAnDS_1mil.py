from clearml import Task, Dataset

from tempfile import gettempdir
import os
from os import listdir
from os.path import isfile, join
import json 
import pandas as pd
from io import StringIO
import codecs
import re
import numpy as np
import gzip

from allennlp.modules.elmo import batch_to_ids

def HAnDS_1mil():
    PROJECT_NAME = "fgET"
    TASK_NAME = "HAnDS_dataset_generate_1mil"
    DATASET_PROJECT = "datasets/multimodal"
    DATASET_PARTIAL_NAME = "fgET HAnDS data"
    DATASET_NAME = "fgET HAnDS 100k preprocessed"

    # Task.add_requirements("-rrequirements.txt")
    Task.force_requirements_env_freeze(force=True, requirements_file='requirements.txt')
    Task.add_requirements("torch")
    task = Task.init(project_name=PROJECT_NAME, task_name=TASK_NAME)
    # task.set_base_docker("nvcr.io/nvidia/pytorch:20.08-py3")
    task.set_base_docker("nvidia/cuda:11.4.0-cudnn8-devel-ubuntu20.04")
    args = {"project":PROJECT_NAME,"source_dataset":DATASET_PARTIAL_NAME,"dataset_project":DATASET_PROJECT, "dataset_name":DATASET_NAME}
    task.connect(args)
    task.execute_remotely()

    import model.constant as C

    logger = task.get_logger()

    # get uploaded dataset
    dataset_dict = Dataset.list_datasets(
        dataset_project=args['dataset_project'], partial_name=args['source_dataset'], only_completed=False
    )

    datasets_obj = [
        Dataset.get(dataset_id=dataset_dict["id"]) for dataset_dict in dataset_dict
    ]

    # reverse list due to child-parent dependency, and get the first dataset_obj
    dataset_obj = datasets_obj[::-1][0]
    
    folder = dataset_obj.get_local_copy()


    files = dataset_obj.list_files()

    print("First 10 files in {} folder: ".format(args['source_dataset']) ,files[:10])

    files_required = []
    
    for i in range(0,20):
        files_required.append('train_{}.parquet'.format(str(i)))
        files_required.append('validation_{}.parquet'.format(str(i)))
        files_required.append('test_{}.parquet'.format(str(i)))
    
    files = [file for file in dataset_obj.list_files() if file in files_required]

    dataset = Dataset.create(
            dataset_project=args['dataset_project'], dataset_name=args['dataset_name']
    )

    train_df = pd.read_parquet(folder + "/" + 'train_0.parquet', engine='fastparquet')
    val_df = pd.read_parquet(folder + "/" + 'validation_0.parquet', engine='fastparquet')
    test_df = pd.read_parquet(folder + "/" + 'test_0.parquet', engine='fastparquet')

    for i in range(1,20):
        train_append = pd.read_parquet(folder + "/" + 'train_{}.parquet'.format(str(i)), engine='fastparquet')
        val_append = pd.read_parquet(folder + "/" + 'validation_{}.parquet'.format(str(i)), engine='fastparquet')
        test_append = pd.read_parquet(folder + "/" + 'test_{}.parquet'.format(str(i)), engine='fastparquet')
        train_df = pd.concat([train_df,train_append])
        val_df = pd.concat([val_df,val_append])
        test_df = pd.concat([test_df,test_append])

    print("train df:", train_df)
    print("val df:", val_df)
    print("test df:", test_df)

    train_df = train_df[~train_df.fine_grained_entities.str.len().eq(0)]
    val_df = val_df[~val_df.fine_grained_entities.str.len().eq(0)]
    test_df = test_df[~test_df.fine_grained_entities.str.len().eq(0)]

    train_df = train_df.sample(n=100000)
    val_df = val_df.sample(n=20000)
    test_df = test_df.sample(n=20000)

    train_df['instance'] = train_df['fine_grained_entities'].apply(process_instance)
    val_df['instance'] = val_df['fine_grained_entities'].apply(process_instance)
    test_df['instance'] = test_df['fine_grained_entities'].apply(process_instance)
    
    print("new train df:", train_df)
    print("new val df:", val_df)
    print("new test df:", test_df)

    train_df.to_parquet(os.path.join(gettempdir(), 'train.parquet'),engine='fastparquet')
    val_df.to_parquet(os.path.join(gettempdir(), 'validation.parquet'),engine='fastparquet')
    test_df.to_parquet(os.path.join(gettempdir(), 'test.parquet'),engine='fastparquet')

    dataset.add_files(os.path.join(gettempdir(), 'train.parquet'))
    dataset.add_files(os.path.join(gettempdir(), 'validation.parquet'))
    dataset.add_files(os.path.join(gettempdir(), 'test.parquet'))

    dataset.upload(output_url='s3://experiment-logging/multimodal')
    dataset.finalize()

def process_instance(inst):
    labels_file_path = get_clearml_file_path('datasets/multimodal','fgET data','ner_tags.json')

    with open(labels_file_path) as json_file:
        label_stoi = json.load(json_file)

    label_size = len(label_stoi)
    print('Label size: {}'.format(len(label_stoi)))

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

def get_clearml_file_path(dataset_project,dataset_name,file_name):

    # get uploaded dataset from clearML
    dataset_dict = Dataset.list_datasets(
        dataset_project=dataset_project, partial_name=dataset_name, only_completed=False
    )

    datasets_obj = [
        Dataset.get(dataset_id=dataset_dict["id"]) for dataset_dict in dataset_dict
    ]

    # reverse list due to child-parent dependency, and get the first dataset_obj
    dataset_obj = datasets_obj[::-1][0]
    
    folder = dataset_obj.get_local_copy()

    file = [file for file in dataset_obj.list_files() if file==file_name][0]

    file_path = folder + "/" + file

    return file_path

if __name__ == '__main__':
    HAnDS_1mil()
