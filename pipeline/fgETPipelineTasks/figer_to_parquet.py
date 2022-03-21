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

def figer_to_parquet():
    PROJECT_NAME = "fgET"
    TASK_NAME = "dataset_parsing_to_parquet"
    DATASET_PROJECT = "datasets/multimodal"
    DATASET_PARTIAL_NAME = "HAnDS data"
    DATASET_NAME = "fgET HAnDS data"

    # Task.add_requirements("-rrequirements.txt")
    Task.force_requirements_env_freeze(force=True, requirements_file='requirements.txt')
    Task.add_requirements("torch")
    task = Task.init(project_name=PROJECT_NAME, task_name=TASK_NAME)
    # task.set_base_docker("nvcr.io/nvidia/pytorch:20.08-py3")
    task.set_base_docker("nvidia/cuda:11.4.0-cudnn8-devel-ubuntu20.04")
    args = {"project":PROJECT_NAME,"source_dataset":DATASET_PARTIAL_NAME,"dataset_project":DATASET_PROJECT, "dataset_name":DATASET_NAME}
    task.connect(args)
    task.execute_remotely()
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

    file = dataset_obj.list_files()[0]

    data_src_path = folder + "/" + file

    data_rows = []
    for line in open(data_src_path, 'r'):
        data_rows.append(json.loads(line))

    print("First 10 data rows:", data_rows[:10])



    dataset = Dataset.create(
            dataset_project=args['dataset_project'], dataset_name=args['dataset_name']
    )

def new_mapping(figer_data_rows):
    figer_labels_dict = dict()

    for row in figer_data_rows:
        for mention in row['mentions']: 
            for label in mention['labels']:
                if label not in figer_labels_dict:
                    figer_labels_dict[label] = 1
                else:
                    figer_labels_dict[label] += 1

    dataset_dict = Dataset.list_datasets(
        dataset_project='datasets/multimodal', partial_name='1k-WFB-g data', only_completed=False
    )

    datasets_obj = [
        Dataset.get(dataset_id=dataset_dict["id"]) for dataset_dict in dataset_dict
    ]

    # reverse list due to child-parent dependency, and get the first dataset_obj
    dataset_obj = datasets_obj[::-1][0]
    
    folder = dataset_obj.get_local_copy()

    file = dataset_obj.list_files()[0]

    data_src_path = folder + "/" + file

    HAnDS_data_rows = []
    for line in open(data_src_path, 'r'):
        HAnDS_data_rows.append(json.loads(line))

    HAnDS_labels_dict = dict()

    for row in HAnDS_data_rows:
        for mention in row['mentions']: 
            for label in mention['labels']:
                if label not in HAnDS_labels_dict:
                    HAnDS_labels_dict[label] = 1
                else:
                    HAnDS_labels_dict[label] += 1

    classes_sim = {}

    for key,value in figer_labels_dict.items():
        if key in HAnDS_labels_dict:
            classes_sim[key] = value

    classes_diff = {}

    for key,value in figer_labels_dict.items():
        if key not in HAnDS_labels_dict:
            classes_diff[key] = value

    classes_second_level = {}

    for label,value in classes_diff.items():  
        labels = label.split('/')
        labels.pop(0)
        labels = labels[-1]
        classes_second_level[label] = '/'+labels

    classes_remapped = {}

    for original,label in classes_second_level.items():
        for key,value in HAnDS_labels_dict.items():
            if label in key:
                classes_remapped[original] = key
        
    classes_remapped['/government/government']= '/organization/government'
    classes_remapped['/government']= '/organization/government'

    classes_combined = list(classes_sim.keys())
    classes_combined.extend(list(classes_remapped.values()))

if __name__ == '__main__':
    figer_to_parquet()