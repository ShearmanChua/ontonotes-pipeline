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

def HAnDS_to_parquet():
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
    # task.execute_remotely()

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

    print(files[:10])

    # data_src_path = folder + "/" + file

    # dataset = Dataset.create(
    #         dataset_project=args['dataset_project'], dataset_name=args['dataset_name']
    # )
    
    # data = []
    # for line in open(data_src_path, 'r'):
    #     data.append(json.loads(line))

if __name__ == '__main__':
    HAnDS_to_parquet()