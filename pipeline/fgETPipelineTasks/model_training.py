import time
import os
from os import listdir
from os.path import isfile, join
from tempfile import gettempdir

from clearml import Task, Dataset

def model_training():

    PROJECT_NAME = "ontonotes"
    TASK_NAME = "model_training"
    JSON_PARTIAL_NAME = "training"

    Task.force_requirements_env_freeze(force=True, requirements_file='requirements.txt')
    Task.add_requirements("torch")
    task = Task.init(project_name=PROJECT_NAME, task_name=TASK_NAME)
    # task.set_base_docker("nvcr.io/nvidia/pytorch:20.08-py3")
    task.set_base_docker("nvidia/cuda:11.4.0-cudnn8-devel-ubuntu20.04")
    args = {'training_dataset':JSON_PARTIAL_NAME}
    task.connect(args)
    task.execute_remotely()

    logger = task.get_logger()

    # ============= imports =============
    from collections import defaultdict
    from pytorch_pretrained_bert import BertAdam
    from model.fgET_model import fgET
    from data import BufferDataset

if __name__ == '__main__':
    model_training()