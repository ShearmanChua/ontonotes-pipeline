import os
from os import listdir
from os.path import isfile, join
from tempfile import gettempdir
from argparse import ArgumentParser

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

    from model import simple_model

    # get datset uploaded
    dataset_dict = Dataset.list_datasets(
        dataset_project=PROJECT_NAME, partial_name=args["training_dataset"], only_completed=False
    )

    datasets_obj = [
        Dataset.get(dataset_id=dataset_dict["id"]) for dataset_dict in dataset_dict
    ]

    # reverse list due to child-parent dependency, and get the first dataset_obj
    dataset_obj = datasets_obj[::-1][0]
    
    json_folder = dataset_obj.get_local_copy()

    train_file = [file for file in dataset_obj.list_files() if file=='train.parquet'][0]

    training_file_path = json_folder + "/" + train_file
    
    tags_file = [file for file in dataset_obj.list_files() if file=='ner_tags.json'][0]

    tags_file_path = json_folder + "/" + tags_file

    word_to_ix_file = [file for file in dataset_obj.list_files() if file=='word_to_ix.json'][0]

    word_to_ix_file_path = json_folder + "/" + word_to_ix_file

    simple_model.model_train(training_file_path,tags_file_path,word_to_ix_file_path,logger)


if __name__ == '__main__':
    model_training()